"""Fine-tuning loop deliberately limited to train and validation folds."""

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from deeprhythm.bench.tempo import tempo_accuracy
from deeprhythm.model.frame_cnn import DeepRhythmModel
from deeprhythm.train.cache import ClipDataset, HcqmCache
from deeprhythm.train.sampling import TempoBalancedSampler
from deeprhythm.utils import bpm_to_class, class_to_bpm


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 1e-4
    early_stopping_patience: int = 5
    balance_bin_width: float = 10.0
    tolerance: float = 0.04
    balance_domains: bool = False
    freeze_feature_extractor: bool = False
    distillation_weight: float = 0.0
    distillation_temperature: float = 2.0
    preservation_domains: tuple[str, ...] = ()


def mixed_supervision_loss(
    student_logits,
    labels,
    domains,
    teacher_logits,
    *,
    preservation_domains,
    distillation_weight,
    temperature,
):
    """Combine hard human supervision with teacher preservation on declared domains."""
    preserve = torch.tensor(
        [domain in preservation_domains for domain in domains], device=student_logits.device, dtype=torch.bool
    )
    losses = []
    if (~preserve).any():
        losses.append(F.cross_entropy(student_logits[~preserve], labels[~preserve]))
    if preserve.any():
        scaled_student = F.log_softmax(student_logits[preserve] / temperature, dim=1)
        scaled_teacher = F.softmax(teacher_logits[preserve] / temperature, dim=1)
        distillation = F.kl_div(scaled_student, scaled_teacher, reduction="batchmean") * temperature**2
        losses.append(distillation_weight * distillation)
    if not losses:
        raise ValueError("batch has neither supervised nor preservation examples")
    return sum(losses)


def evaluate_validation(model, loader, criterion, device, tolerance):
    model.eval()
    losses, by_track = [], {}
    with torch.no_grad():
        for batch in loader:
            if len(batch) != 3:
                raise ValueError("validation loader must return clip, tempo, and track identity")
            inputs, tempos, track_ids = batch
            inputs, tempos = inputs.to(device), tempos.to(device)
            labels = torch.tensor([bpm_to_class(x) for x in tempos.tolist()], device=device)
            outputs = model(inputs)
            losses.append(criterion(outputs, labels).item())
            probabilities = torch.softmax(outputs, dim=1).cpu()
            for probability, tempo, track_id in zip(probabilities, tempos.cpu().tolist(), track_ids):
                record = by_track.setdefault(track_id, {"probabilities": [], "tempo": tempo})
                if record["tempo"] != tempo:
                    raise ValueError(f"track {track_id} has inconsistent reference tempos")
                record["probabilities"].append(probability)
    if not losses:
        raise ValueError("validation split has no clips")
    predictions = []
    references = []
    for record in by_track.values():
        predicted_class = torch.stack(record["probabilities"]).mean(0).argmax().item()
        predictions.append(class_to_bpm(predicted_class))
        references.append(record["tempo"])
    metrics = tempo_accuracy(predictions, references, tolerance=tolerance)
    metrics["loss"] = sum(losses) / len(losses)
    return metrics


def fit(cache_dir, output_path, *, config=TrainingConfig(), start_weights=None, device=None):
    """Fit on ``train`` and select on ``val``; never opens or evaluates ``test``."""
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    use_distillation = bool(config.preservation_domains and config.distillation_weight > 0)
    train_set = ClipDataset(cache_dir, "train", return_tempo=False, return_domain=use_distillation)
    validation_set = ClipDataset(cache_dir, "val", return_tempo=True, return_track=True)
    if not train_set or not validation_set:
        raise ValueError("cache requires non-empty train and val splits")
    sampler = TempoBalancedSampler(
        train_set.tempos,
        domains=train_set.domains if config.balance_domains else None,
        bin_width=config.balance_bin_width,
    )
    train_loader = DataLoader(train_set, batch_size=config.batch_size, sampler=sampler)
    validation_loader = DataLoader(validation_set, batch_size=config.batch_size)
    model = DeepRhythmModel().to(device)
    if start_weights:
        model.load_state_dict(torch.load(start_weights, map_location=device, weights_only=True))
    teacher = None
    if use_distillation:
        if not start_weights:
            raise ValueError("preservation distillation requires start_weights")
        teacher = DeepRhythmModel().to(device)
        teacher.load_state_dict(torch.load(start_weights, map_location=device, weights_only=True))
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad = False
    if config.freeze_feature_extractor:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.fc2.parameters():
            parameter.requires_grad = True
    criterion = nn.CrossEntropyLoss()
    trainable_parameters = (parameter for parameter in model.parameters() if parameter.requires_grad)
    optimizer = Adam(trainable_parameters, lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=2, factor=0.5)
    best_loss, stale, history = float("inf"), 0, []
    output_path = Path(output_path)
    for epoch in range(config.epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            inputs, labels = batch[:2]
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            if use_distillation:
                domains = batch[2]
                with torch.no_grad():
                    teacher_outputs = teacher(inputs)
                loss = mixed_supervision_loss(
                    outputs,
                    labels,
                    domains,
                    teacher_outputs,
                    preservation_domains=config.preservation_domains,
                    distillation_weight=config.distillation_weight,
                    temperature=config.distillation_temperature,
                )
            else:
                loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        validation = evaluate_validation(model, validation_loader, criterion, device, config.tolerance)
        scheduler.step(validation["loss"])
        record = {"epoch": epoch + 1, "train_loss": sum(train_losses) / len(train_losses), **validation}
        history.append(record)
        if validation["loss"] < best_loss:
            best_loss, stale = validation["loss"], 0
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "config": asdict(config),
                "manifest_hash": HcqmCache(cache_dir).load_index().get("manifest_hash"),
                "validation": validation,
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(checkpoint, output_path)
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                break
    return history