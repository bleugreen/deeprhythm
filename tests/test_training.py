import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from deeprhythm.bench.tempo import tempo_accuracy
from deeprhythm.train.cache import ClipDataset, HcqmCache, build_hcqm_cache, manifest_key
from deeprhythm.train.sampling import TempoBalancedSampler, stretch_audio_and_tempo
from deeprhythm.train.trainer import evaluate_validation, mixed_supervision_loss
from deeprhythm.utils import bpm_to_class


def fake_hcqm(value, clips=3):
    return np.full((clips, 6, 240, 8), value, dtype=np.float32)


def test_manifest_key_invalidates_annotation_and_checksum_changes():
    row = {"audio_path": "audio.wav", "tempo": 120, "split": "train", "md5": "a"}
    assert manifest_key(row) == manifest_key(dict(reversed(list(row.items()))))
    assert manifest_key(row) != manifest_key({**row, "tempo": 121})
    assert manifest_key(row) != manifest_key({**row, "md5": "b"})
    assert manifest_key(row) != manifest_key({**row, "split": "val"})


def test_cache_is_incremental_and_dataset_is_memory_backed(tmp_path):
    rows = [
        {"audio_path": "one.wav", "tempo": 120, "split": "train", "md5": "a"},
        {"audio_path": "two.wav", "tempo": 90, "split": "val", "md5": "b"},
    ]
    calls = []

    def extract(path):
        calls.append(path)
        return fake_hcqm(len(calls), clips=7 if path == "one.wav" else 2)

    index = build_hcqm_cache(rows, tmp_path, extractor=extract)
    build_hcqm_cache(rows, tmp_path, extractor=extract)
    assert calls == ["one.wav", "two.wav"]
    assert index["manifest_hash"]
    train = ClipDataset(tmp_path, "train")
    validation = ClipDataset(tmp_path, "val", return_tempo=True)
    assert len(train) == 5  # historical edge trimming, without its off-by-one bug
    assert len(validation) == 2
    clip, label = train[0]
    assert clip.shape == (6, 240, 8)
    assert label.item() == bpm_to_class(120)
    assert validation[0][1].item() == 90


def test_cache_rebuild_drops_rows_removed_from_manifest(tmp_path):
    rows = [
        {"audio_path": "one.wav", "tempo": 120, "split": "train"},
        {"audio_path": "two.wav", "tempo": 90, "split": "val"},
    ]
    build_hcqm_cache(rows, tmp_path, extractor=lambda _path: fake_hcqm(1))
    build_hcqm_cache(rows[:1], tmp_path, extractor=lambda _path: fake_hcqm(1))
    assert len(HcqmCache(tmp_path).load_index()["entries"]) == 1
    assert len(ClipDataset(tmp_path, "val")) == 0


def test_canonical_fold_manifests_build_train_and_validation_sets(tmp_path):
    rows = [
        {"audio_path": "train.wav", "tempo": 120, "fold": "train", "md5": "a"},
        {"audio_path": "validation.wav", "tempo": 90, "fold": "val", "md5": "b"},
    ]
    build_hcqm_cache(rows, tmp_path, extractor=lambda _path: fake_hcqm(1))
    assert len(ClipDataset(tmp_path, "train")) == 3
    assert len(ClipDataset(tmp_path, "val")) == 3


def test_dataset_can_return_training_domain(tmp_path):
    rows = [{"audio_path": "replay.wav", "tempo": 120, "fold": "train", "dataset": "replay"}]
    build_hcqm_cache(rows, tmp_path, extractor=lambda _path: fake_hcqm(1))
    _clip, _label, domain = ClipDataset(tmp_path, "train", return_domain=True)[0]
    assert domain == "replay"


def test_mixed_supervision_distills_only_preservation_domains():
    labels = torch.tensor([0, 1])
    student = torch.zeros((2, 3), requires_grad=True)
    teacher = torch.tensor([[100.0, 0.0, 0.0], [0.0, 100.0, 0.0]])
    loss = mixed_supervision_loss(
        student,
        labels,
        ["human", "replay"],
        teacher,
        preservation_domains=("replay",),
        distillation_weight=2.0,
        temperature=1.0,
    )
    loss.backward()
    assert student.grad[0].argmin().item() == 0
    assert student.grad[1].argmin().item() == 1


def test_corrupt_cache_version_is_rejected(tmp_path):
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "index.json").write_text(json.dumps({"version": 99, "entries": {}}))
    with pytest.raises(ValueError, match="version"):
        HcqmCache(tmp_path).load_index()


def test_tempo_balanced_sampler_equalizes_bin_mass():
    sampler = TempoBalancedSampler([60, 61, 62, 120], bin_width=10, num_samples=4)
    weights = sampler.weights.tolist()
    assert weights[:3] == pytest.approx([1 / 3] * 3)
    assert weights[3] == pytest.approx(1.0)
    assert sum(weights[:3]) == pytest.approx(weights[3])


def test_tempo_balanced_sampler_equalizes_domain_mass():
    sampler = TempoBalancedSampler([60, 61, 120, 90], domains=["public", "public", "public", "soulseek"])
    weights = sampler.weights.tolist()
    assert sum(weights[:3]) == pytest.approx(weights[3])


def test_time_stretch_relabels_and_enforces_model_range():
    audio = np.sin(np.linspace(0, 8 * np.pi, 1000)).astype(np.float32)
    stretched, tempo = stretch_audio_and_tempo(audio, 100, 1.25)
    assert tempo == 125
    assert len(stretched) == 800
    with pytest.raises(ValueError, match="outside"):
        stretch_audio_and_tempo(audio, 280, 1.25)


class FixedModel(nn.Module):
    def forward(self, inputs):
        logits = torch.full((len(inputs), 256), -100.0)
        logits[torch.arange(len(inputs)), inputs[:, 0].long()] = 100.0
        return logits


def test_validation_acc2_reuses_benchmark_scoring():
    reference_tempos = torch.tensor([120.0, 100.0])
    predicted_classes = torch.tensor([bpm_to_class(60), bpm_to_class(100)], dtype=torch.float32).unsqueeze(1)
    dataset = [(predicted_classes[i], reference_tempos[i], f"track-{i}") for i in range(2)]
    loader = DataLoader(dataset, batch_size=2)
    metrics = evaluate_validation(FixedModel(), loader, nn.CrossEntropyLoss(), torch.device("cpu"), 0.04)
    predicted_tempos = [float(int(60)), float(int(100))]
    expected = tempo_accuracy(predicted_tempos, reference_tempos.tolist(), 0.04)
    assert metrics["acc1"] == expected["acc1"] == 0.5
    assert metrics["acc2"] == expected["acc2"] == 1.0


def test_validation_aggregates_probabilities_per_track():
    reference = torch.tensor(120.0)
    slow = float(bpm_to_class(60))
    correct = float(bpm_to_class(120))
    dataset = [
        (torch.tensor([slow]), reference, "same-track"),
        (torch.tensor([correct]), reference, "same-track"),
        (torch.tensor([correct]), reference, "same-track"),
    ]
    metrics = evaluate_validation(
        FixedModel(), DataLoader(dataset, batch_size=3), nn.CrossEntropyLoss(), torch.device("cpu"), 0.04
    )
    assert metrics["acc1"] == 1.0