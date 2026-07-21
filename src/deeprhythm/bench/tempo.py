import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import librosa
import numpy as np

from deeprhythm.batch_infer import main as run_deeprhythm
from deeprhythm.utils import get_device

OCTAVE_FACTORS = (1.0, 0.5, 2.0, 1.0 / 3.0, 3.0)


def tempo_accuracy(predictions, references, tolerance=0.02):
    """Return the standard tempo Acc1 and metrical-level-tolerant Acc2."""
    predicted = np.asarray(predictions, dtype=float)
    reference = np.asarray(references, dtype=float)
    if predicted.shape != reference.shape or predicted.ndim != 1:
        raise ValueError("predictions and references must be equally sized one-dimensional sequences")
    if predicted.size == 0:
        raise ValueError("at least one prediction is required")
    if not 0 < tolerance < 1:
        raise ValueError("tolerance must be between zero and one")
    if np.any(predicted <= 0) or np.any(reference <= 0):
        raise ValueError("tempos must be positive")

    relative_error = np.abs(predicted / reference - 1.0)
    acc1 = relative_error <= tolerance
    acc2 = np.any(
        np.stack([np.abs(predicted / (reference * factor) - 1.0) <= tolerance for factor in OCTAVE_FACTORS]),
        axis=0,
    )
    return {"acc1": float(acc1.mean()), "acc2": float(acc2.mean())}


def load_manifest(path):
    """Load JSONL rows containing unique audio paths and positive reference tempos."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    audio_paths = [str(Path(row["audio_path"]).expanduser().resolve()) for row in rows]
    references = [float(row["tempo"]) for row in rows]
    if len(set(audio_paths)) != len(audio_paths):
        raise ValueError("manifest audio paths must be unique")
    if any(tempo <= 0 for tempo in references):
        raise ValueError("manifest tempos must be positive")
    return audio_paths, references


def predict_librosa(audio_paths, workers=8):
    def predict(path):
        audio, sample_rate = librosa.load(path, sr=22050, mono=True)
        tempo, _ = librosa.beat.beat_track(y=audio, sr=sample_rate)
        return float(np.asarray(tempo).reshape(-1)[0])

    with ThreadPoolExecutor(max_workers=workers) as executor:
        return list(executor.map(predict, audio_paths))


def predict_deeprhythm(audio_paths, output_path, device, workers=8, batch_size=128, details_path=None):
    output_path = Path(output_path)
    output_path.unlink(missing_ok=True)
    if details_path is not None:
        Path(details_path).unlink(missing_ok=True)
    run_deeprhythm(
        audio_paths,
        n_workers=workers,
        max_len_batch=batch_size,
        data_path=str(output_path),
        device=device,
        conf=True,
        quiet=True,
        details_path=str(details_path) if details_path is not None else None,
    )
    by_path = {}
    for line in output_path.read_text().splitlines():
        row = json.loads(line)
        by_path[str(Path(row["filename"]).resolve())] = float(row["bpm"])
    missing = set(audio_paths) - by_path.keys()
    if missing:
        raise RuntimeError(f"DeepRhythm skipped {len(missing)} manifest files")
    return [by_path[path] for path in audio_paths]


def evaluate(predictions, references, elapsed_seconds, tolerance=0.04):
    results = {
        "tracks": len(references),
        "elapsed_seconds": elapsed_seconds,
        "batched_milliseconds_per_track": elapsed_seconds * 1000 / len(references),
    }
    results["tolerance"] = tolerance
    results.update(tempo_accuracy(predictions, references, tolerance))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="Benchmark DeepRhythm and a Librosa control on a tempo manifest")
    parser.add_argument("manifest", help="JSONL with audio_path and tempo fields")
    parser.add_argument("--output-dir", default="benchmark-results")
    parser.add_argument("--device", default=get_device())
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tolerance", type=float, default=0.04)
    args = parser.parse_args(argv)

    audio_paths, references = load_manifest(args.manifest)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    start = time.perf_counter()
    predictions = predict_deeprhythm(
        audio_paths, output_dir / "deeprhythm-predictions.jsonl", args.device, args.workers, args.batch_size,
        output_dir / "deeprhythm-details.jsonl"
    )
    results["deeprhythm"] = evaluate(predictions, references, time.perf_counter() - start, args.tolerance)

    start = time.perf_counter()
    predictions = predict_librosa(audio_paths, args.workers)
    results["librosa"] = evaluate(predictions, references, time.perf_counter() - start, args.tolerance)

    result_path = output_dir / "summary.json"
    result_path.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()