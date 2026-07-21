"""Train/validation-only tooling for metrical-level correction experiments."""

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from deeprhythm.bench.taxonomy import expected_calibration_error, metrical_relation
from deeprhythm.bench.tempo import tempo_accuracy
from deeprhythm.correction import (
    FACTORS,
    FactorClassifier,
    clip_disagreement,
    global_threshold,
    identity,
    select_parameter,
    target_factor,
    top_k_ratio,
)


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def join_rows(manifest, details, features=None):
    """Join rows by path and reject held-out rows at the fitting boundary."""
    if any(row.get("fold") == "test" for row in manifest):
        raise ValueError("held-out test folds cannot be used by correction selection tooling")
    details_by_path = {str(Path(row["filename"]).expanduser().resolve()): row for row in details}
    features_by_path = {
        str(Path(row["filename"]).expanduser().resolve()): row["features"] for row in (features or [])
    }
    joined = []
    for row in manifest:
        path = str(Path(row["audio_path"]).expanduser().resolve())
        if path not in details_by_path:
            raise ValueError(f"missing inference details for {path}")
        joined.append({**row, "detail": details_by_path[path], "features": features_by_path.get(path)})
    return joined


def evaluate_rows(rows, predictions, confidences=None, latency_seconds=0.0):
    references = [float(row["tempo"]) for row in rows]
    report = {"tracks": len(rows), "latency": {
        "total_seconds": latency_seconds,
        "added_milliseconds_per_track": latency_seconds * 1000 / len(rows),
    }}
    for tolerance in (0.02, 0.04):
        report[f"metrics_{int(tolerance * 100)}pct"] = tempo_accuracy(predictions, references, tolerance)
    relations = [metrical_relation(prediction, reference, 0.04)
                 for prediction, reference in zip(predictions, references)]
    report["taxonomy"] = dict(sorted((name, relations.count(name)) for name in set(relations)))
    slices = defaultdict(list)
    for row, prediction in zip(rows, predictions):
        label = str(row.get("genre", row.get("style", "unknown")))
        slices[label].append((prediction, float(row["tempo"])))
    report["slices"] = {
        label: {"tracks": len(values), **tempo_accuracy(*zip(*values), tolerance=0.04)}
        for label, values in sorted(slices.items())
    }
    if confidences is not None:
        report["calibration"] = expected_calibration_error(
            confidences, [relation == "correct" for relation in relations]
        )
    return report


def run_experiment(train_rows, val_rows):
    """Fit all variants on train/validation data and return validation reports."""
    val_details = [row["detail"] for row in val_rows]
    val_refs = [float(row["tempo"]) for row in val_rows]
    grids = {
        "global_threshold": np.arange(50, 151, 5),
        "top_k_ratio": np.geomspace(0.01, 10.0, 31),
        "clip_disagreement": np.linspace(0.125, 1.0, 8),
    }
    functions = {"global_threshold": global_threshold, "top_k_ratio": top_k_ratio,
                 "clip_disagreement": clip_disagreement}
    output = {"identity": evaluate_rows(val_rows, [identity(detail) for detail in val_details])}
    for name, grid in grids.items():
        parameter = select_parameter(val_details, val_refs, name, grid)
        start = time.perf_counter()
        predictions = [functions[name](detail, parameter) for detail in val_details]
        elapsed = time.perf_counter() - start
        output[name] = {
            "selected_parameter": parameter,
            **evaluate_rows(val_rows, predictions, latency_seconds=elapsed),
        }

    if all(row["features"] is not None for row in train_rows + val_rows):
        classifier = FactorClassifier().fit(
            [row["features"] for row in train_rows],
            [target_factor(float(row["detail"]["bpm"]), float(row["tempo"])) for row in train_rows],
        )
        start = time.perf_counter()
        probabilities = classifier.predict_proba([row["features"] for row in val_rows])
        predictions = [float(row["detail"]["bpm"]) * FACTORS[index]
                       for row, index in zip(val_rows, probabilities.argmax(axis=1))]
        elapsed = time.perf_counter() - start
        output["learned"] = {
            "model": classifier.to_dict(),
            **evaluate_rows(val_rows, predictions, probabilities.max(axis=1), elapsed),
        }
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--train-details", required=True)
    parser.add_argument("--val-manifest", required=True)
    parser.add_argument("--val-details", required=True)
    parser.add_argument("--train-features")
    parser.add_argument("--val-features")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args(argv)
    train = join_rows(_read(args.train_manifest), _read(args.train_details),
                      _read(args.train_features) if args.train_features else None)
    val = join_rows(_read(args.val_manifest), _read(args.val_details),
                    _read(args.val_features) if args.val_features else None)
    result = run_experiment(train, val)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()