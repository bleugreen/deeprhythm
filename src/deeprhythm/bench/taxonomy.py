"""Error taxonomy and confidence calibration for instrumented tempo benchmarks."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RELATIONS = (("correct", 1.0), ("half", 0.5), ("double", 2.0), ("third", 1 / 3), ("triple", 3.0))


def metrical_relation(prediction, reference, tolerance=0.04):
    """Classify a prediction by its closest recognized metrical relation."""
    matches = [
        (abs(prediction / (reference * factor) - 1.0), name)
        for name, factor in RELATIONS
        if abs(prediction / (reference * factor) - 1.0) <= tolerance
    ]
    return min(matches)[1] if matches else "other"


def expected_calibration_error(confidences, outcomes, bins=10):
    """Compute equal-width expected calibration error."""
    confidences = np.asarray(confidences, dtype=float)
    outcomes = np.asarray(outcomes, dtype=float)
    if confidences.shape != outcomes.shape or confidences.ndim != 1 or confidences.size == 0:
        raise ValueError("confidence and outcome arrays must be equally sized and non-empty")
    if np.any((confidences < 0) | (confidences > 1)):
        raise ValueError("confidences must be between zero and one")
    edges = np.linspace(0, 1, bins + 1)
    result = []
    ece = 0.0
    for index in range(bins):
        selected = (confidences >= edges[index]) & (
            confidences <= edges[index + 1] if index == bins - 1 else confidences < edges[index + 1]
        )
        if not selected.any():
            continue
        confidence = float(confidences[selected].mean())
        accuracy = float(outcomes[selected].mean())
        count = int(selected.sum())
        ece += count / confidences.size * abs(accuracy - confidence)
        result.append({"lower": edges[index], "upper": edges[index + 1], "count": count,
                       "confidence": confidence, "accuracy": accuracy})
    return {"ece": ece, "bins": result}


def analyze(manifest_rows, detail_rows, tolerance=0.04):
    """Join manifest and details rows by resolved audio path and summarize errors."""
    details = {str(Path(row["filename"]).expanduser().resolve()): row for row in detail_rows}
    grouped = defaultdict(lambda: defaultdict(int))
    joined = []
    for manifest in manifest_rows:
        path = str(Path(manifest["audio_path"]).expanduser().resolve())
        if path not in details:
            raise ValueError(f"missing details for {path}")
        detail = details[path]
        reference = float(manifest["tempo"])
        prediction = float(detail["bpm"])
        relation = metrical_relation(prediction, reference, tolerance)
        tempo_range = f"{int(reference // 20) * 20}-{int(reference // 20) * 20 + 19}"
        genre = str(manifest.get("genre", manifest.get("style", "unknown")))
        grouped["relation"][relation] += 1
        grouped["tempo_range"][tempo_range + ":" + relation] += 1
        grouped["genre"][genre + ":" + relation] += 1
        joined.append({**manifest, "prediction": prediction, "relation": relation,
                       "confidence": float(detail["confidence"])})
    calibration = expected_calibration_error(
        [row["confidence"] for row in joined], [row["relation"] == "correct" for row in joined]
    )
    return {"tracks": len(joined), "counts": {key: dict(value) for key, value in grouped.items()},
            "calibration": calibration}


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyze instrumented DeepRhythm benchmark errors")
    parser.add_argument("manifest")
    parser.add_argument("details")
    parser.add_argument("-o", "--output")
    parser.add_argument("--tolerance", type=float, default=0.04)
    args = parser.parse_args(argv)
    result = analyze(_read_jsonl(args.manifest), _read_jsonl(args.details), args.tolerance)
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()