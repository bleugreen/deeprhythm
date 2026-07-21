import hashlib
import json
from pathlib import Path


def test_frozen_research_results_match_committed_manifests_and_weights():
    root = Path(__file__).parents[1]
    result = json.loads((root / "docs/research/results/final-test-results.json").read_text())
    for dataset in ("giantsteps", "gtzan", "ballroom"):
        manifest = root / "data/splits" / dataset / "test.jsonl"
        assert hashlib.sha256(manifest.read_bytes()).hexdigest() == result["manifest_hashes"][dataset]
    weights = root / "weights/deeprhythm-v0.8-research.pth"
    assert hashlib.sha256(weights.read_bytes()).hexdigest() == result["weights_sha256"]


def test_v07_apples_to_apples_rows_are_frozen_on_canonical_tests():
    root = Path(__file__).parents[1]
    result = json.loads((root / "docs/research/results/final-test-results.json").read_text())
    expected = {"giantsteps": 0.7099236641221374, "gtzan": 0.6900958466453674, "ballroom": 0.5970149253731343}
    for dataset, acc1 in expected.items():
        baseline = result["datasets"][dataset]["baseline"]["uncorrected"]
        assert baseline["tracks"] == len((root / "data/splits" / dataset / "test.jsonl").read_text().splitlines())
        assert baseline["metrics_4pct"]["acc1"] == acc1
