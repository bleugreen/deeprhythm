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


def test_packaged_v08_qualification_matches_bundled_weights():
    root = Path(__file__).parents[1]
    result = json.loads((root / "docs/research/results/production-v08-qualification.json").read_text())

    assert result["tracks"] == 798
    assert len(result["predictions"]) == result["tracks"]
    assert result["summary"]["canonical-test"]["overall"]["acc1"] == 0.7820069204152249
    for filename, digest in result["weights"].items():
        weights = root / "src/deeprhythm/weights" / filename
        assert hashlib.sha256(weights.read_bytes()).hexdigest() == digest
    bundle = root / "src/deeprhythm/weights/v0.8-bundle.json"
    assert hashlib.sha256(bundle.read_bytes()).hexdigest() == result["bundle_manifest_sha256"]
    assert all(row["equal"] for row in result["cpu_accelerator_parity"])
