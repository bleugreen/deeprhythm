"""Build and validate the canonical research dataset partitions."""

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FOLDS = ("train", "val", "test")
GTZAN_HISTORICAL_EXCLUSIONS = {"jazz/jazz.00054.wav", "reggae/reggae.00086.wav"}
BALLROOM_REPLICAS = (
    ("Quickstep/Albums-AnaBelen_Veneo-11.wav", "Quickstep/Albums-Chrisanne2-12.wav"),
    ("ChaChaCha/Albums-Fire-08.wav", "Samba/Albums-Fire-09.wav"),
    ("ChaChaCha/Albums-Latin_Jam2-05.wav", "ChaChaCha/Albums-Latin_Jam2-13.wav"),
    ("Waltz/Albums-Secret_Garden-01.wav", "Waltz/Media-104705.wav"),
    ("Rumba-International/Albums-AnaBelen_Veneo-03.wav", "Rumba-International/Albums-AnaBelen_Veneo-15.wav"),
    ("Waltz/Albums-Ballroom_Magic-03.wav", "Waltz/Albums-Ballroom_Magic-18.wav"),
    ("ChaChaCha/Albums-Latin_Jam-04.wav", "ChaChaCha/Albums-Latin_Jam-13.wav"),
    ("Rumba-International/Albums-Latin_Jam-08.wav", "Rumba-International/Albums-Latin_Jam-14.wav"),
    ("Samba/Albums-Latin_Jam-06.wav", "Samba/Albums-Latin_Jam-15.wav"),
    ("Samba/Albums-Latin_Jam2-02.wav", "Samba/Albums-Latin_Jam2-14.wav"),
    ("Rumba-International/Albums-Latin_Jam2-07.wav", "Rumba-International/Albums-Latin_Jam2-15.wav"),
    ("ChaChaCha/Albums-Latin_Jam3-02.wav", "ChaChaCha/Media-103414.wav"),
    ("ChaChaCha/Media-103402.wav", "ChaChaCha/Media-103415.wav"),
)


def digest(path):
    value = hashlib.md5()  # nosec: manifest identity, not a security primitive
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_tempo(path):
    value = float(path.read_text().strip())
    if value <= 0:
        raise ValueError(f"non-positive tempo in {path}")
    return value


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in sorted(rows, key=lambda r: r["audio_path"])))


def validate_partitions(rows):
    """Reject paths, group keys, or duplicate fingerprints crossing partitions."""
    required = {"dataset", "audio_path", "md5", "tempo", "group_key", "fold", "fingerprint", "provenance"}
    seen_paths = set()
    ownership = {"group_key": {}, "fingerprint": {}}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"manifest row missing {sorted(missing)}")
        if row["fold"] not in FOLDS or float(row["tempo"]) <= 0:
            raise ValueError(f"invalid fold or tempo in {row['audio_path']}")
        path_key = (row["dataset"], row["audio_path"])
        if path_key in seen_paths:
            raise ValueError(f"duplicate audio path: {path_key}")
        seen_paths.add(path_key)
        for field, by_value in ownership.items():
            key = (row["dataset"], row[field])
            previous = by_value.setdefault(key, row["fold"])
            if previous != row["fold"]:
                raise ValueError(f"{field} {row[field]!r} spans {previous} and {row['fold']}")
    return True


def _gtzan_index(path):
    result = {}
    for line in path.read_text(errors="replace").splitlines():
        if line.startswith("#") or " ::: " not in line:
            continue
        parts = line.split(" ::: ", 2)
        filename, artist = parts[:2]
        artist = artist.strip()
        result[filename] = artist if artist.strip(": ") else "?"
    return result


def build_gtzan(root, sources):
    index = _gtzan_index(sources / "index.txt")
    assignment = {}
    for fold, filename in (("train", "train_filtered.txt"), ("val", "valid_filtered.txt"), ("test", "test_filtered.txt")):
        for line in (sources / filename).read_text().splitlines():
            assignment[line.strip()] = fold
    rows = []
    for relative, fold in sorted(assignment.items()):
        if relative in GTZAN_HISTORICAL_EXCLUSIONS:
            continue
        genre, filename = relative.split("/")
        audio = root / "gtzan_genre" / "genres" / relative
        track_number = Path(filename).stem[-5:]
        annotation = root / "gtzan_tempo_beat-main" / "tempo" / f"gtzan_{genre}_{track_number}.bpm"
        artist = index.get(filename)
        if not artist or artist == "?":
            group = f"track:{filename}"
        else:
            group = f"artist:{artist.casefold()}"
        md5 = digest(audio)
        rows.append({"dataset": "gtzan", "audio_path": str(audio.relative_to(root)), "md5": md5,
                     "tempo": read_tempo(annotation), "group_key": group, "fold": fold,
                     "source_fold": f"kereliuk-{fold}", "fingerprint": f"md5:{md5}",
                     "provenance": "Sturm GTZAN fedc781; gtzan_tempo_beat"})
    priority = {"train": 0, "val": 1, "test": 2}
    artist_fold = {}
    for row in rows:
        if row["group_key"].startswith("artist:"):
            previous = artist_fold.get(row["group_key"], "train")
            artist_fold[row["group_key"]] = max((previous, row["fold"]), key=priority.get)
    for row in rows:
        if row["group_key"] in artist_fold:
            row["fold"] = artist_fold[row["group_key"]]
    return rows


def build_giantsteps(root):
    repository = next(root.glob("giantsteps-tempo-dataset-*"))
    official = {}
    for number in range(1, 11):
        target = "train" if number <= 7 else "val" if number == 8 else "test"
        for filename in (repository / "splits" / f"fold{number:02d}.txt").read_text().splitlines():
            official[filename] = (target, number)
    rows = []
    for filename, (fold, number) in sorted(official.items()):
        annotation = repository / "annotations_v2" / "tempo" / filename.replace(".mp3", ".bpm")
        audio = root / "audio_canonical" / filename
        if not annotation.exists() or not audio.exists():
            continue
        tempo = float(annotation.read_text().strip())
        if tempo <= 0:
            continue
        expected = (repository / "md5" / filename.replace(".mp3", ".md5")).read_text().split()[0]
        actual = digest(audio)
        if actual != expected:
            raise ValueError(f"GiantSteps checksum mismatch: {filename}")
        track = filename.split(".", 1)[0]
        rows.append({"dataset": "giantsteps", "audio_path": str(audio.relative_to(root)), "md5": actual,
                     "tempo": tempo, "group_key": f"track:{track}", "fold": fold,
                     "source_fold": f"official-fold{number:02d}", "fingerprint": f"md5:{actual}",
                     "provenance": "GiantSteps tempo dataset 0b7d47b; v2 crowd annotation"})
    return rows


def _ballroom_group(stem):
    album = re.fullmatch(r"Albums-(.+)-\d+", stem)
    if album:
        return f"album:{album.group(1).casefold()}"
    media = re.fullmatch(r"Media-(\d+)", stem)
    if media:
        return f"media-prefix:{media.group(1)[:-2]}"
    return f"track:{stem.casefold()}"


def _assign_ballroom(groups):
    targets = {"train": 0.7, "val": 0.1, "test": 0.2}
    totals = Counter(key for tracks in groups.values() for key in tracks)
    counts = {fold: Counter() for fold in FOLDS}
    result = {}
    for group, tracks in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        contribution = Counter(tracks)
        def cost(fold):
            return sum(((counts[fold][key] + contribution[key]) / max(totals[key], 1) - targets[fold]) ** 2 for key in contribution)
        fold = min(FOLDS, key=lambda name: (cost(name), sum(counts[name].values()) / targets[name], name))
        result[group] = fold
        counts[fold].update(contribution)
    return result


def build_ballroom(root):
    excluded = {duplicate.casefold() for _keeper, duplicate in BALLROOM_REPLICAS}
    candidates = []
    groups = defaultdict(list)
    for audio in sorted((root / "audio").glob("*/*.wav")):
        relative = str(audio.relative_to(root / "audio"))
        if relative.casefold() in excluded:
            continue
        annotation = next((root / "annotations" / "tempo").glob(f"**/{audio.stem}.bpm"))
        tempo = read_tempo(annotation)
        group = _ballroom_group(audio.stem)
        stratum = (audio.parent.name.casefold(), int(tempo // 20))
        groups[group].append(stratum)
        candidates.append((audio, relative, tempo, group))
    assignment = _assign_ballroom(groups)
    rows = []
    for audio, relative, tempo, group in candidates:
        md5 = digest(audio)
        rows.append({"dataset": "ballroom", "audio_path": str(audio.relative_to(root)), "md5": md5,
                     "tempo": tempo, "group_key": group, "fold": assignment[group], "source_fold": "deeprhythm-v0.8",
                     "fingerprint": f"md5:{md5}", "genre": audio.parent.name,
                     "provenance": "Ballroom B_1.0; CPJKU tempo and replica annotations"})
    return rows


def historical_rows(dataset, root):
    rows = []
    if dataset == "ballroom":
        files = sorted((root / "audio").glob("*/*.wav"))
        tempo = lambda p: read_tempo(next((root / "annotations" / "tempo").glob(f"**/{p.stem}.bpm")))
    elif dataset == "gtzan":
        files = sorted((root / "gtzan_genre" / "genres").glob("*/*.wav"))
        files = [p for p in files if str(p.relative_to(root / "gtzan_genre" / "genres")) not in GTZAN_HISTORICAL_EXCLUSIONS]
        tempo = lambda p: read_tempo(root / "gtzan_tempo_beat-main" / "tempo" / f"gtzan_{p.parent.name}_{p.stem[-5:]}.bpm")
    else:
        repository = next(root.glob("giantsteps-tempo-dataset-*"))
        files = sorted((root / "audio_canonical").glob("*.mp3"))
        files = [p for p in files if (repository / "annotations_v2" / "tempo" / p.name.replace(".mp3", ".bpm")).exists()
                 and float((repository / "annotations_v2" / "tempo" / p.name.replace(".mp3", ".bpm")).read_text()) > 0]
        tempo = lambda p: read_tempo(repository / "annotations_v2" / "tempo" / p.name.replace(".mp3", ".bpm"))
    for path in files:
        rows.append({"audio_path": str(path.relative_to(root)), "tempo": tempo(path), "historical": "deeprhythm-0.7-full-set"})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets-root", type=Path, default=Path("~/mir_datasets").expanduser())
    parser.add_argument("--output", type=Path, default=Path("data/splits"))
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.validate_only:
        rows = [json.loads(line) for path in args.output.glob("*/train.jsonl") for line in path.read_text().splitlines()]
        rows += [json.loads(line) for pattern in ("*/val.jsonl", "*/test.jsonl") for path in args.output.glob(pattern) for line in path.read_text().splitlines()]
        validate_partitions(rows)
        return
    sources = args.output / "sources" / "gtzan"
    built = {
        "giantsteps": build_giantsteps(args.datasets_root / "giantsteps_tempo"),
        "gtzan": build_gtzan(args.datasets_root / "gtzan_genre", sources),
        "ballroom": build_ballroom(args.datasets_root / "ballroom" / "B_1.0"),
    }
    for dataset, rows in built.items():
        validate_partitions(rows)
        for fold in FOLDS:
            write_rows(args.output / dataset / f"{fold}.jsonl", [row for row in rows if row["fold"] == fold])
    write_rows(args.output / "archive-v0.7" / "giantsteps.jsonl", historical_rows("giantsteps", args.datasets_root / "giantsteps_tempo"))
    write_rows(args.output / "archive-v0.7" / "gtzan.jsonl", historical_rows("gtzan", args.datasets_root / "gtzan_genre"))
    write_rows(args.output / "archive-v0.7" / "ballroom.jsonl", historical_rows("ballroom", args.datasets_root / "ballroom" / "B_1.0"))


if __name__ == "__main__":
    main()