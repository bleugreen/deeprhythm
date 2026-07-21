"""Disk-backed HCQM cache keyed by immutable manifest content."""

import hashlib
import json
import os
from pathlib import Path
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from deeprhythm.audio_proc.hcqm import compute_hcqm, make_kernels
from deeprhythm.utils import bpm_to_class, load_and_split_audio

CACHE_VERSION = 1
MANIFEST_ID_FIELDS = ("audio_path", "filename", "md5", "checksum", "tempo", "split")


def _canonical_row(row: Mapping) -> dict:
    value = {key: row[key] for key in MANIFEST_ID_FIELDS if key in row}
    if "audio_path" not in value and "filename" not in value:
        raise ValueError("manifest row requires audio_path or filename")
    if "tempo" not in value or float(value["tempo"]) <= 0:
        raise ValueError("manifest row requires a positive tempo")
    return value


def manifest_key(row: Mapping, *, cache_version: int = CACHE_VERSION) -> str:
    """Return a stable cache key; changing relevant manifest data invalidates it."""
    payload = {"cache_version": cache_version, "row": _canonical_row(row)}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def manifest_hash(rows: Iterable[Mapping]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(manifest_key(row).encode())
        digest.update(b"\n")
    return digest.hexdigest()


class HcqmCache:
    """A directory of memory-mappable NumPy arrays plus a canonical JSON index."""

    def __init__(self, root):
        self.root = Path(root)
        self.arrays = self.root / "arrays"
        self.index_path = self.root / "index.json"

    def path_for(self, key: str) -> Path:
        return self.arrays / f"{key}.npy"

    def load_index(self) -> dict:
        if not self.index_path.exists():
            return {"version": CACHE_VERSION, "entries": {}}
        index = json.loads(self.index_path.read_text())
        if index.get("version") != CACHE_VERSION:
            raise ValueError("unsupported HCQM cache version")
        return index

    def store(self, key: str, clips: np.ndarray) -> None:
        self.arrays.mkdir(parents=True, exist_ok=True)
        destination = self.path_for(key)
        temporary = destination.with_suffix(f".{os.getpid()}.tmp")
        with temporary.open("wb") as handle:
            np.save(handle, np.asarray(clips, dtype=np.float32), allow_pickle=False)
        os.replace(temporary, destination)

    def write_index(self, index: dict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        temporary = self.index_path.with_suffix(f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(index, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, self.index_path)


def _default_extract(path: str, specs) -> np.ndarray:
    audio = load_and_split_audio(path).to(device=specs[0].wsin.device)
    with torch.no_grad():
        hcqm = compute_hcqm(audio, *specs).permute(0, 3, 1, 2)
    return hcqm.cpu().numpy()


def build_hcqm_cache(
    rows: Sequence[Mapping],
    cache_dir,
    *,
    device: Optional[str] = None,
    extractor: Optional[Callable[[str], np.ndarray]] = None,
) -> dict:
    """Materialize missing HCQMs and atomically update the manifest index.

    ``extractor`` is an injection point for tests and custom preprocessing. Its
    result must have shape ``(clips, 6, 240, 8)``.
    """
    cache = HcqmCache(cache_dir)
    index = cache.load_index()
    specs = None if extractor else make_kernels(device=device)
    for source_row in rows:
        row = _canonical_row(source_row)
        key = manifest_key(row)
        path = str(Path(row.get("audio_path", row.get("filename"))).expanduser())
        destination = cache.path_for(key)
        if not destination.exists():
            clips = extractor(path) if extractor else _default_extract(path, specs)
            clips = np.asarray(clips)
            if clips.ndim != 4 or tuple(clips.shape[1:]) != (6, 240, 8):
                raise ValueError("HCQM extractor must return (clips, 6, 240, 8)")
            if len(clips) == 0:
                raise ValueError(f"no clips extracted from {path}")
            cache.store(key, clips)
        index["entries"][key] = {
            "path": str(destination.relative_to(cache.root)),
            "tempo": float(row["tempo"]),
            "split": str(row.get("split", "train")),
            "source": path,
            "clips": int(np.load(destination, mmap_mode="r").shape[0]),
        }
    index["manifest_hash"] = manifest_hash(rows)
    cache.write_index(index)
    return index


class ClipDataset(Dataset):
    """Lazy, memory-mapped HCQM clips from one declared manifest split."""

    def __init__(self, cache_dir, split="train", *, trim_edges=True, return_tempo=False):
        self.cache = HcqmCache(cache_dir)
        self.return_tempo = return_tempo
        self.entries = []
        for key, entry in self.cache.load_index()["entries"].items():
            if entry["split"] != split:
                continue
            count = int(entry["clips"])
            indices = range(1, count - 1) if trim_edges and count > 5 else range(count)
            self.entries.extend((key, clip, float(entry["tempo"])) for clip in indices)

    @property
    def tempos(self):
        return [entry[2] for entry in self.entries]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        key, clip_index, tempo = self.entries[index]
        clips = np.load(self.cache.path_for(key), mmap_mode="r")
        clip = torch.from_numpy(np.array(clips[clip_index], copy=True)).float()
        if self.return_tempo:
            return clip, torch.tensor(tempo, dtype=torch.float32)
        return clip, torch.tensor(bpm_to_class(tempo), dtype=torch.long)