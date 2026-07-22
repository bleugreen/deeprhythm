"""Prepare and serve a local metrical-level annotation harness."""

# ruff: noqa: E501  # Embedded self-contained HTML/JavaScript is intentionally compact.

import argparse
import hashlib
import json
import mimetypes
import os
import sys
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

import librosa
import numpy as np
import soundfile as sf

AUDIO_SUFFIXES = {".flac", ".mp3", ".m4a", ".ogg", ".wav", ".aif", ".aiff"}
LABELS = {"half", "correct", "double", "other", "skip"}


def metrical_grids(beats, duration):
    """Return half/current/double click grids relative to one excerpt."""
    current = np.asarray([beat for beat in beats if 0 <= beat < duration], dtype=float)
    half = current[::2]
    if len(current) > 1:
        midpoints = (current[:-1] + current[1:]) / 2
        double = np.sort(np.concatenate([current, midpoints]))
    else:
        double = current.copy()
    return {"half": half, "current": current, "double": double}


def active_excerpt(audio, sample_rate, duration=15.0):
    """Choose the duration-sized window with the greatest onset activity."""
    samples = min(len(audio), int(duration * sample_rate))
    if len(audio) <= samples:
        return 0.0, np.pad(audio, (0, samples - len(audio)))
    hop = 512
    onset = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop)
    window = max(1, int(duration * sample_rate / hop))
    energy = np.convolve(onset, np.ones(window), mode="valid")
    frame = int(np.argmax(energy))
    # Onset frames are centered around an analysis window; step back one FFT radius
    # so the transient that selected the excerpt is not clipped off its front.
    start = max(0, frame * hop - 1024)
    return start / sample_rate, audio[start : start + samples]


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class AnnotationStore:
    def __init__(self, root):
        self.root = Path(root)
        self.manifest_path = self.root / "manifest.jsonl"
        self.annotations_path = self.root / "annotations.jsonl"

    def manifest(self):
        return load_jsonl(self.manifest_path)

    def annotations(self):
        return load_jsonl(self.annotations_path)

    def completed_ids(self):
        return {row["id"] for row in self.annotations()}

    def append(self, item_id, label):
        if label not in LABELS:
            raise ValueError(f"unknown label: {label}")
        if item_id not in {row["id"] for row in self.manifest()}:
            raise ValueError("unknown manifest item")
        row = {"id": item_id, "label": label, "annotated_at": datetime.now(timezone.utc).isoformat()}
        with self.annotations_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        return row


def _prediction_rows(path):
    rows = load_jsonl(path)
    output = []
    for row in rows:
        filename = row.get("filename", row.get("audio_path"))
        bpm = row.get("bpm", row.get("deeprhythm_v07"))
        if filename and bpm:
            output.append((str(Path(filename).expanduser().resolve()), float(bpm)))
    return output


def _phasefinder(phasefinder_path, bpm_holder):
    sys.path.insert(0, str(Path(phasefinder_path).expanduser().resolve()))
    import phasefinder.predictor as predictor_module

    class InjectedTempoPredictor:
        def __init__(self, *_args, **_kwargs):
            pass

        def predict(self, _path, include_confidence=False):
            result = (bpm_holder[0], 1.0)
            return result if include_confidence else result[0]

    predictor_module.DeepRhythmPredictor = InjectedTempoPredictor
    return predictor_module.Phasefinder(quiet=True)


def prepare(predictions, output, phasefinder_path, limit=None):
    """Generate resumable excerpt and click assets from prediction JSONL."""
    output = Path(output)
    assets = output / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    store = AnnotationStore(output)
    existing = {row["id"]: row for row in store.manifest()}
    holder = [120.0]
    phasefinder = _phasefinder(phasefinder_path, holder)
    prepared = 0
    for filename, bpm in _prediction_rows(predictions):
        if Path(filename).suffix.lower() not in AUDIO_SUFFIXES or not Path(filename).exists():
            continue
        item_id = hashlib.sha256(filename.encode()).hexdigest()[:16]
        if item_id in existing:
            continue
        holder[0] = bpm
        beat_times = np.asarray(phasefinder.predict(filename), dtype=float)
        audio, sample_rate = librosa.load(filename, sr=22050, mono=True)
        offset, excerpt = active_excerpt(audio, sample_rate)
        relative_beats = beat_times - offset
        grids = metrical_grids(relative_beats, len(excerpt) / sample_rate)
        item_dir = assets / item_id
        item_dir.mkdir(exist_ok=True)
        sf.write(item_dir / "audio.flac", excerpt, sample_rate)
        click_paths = {}
        for name, times in grids.items():
            click = librosa.clicks(times=times, sr=sample_rate, length=len(excerpt))
            path = item_dir / f"{name}.flac"
            sf.write(path, click, sample_rate)
            click_paths[name] = f"/assets/{item_id}/{name}.flac"
        row = {
            "id": item_id,
            "filename": filename,
            "sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest(),
            "predicted_bpm": bpm,
            "excerpt_offset_seconds": offset,
            "excerpt_duration_seconds": len(excerpt) / sample_rate,
            "audio": f"/assets/{item_id}/audio.flac",
            "clicks": click_paths,
        }
        with store.manifest_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
        existing[item_id] = row
        prepared += 1
        print(f"prepared {prepared}: {Path(filename).name}", flush=True)
        if limit and prepared >= limit:
            break
    return prepared


HTML = r"""<!doctype html><meta charset="utf-8"><title>DeepRhythm metrical annotation</title>
<style>
body{font:16px system-ui;margin:0;background:#111;color:#eee}main{max-width:850px;margin:7vh auto;padding:28px}
.muted{color:#999}.card{background:#1b1b1b;border:1px solid #333;border-radius:14px;padding:24px}
button{font:inherit;color:#eee;background:#292929;border:1px solid #555;border-radius:9px;padding:12px;margin:5px}
button:hover,.active{background:#485bff}kbd{background:#333;padding:3px 7px;border-radius:5px}h1{font-size:25px}
</style><main><h1>Metrical-level annotation</h1><p id="progress" class="muted"></p><section class="card">
<h2 id="name">Loading…</h2><p><strong id="bpm"></strong> <span id="offset" class="muted"></span></p>
<p>Preview click: <button data-grid="half">H · ½×</button><button data-grid="current" class="active">C · predicted</button><button data-grid="double">D · 2×</button></p>
<p><button id="play">Space · Play / pause</button></p><hr>
<p>How should the <em>tempo estimate</em> change?</p>
<button data-label="half">1 · Needs 2× (model half)</button><button data-label="correct">2 · Correct</button>
<button data-label="double">3 · Needs ½× (model double)</button><button data-label="other">4 · Other</button>
<button data-label="skip">S · Skip</button></section></main><script>
let item=null,grid='current',audio=new Audio(),click=new Audio();
async function next(){let r=await fetch('/api/next'),d=await r.json();item=d.item;
 document.querySelector('#progress').textContent=`${d.completed} / ${d.total} complete`;
 if(!item){document.querySelector('#name').textContent='All done';return}
 document.querySelector('#name').textContent=item.filename.split('/').pop();document.querySelector('#bpm').textContent=`${item.predicted_bpm} BPM`;
 document.querySelector('#offset').textContent=`excerpt at ${item.excerpt_offset_seconds.toFixed(1)}s`;audio.src=item.audio;setGrid('current')}
function setGrid(g){grid=g;if(!item)return;let was=!audio.paused,t=audio.currentTime;click.pause();click.src=item.clicks[g];click.currentTime=t;
 document.querySelectorAll('[data-grid]').forEach(b=>b.classList.toggle('active',b.dataset.grid===g));if(was)click.play()}
function toggle(){if(!item)return;if(audio.paused){if(audio.ended){audio.currentTime=0;click.currentTime=0}click.currentTime=audio.currentTime;audio.play();click.play()}else{audio.pause();click.pause()}}
async function label(value){audio.pause();click.pause();await fetch('/api/annotate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:item.id,label:value})});next()}
audio.addEventListener('seeked',()=>click.currentTime=audio.currentTime);audio.addEventListener('pause',()=>click.pause());
document.querySelector('#play').onclick=toggle;document.querySelectorAll('[data-grid]').forEach(b=>b.onclick=()=>setGrid(b.dataset.grid));document.querySelectorAll('[data-label]').forEach(b=>b.onclick=()=>label(b.dataset.label));
addEventListener('keydown',e=>{if(e.repeat)return;let labels={'1':'half','2':'correct','3':'double','4':'other','s':'skip'};if(e.key===' '){e.preventDefault();toggle()}else if(labels[e.key.toLowerCase()])label(labels[e.key.toLowerCase()]);else if('hcd'.includes(e.key.toLowerCase()))setGrid({h:'half',c:'current',d:'double'}[e.key.toLowerCase()])});next();
</script>"""


def handler(store):
    class Handler(BaseHTTPRequestHandler):
        def _json(self, value, status=200):
            body = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path == "/":
                body = HTML.encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif path == "/api/next":
                rows, done = store.manifest(), store.completed_ids()
                item = next((row for row in rows if row["id"] not in done), None)
                self._json({"item": item, "completed": len(done), "total": len(rows)})
            elif path.startswith("/assets/"):
                relative = Path(path.removeprefix("/assets/"))
                target = (store.root / "assets" / relative).resolve()
                assets = (store.root / "assets").resolve()
                if assets not in target.parents or not target.is_file():
                    self.send_error(404)
                    return
                body = target.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", mimetypes.guess_type(target)[0] or "application/octet-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self.send_error(404)

        def do_POST(self):
            if urlparse(self.path).path != "/api/annotate":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(size))
                self._json(store.append(payload["id"], payload["label"]), HTTPStatus.CREATED)
            except (KeyError, ValueError, json.JSONDecodeError) as error:
                self._json({"error": str(error)}, HTTPStatus.BAD_REQUEST)

        def log_message(self, _format, *_args):
            return

    return Handler


def serve(output, host="127.0.0.1", port=8765):
    store = AnnotationStore(output)
    server = ThreadingHTTPServer((host, port), handler(store))
    print(f"annotation harness: http://{host}:{port}")
    server.serve_forever()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prep = subparsers.add_parser("prepare")
    prep.add_argument("--predictions", required=True)
    prep.add_argument("--output", required=True)
    prep.add_argument("--phasefinder-path", default="~/projects/phasefinder")
    prep.add_argument("--limit", type=int)
    web = subparsers.add_parser("serve")
    web.add_argument("--output", required=True)
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8765)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        prepare(args.predictions, args.output, args.phasefinder_path, args.limit)
    else:
        serve(args.output, args.host, args.port)


if __name__ == "__main__":
    main()