"""Transcribe the prepared test sets with one model and record the hypothesis and the compute time of every clip.

    python -m bench.run --backend whisper --model openai/whisper-large-v3-turbo
    python -m bench.run --backend murmel --model theaifactorybv/murmel-flash-m-v5 --sets fleurs --max-clips 50

Timing: clips go through the model one at a time (batch 1), the way a user transcribes a file; the clock covers
feature extraction, the model and decoding, not reading the file from disk. CUDA is synchronised before the clock
stops. The first 5 clips of the first set are run once untimed to warm up kernels and caches.
RTF = compute seconds / audio seconds (lower is faster); RTFx = 1 / RTF.

Output: results/<model>/<set>.jsonl (id, reference, hypothesis, audio seconds, compute seconds) and
results/<model>/run.json (model, backend, settings, hardware, software versions).
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .sets import SETS, load

ROOT = Path(__file__).resolve().parents[1]


def hardware() -> dict:
    info = {"python": platform.python_version(), "machine": platform.machine(), "cpu": platform.processor() or platform.machine()}
    try:
        import torch
        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["cuda"] = torch.version.cuda
    except ImportError:
        pass
    return info


def sync():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", required=True, choices=["murmel", "whisper", "nemo", "voxtral"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    ap.add_argument("--max-clips", type=int, default=0, help="evenly spaced subset per set (0 = all)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=str(ROOT / "results"))
    a = ap.parse_args()

    backend = importlib.import_module(f"bench.backends.{a.backend}").Backend(a.model, device=a.device)
    out = Path(a.out) / a.model.split("/")[-1]; out.mkdir(parents=True, exist_ok=True)
    meta = {"model": a.model, "backend": a.backend, "settings": backend.settings, "hardware": hardware(),
            "versions": backend.versions(), "date": time.strftime("%Y-%m-%d"), "max_clips": a.max_clips}
    (out / "run.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)

    warm = True
    for s in a.sets:
        rows = load(s, a.max_clips)
        if warm:
            for r in rows[:5]:
                backend.transcribe(sf.read(r["path"], dtype="float32")[0])
            warm = False
        done = {}
        f = out / f"{s}.jsonl"
        if f.exists():                                               # resumable: keep clips already transcribed
            done = {j["id"]: j for j in map(json.loads, open(f, encoding="utf-8"))}
        t_audio = t_comp = 0.0
        with open(f, "a", encoding="utf-8") as fo:
            for i, r in enumerate(rows):
                if r["id"] in done:
                    continue
                wave = sf.read(r["path"], dtype="float32")[0]
                sync(); t0 = time.perf_counter()
                hyp = backend.transcribe(wave)
                sync(); dt = time.perf_counter() - t0
                t_audio += len(wave) / 16000; t_comp += dt
                fo.write(json.dumps({"id": r["id"], "ref": r["text"], "hyp": hyp, "audio_s": round(len(wave) / 16000, 3),
                                     "compute_s": round(dt, 4)}, ensure_ascii=False) + "\n")
                if (i + 1) % 200 == 0:
                    fo.flush()
                    print(f"  {s}: {i + 1}/{len(rows)}  RTFx {t_audio / max(t_comp, 1e-9):.0f}", flush=True)
        print(f"{a.model} {s}: {len(rows)} clips, RTFx {t_audio / max(t_comp, 1e-9):.0f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
