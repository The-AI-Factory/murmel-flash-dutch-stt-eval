"""The open Dutch test sets, downloaded once into data/<set>/ as 16 kHz mono FLAC + manifest.jsonl.

Every backend reads only these plain files, so no backend environment needs the `datasets` library and every model
sees exactly the same audio and references.

    python -m bench.sets                     # all sets
    python -m bench.sets --sets fleurs       # one set
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import subprocess
import tarfile
from pathlib import Path

import numpy as np
import soundfile as sf

SR = 16000
DATA = Path(__file__).resolve().parents[1] / "data"

# name -> where it comes from. All are openly licensed; Common Voice comes from a complete CC0 mirror of release 17
# (Mozilla no longer hosts Common Voice on the Hugging Face Hub).
SETS = {
    "fleurs": {"title": "FLEURS nl_nl test", "repo": "google/fleurs", "licence": "CC BY 4.0",
               "file": "parquet-data/nl_nl/test-00000-of-00001.parquet", "audio": "audio", "text": "transcription",
               "id": "id"},
    "common_voice": {"title": "Common Voice 17.0 nl test", "repo": "fsicoli/common_voice_17_0", "licence": "CC0 1.0",
                     "tar": "audio/nl/test/nl_test_0.tar", "tsv": "transcript/nl/test.tsv"},
    "mls": {"title": "Multilingual LibriSpeech Dutch test", "repo": "facebook/multilingual_librispeech",
            "licence": "CC BY 4.0", "file": "dutch/test-00000-of-00001.parquet", "audio": "audio", "text": "transcript",
            "id": "id"},
    "ifadv": {"title": "IFADV conversations", "repo": "zenodo.org/records/14906857", "licence": "AGPL-3.0",
              "zenodo": "https://zenodo.org/records/14906857/files/{}?download=1"},
    "voxpopuli": {"title": "VoxPopuli nl test", "repo": "facebook/voxpopuli", "licence": "CC0 1.0",
                  "file": "nl/test-00000-of-00001.parquet", "audio": "audio", "text": "normalized_text",
                  "id": "audio_id"},
}


def decode(data: bytes, name: str = "") -> np.ndarray:
    """Any audio bytes -> 16 kHz mono float32 (soundfile first, ffmpeg for mp3/opus and anything else)."""
    try:
        x, sr = sf.read(io.BytesIO(data), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(1)
        if sr == SR:
            return x
    except Exception:
        pass
    out = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"],
                         input=data, capture_output=True, check=True).stdout
    return np.frombuffer(out, dtype=np.float32).copy()


def _write(name: str, rows) -> int:
    d = DATA / name; (d / "audio").mkdir(parents=True, exist_ok=True)
    n = 0
    with open(d / "manifest.jsonl.tmp", "w") as m:
        for cid, wave, text in rows:
            cid = str(cid).replace("/", "_")
            sf.write(d / "audio" / f"{cid}.flac", wave, SR, subtype="PCM_16")
            m.write(json.dumps({"id": cid, "audio": f"audio/{cid}.flac", "text": text, "duration": round(len(wave) / SR, 3)},
                               ensure_ascii=False) + "\n")
            n += 1
    (d / "manifest.jsonl.tmp").rename(d / "manifest.jsonl")
    return n


# CGN transcription conventions in IFADV: *d/*v/*a/*u/*z/*n/*t mark dialect, foreign, broken-off, ... words (the word
# itself is kept); xxx = unintelligible (the utterance is left out); ggg = laughter and other sounds (removed).
_CGN_MARK = re.compile(r"\*[a-z]\b")


# spoken-form spellings in the CGN transcripts that every model writes in standard spelling; only unambiguous ones
_CGN_SPELLING = {"'t": "het", "'k": "ik", "m'n": "mijn", "z'n": "zijn", "'n": "een"}
# Praat's backslash codes for accented letters (\e' = é, \e" = ë, ...)
_PRAAT = {"'": "\u0301", "`": "\u0300", '"': "\u0308", "^": "\u0302", ",": "\u0327"}


def _praat_accents(t: str) -> str:
    import unicodedata
    return unicodedata.normalize("NFC", re.sub(r"\\([a-zA-Z])(['`\",^])", lambda m: m.group(1) + _PRAAT[m.group(2)], t))


def _ifadv_text(t: str):
    if "xxx" in t.lower():
        return None
    t = _praat_accents(t)
    t = " ".join(_CGN_SPELLING.get(w.lower(), w) for w in t.split())
    t = _CGN_MARK.sub("", t)
    t = re.sub(r"\bggg\b", " ", t)
    t = " ".join(t.split())
    return t if re.search(r"\w", t) else None


def _textgrid(path: Path) -> dict:
    lines = [l.strip() for l in open(path, encoding="latin-1").read().splitlines()]
    tiers, i = {}, 0
    while i < len(lines):
        if lines[i] == '"IntervalTier"':
            name, n, j, iv = lines[i + 1].strip('"'), int(lines[i + 4]), i + 5, []
            for _ in range(n):
                a, b, t = float(lines[j]), float(lines[j + 1]), lines[j + 2].strip('"'); j += 3
                if t:
                    iv.append((a, b, t))
            tiers[name] = iv; i = j
        else:
            i += 1
    return tiers


def _ifadv_rows(spec):
    """IFADV: 20 annotated dialogues, 15 min each, two head-mounted microphones (stereo). One clip per annotated
    utterance from that speaker's own microphone (the channel that is relatively loudest while they speak, decided
    once per dialogue), 0.15 s margin. Utterances that overlap the other speaker for more than 30 % are left out: the
    other voice dominates them while the reference holds one speaker. The transcript version (original, corrected or
    time-shifted) whose intervals best match the speech energy is used."""
    import urllib.request, zipfile
    raw = DATA / "_ifadv_raw"; raw.mkdir(parents=True, exist_ok=True)
    for z in ("Annotations.zip", "AudioFLAC.zip"):
        if not (raw / z.replace(".zip", "")).exists():
            if not (raw / z).exists():
                print(f"  downloading {z} from Zenodo", flush=True)
                urllib.request.urlretrieve(spec["zenodo"].format(z), raw / f"{z}.part"); (raw / f"{z}.part").rename(raw / z)
            with zipfile.ZipFile(raw / z) as zf:
                zf.extractall(raw)
            (raw / z).unlink()
    orts = sorted((raw / "Annotations" / "ort").glob("*.ort"))
    for audio in sorted((raw / "AudioFLAC").glob("DVA*.flac")):
        cands = [o for o in orts if o.name.startswith(audio.stem + "_")]
        if not cands:
            continue                                                  # 4 of the 24 recordings are not annotated
        x, sr = sf.read(audio, dtype="float32", always_2d=True)
        frame = sr // 100
        energy = (x[: len(x) // frame * frame].reshape(-1, frame, x.shape[1]) ** 2).mean(1).max(1)
        def fit(o):
            tg = _textgrid(o); inside = np.zeros(len(energy), bool)
            for tier in ("spreker1", "spreker2"):
                for a, b, _ in tg.get(tier, []):
                    inside[int(a * 100):int(b * 100)] = True
            if not inside.any() or inside.all():
                return -np.inf
            return np.log(energy[inside].mean() + 1e-12) - np.log(energy[~inside].mean() + 1e-12)
        tg = _textgrid(max(cands, key=fit))
        tiers = ("spreker1", "spreker2")
        mask = {t: np.zeros(len(x) // frame + 1, bool) for t in tiers}
        for t in tiers:
            for a, b, _ in tg.get(t, []):
                mask[t][int(a * 100):int(b * 100)] = True
        def level(t):                                                 # mean log energy per channel while t speaks alone
            alone = mask[t][: len(energy)] & ~mask[[o for o in tiers if o != t][0]][: len(energy)]
            fr = x[: len(energy) * frame].reshape(len(energy), frame, -1)[alone]
            return np.log((fr ** 2).mean(1) + 1e-12).mean(0)
        rel = level("spreker1") - level("spreker2")
        chan = {"spreker1": int(np.argmax(rel)), "spreker2": int(np.argmin(rel))}
        for tier in tiers:
            other = mask[[o for o in tiers if o != tier][0]]
            for k, (a, b, t) in enumerate(tg.get(tier, [])):
                text = _ifadv_text(t)
                if text is None or b - a < 0.4 or other[int(a * 100):int(b * 100)].mean() > 0.3:
                    continue
                s0, s1 = max(0, int((a - 0.15) * sr)), min(len(x), int((b + 0.15) * sr))
                wave = x[s0:s1, chan[tier]]
                if sr != SR:
                    wave = decode(_wav_bytes(wave, sr))
                yield f"{audio.stem}-{tier[-1]}-{k:04d}", wave, text


def _wav_bytes(wave, sr) -> bytes:
    buf = io.BytesIO(); sf.write(buf, wave, sr, format="WAV", subtype="FLOAT"); return buf.getvalue()


def prepare(name: str) -> int:
    from huggingface_hub import hf_hub_download
    spec = SETS[name]
    if (DATA / name / "manifest.jsonl").exists():
        return sum(1 for _ in open(DATA / name / "manifest.jsonl"))
    if "zenodo" in spec:
        return _write(name, _ifadv_rows(spec))
    if "file" in spec:                                              # one parquet file with audio bytes inside
        import pyarrow.parquet as pq
        t = pq.read_table(hf_hub_download(spec["repo"], spec["file"], repo_type="dataset"))
        def rows():
            for r in t.to_pylist():
                a = r[spec["audio"]]
                yield r[spec["id"]], decode(a["bytes"], a.get("path") or ""), r[spec["text"]]
        return _write(name, rows())
    tsv = hf_hub_download(spec["repo"], spec["tsv"], repo_type="dataset")
    text = {r["path"]: r["sentence"] for r in csv.DictReader(open(tsv, encoding="utf-8"), delimiter="\t", quoting=csv.QUOTE_NONE)}
    tar = hf_hub_download(spec["repo"], spec["tar"], repo_type="dataset")
    def rows():
        with tarfile.open(tar) as tf:
            for m in tf:
                base = m.name.rsplit("/", 1)[-1]
                if m.isfile() and base in text:
                    yield base.rsplit(".", 1)[0], decode(tf.extractfile(m).read(), base), text[base]
    return _write(name, rows())


def load(name: str, max_clips: int = 0) -> list[dict]:
    rows = [json.loads(l) for l in open(DATA / name / "manifest.jsonl", encoding="utf-8")]
    rows.sort(key=lambda r: r["id"])
    if max_clips and len(rows) > max_clips:                          # evenly spaced, the same subset for every model
        step = len(rows) / max_clips
        rows = [rows[int(i * step)] for i in range(max_clips)]
    for r in rows:
        r["path"] = str(DATA / name / r["audio"])
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sets", nargs="+", default=list(SETS))
    a = ap.parse_args()
    for s in a.sets:
        print(f"{s}: {prepare(s)} clips in {DATA / s}", flush=True)


if __name__ == "__main__":
    main()
