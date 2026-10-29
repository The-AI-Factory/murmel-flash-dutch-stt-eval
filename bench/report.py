"""Score every results/<model>/<set>.jsonl and write results/RESULTS.md + results/results.json.

WER = word edit distance / reference words over the whole set (corpus WER), after bench.normalize on both sides.
Average = unweighted mean over the sets a model has (each set counts equally).
Pooled = errors over reference words of all sets together (every word counts the same; big sets weigh more).
RTFx = audio seconds / compute seconds over all clips of all sets (higher is faster).

    python -m bench.report
"""
from __future__ import annotations

import json
from pathlib import Path

import jiwer

from .normalize import normalize
from .sets import SETS

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
PUBLISHER = {"theaifactorybv": "The AI Factory", "openai": "OpenAI", "nvidia": "NVIDIA", "mistralai": "Mistral AI"}


def score(f: Path) -> dict:
    rows = [json.loads(l) for l in open(f, encoding="utf-8")]
    refs, hyps = [], []
    for r in rows:
        ref = normalize(r["ref"])
        if ref:                                                      # a reference that normalises to nothing has no words to score
            refs.append(ref); hyps.append(normalize(r["hyp"]))
    o = jiwer.process_words(refs, hyps)
    return {"clips": len(rows), "wer": 100 * o.wer, "errors": o.substitutions + o.deletions + o.insertions,
            "words": sum(len(r.split()) for r in refs), "audio_s": sum(r["audio_s"] for r in rows),
            "compute_s": sum(r["compute_s"] for r in rows), "subs": o.substitutions, "dels": o.deletions, "ins": o.insertions}


def main():
    table = {}
    for d in sorted(p for p in RES.iterdir() if p.is_dir()):
        run = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else {}
        sets = {s: score(d / f"{s}.jsonl") for s in SETS if (d / f"{s}.jsonl").exists()}
        if not sets:
            continue
        a = sum(v["audio_s"] for v in sets.values()); c = sum(v["compute_s"] for v in sets.values())
        table[run.get("model", d.name)] = {"sets": sets, "avg_wer": sum(v["wer"] for v in sets.values()) / len(sets),
                                           "pooled_wer": 100 * sum(v["errors"] for v in sets.values())
                                                         / max(1, sum(v["words"] for v in sets.values())),
                                           "rtfx": a / max(c, 1e-9), "run": run}
    (RES / "results.json").write_text(json.dumps(table, indent=2))
    names = [s for s in SETS if any(s in v["sets"] for v in table.values())]   # only sets that were run
    hdr = "| Model | " + " | ".join(SETS[s]["title"].replace(" test", "") for s in names) + " | Average | Pooled | RTFx |"
    hdr = hdr.replace("| Model |", "| Model | Publisher |", 1)
    lines = [hdr, "|" + "---|" * (len(names) + 5)]
    for m, v in sorted(table.items(), key=lambda kv: kv[1]["avg_wer"]):
        cells = [f"{v['sets'][s]['wer']:.2f}" if s in v["sets"] else "-" for s in names]
        lines.append(f"| {m} | {PUBLISHER.get(m.split('/')[0], m.split('/')[0])} | " + " | ".join(cells) + f" | **{v['avg_wer']:.2f}** | {v['pooled_wer']:.2f} | {v['rtfx']:.0f} |")
    gpus = sorted({v["run"].get("hardware", {}).get("gpu", "cpu") for v in table.values()})
    n = {s: max((v["sets"][s]["clips"] for v in table.values() if s in v["sets"]), default=0) for s in names}
    lines += ["", "Average = mean of the per-set WERs (each set counts equally). "
              "Pooled = all words of all sets together (larger sets count more; Common Voice is most of the clips).",
              "", f"WER in %, normalised (lowercase, no punctuation, digits as Dutch words, fillers removed); lower is better. "
              f"RTFx = seconds of audio per second of compute, batch 1, on {', '.join(gpus)}; higher is faster. "
              "Clips per set: " + ", ".join(f"{SETS[s]['title']} {n[s]:,}" for s in names) + "."]
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
