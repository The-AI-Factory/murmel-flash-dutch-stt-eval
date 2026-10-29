"""NVIDIA NeMo ASR models (Parakeet TDT, Canary) with their default decoding. Canary gets source and target language
Dutch (transcription, not translation)."""
from __future__ import annotations


class Backend:
    def __init__(self, model: str, device: str | None = None):
        import torch
        from nemo.collections.asr.models import ASRModel
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ASRModel.from_pretrained(model, map_location=self.device).eval()
        self.canary = "canary" in model.lower()
        self.kw = {"source_lang": "nl", "target_lang": "nl"} if self.canary else {}
        self.settings = {"decoding": "model default", **self.kw}

    def versions(self) -> dict:
        import nemo
        return {"nemo_toolkit": nemo.__version__}

    def transcribe(self, wave) -> str:
        import torch
        with torch.inference_mode():
            out = self.model.transcribe([wave], batch_size=1, verbose=False, **self.kw)
        h = out[0] if isinstance(out, list) else out
        if isinstance(h, tuple):                                     # (best, all) on older RNNT/TDT models
            h = h[0]
        h = h[0] if isinstance(h, list) else h
        return (getattr(h, "text", h) or "").strip()
