"""murmel-flash via its own package (github.com/The-AI-Factory/murmel-flash-dutch-stt), with the package defaults:
attention beam search (beam 4), CTC-refereed, no external language model."""
from __future__ import annotations


class Backend:
    def __init__(self, model: str, device: str | None = None, mode: str = "beam", beam: int = 4):
        from murmel_flash.inference import Transcriber
        load = getattr(Transcriber, "from_pretrained", None) or Transcriber.from_hub
        self.tr = load(model, device=device)
        self.mode, self.beam = mode, beam
        self.settings = {"mode": mode, "beam": beam, "language_model": None}

    def versions(self) -> dict:
        import murmel_flash
        return {"murmel_flash": getattr(murmel_flash, "__version__", "?")}

    def transcribe(self, wave) -> str:
        return self.tr.transcribe(wave, mode=self.mode, beam=self.beam)["text"].strip()
