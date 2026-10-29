"""Mistral Voxtral in transcription mode through transformers (bf16 on GPU, language given as Dutch, greedy)."""
from __future__ import annotations


class Backend:
    def __init__(self, model: str, device: str | None = None):
        import torch
        from transformers import AutoProcessor, VoxtralForConditionalGeneration
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device.startswith("cuda") else torch.float32
        self.repo = model
        self.proc = AutoProcessor.from_pretrained(model)
        self.model = VoxtralForConditionalGeneration.from_pretrained(model, torch_dtype=self.dtype).to(self.device).eval()
        self.settings = {"language": "nl", "mode": "transcription request", "decoding": "greedy", "dtype": str(self.dtype)}

    def versions(self) -> dict:
        import transformers
        return {"transformers": transformers.__version__}

    def transcribe(self, wave) -> str:
        import torch
        inputs = self.proc.apply_transcription_request(language="nl", audio=wave, format=["wav"], model_id=self.repo,
                                                     sampling_rate=16000)
        inputs = inputs.to(self.device, dtype=self.dtype)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=448, do_sample=False)
        return self.proc.batch_decode(out[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
