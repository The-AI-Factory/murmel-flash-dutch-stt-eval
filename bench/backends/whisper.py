"""Whisper-family models through Hugging Face transformers: fp16 on GPU, SDPA attention, language fixed to Dutch,
task transcribe, greedy decoding (the transformers default), long-form generation for clips over 30 s."""
from __future__ import annotations


class Backend:
    def __init__(self, model: str, device: str | None = None, num_beams: int = 1):
        import torch
        from transformers import AutoProcessor, WhisperForConditionalGeneration
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self.proc = AutoProcessor.from_pretrained(model)
        self.model = WhisperForConditionalGeneration.from_pretrained(model, torch_dtype=self.dtype,
                                                                     attn_implementation="sdpa").to(self.device).eval()
        self.num_beams = num_beams
        self.settings = {"language": "nl", "task": "transcribe", "num_beams": num_beams, "dtype": str(self.dtype)}

    def versions(self) -> dict:
        import transformers
        return {"transformers": transformers.__version__}

    def transcribe(self, wave) -> str:
        import torch
        long = len(wave) > 30 * 16000
        feats = self.proc(wave, sampling_rate=16000, return_tensors="pt", truncation=not long,
                          padding="longest" if long else "max_length", return_attention_mask=True)
        kw = {"input_features": feats.input_features.to(self.device, self.dtype),
              "attention_mask": feats.attention_mask.to(self.device)}
        with torch.inference_mode():
            ids = self.model.generate(**kw, language="nl", task="transcribe", num_beams=self.num_beams,
                                      return_timestamps=long)
        return self.proc.batch_decode(ids, skip_special_tokens=True)[0].strip()
