| Model | Publisher | FLEURS nl_nl | Common Voice 17.0 nl | Multilingual LibriSpeech Dutch | VoxPopuli nl | Average | Pooled | RTFx |
|---|---|---|---|---|---|---|---|---|
| theaifactorybv/murmel-flash-l-v5 | The AI Factory | 7.77 | 3.68 | 7.52 | 9.41 | **7.09** | 6.22 | 30 |
| nvidia/canary-1b-v2 | NVIDIA | 6.04 | 4.38 | 9.77 | 11.20 | **7.85** | 7.70 | 16 |
| mistralai/Voxtral-Mini-3B-2507 | Mistral AI | 5.80 | 5.02 | 9.83 | 14.05 | **8.68** | 8.25 | 7 |
| theaifactorybv/murmel-flash-m-v5 | The AI Factory | 9.67 | 5.12 | 9.35 | 10.64 | **8.70** | 7.84 | 40 |
| nvidia/parakeet-tdt-0.6b-v3 | NVIDIA | 7.91 | 4.90 | 11.07 | 11.51 | **8.85** | 8.62 | 101 |
| openai/whisper-large-v3 | OpenAI | 5.56 | 4.23 | 9.82 | 23.06 | **10.66** | 8.80 | 8 |
| openai/whisper-large-v3-turbo | OpenAI | 6.03 | 5.82 | 9.60 | 28.38 | **12.46** | 9.85 | 37 |

Average = mean of the per-set WERs (each set counts equally). Pooled = all words of all sets together (larger sets count more; Common Voice is most of the clips).

WER in %, normalised (lowercase, no punctuation, digits as Dutch words, fillers removed); lower is better. RTFx = seconds of audio per second of compute, batch 1, on NVIDIA A100-SXM4-40GB; higher is faster. Clips per set: FLEURS nl_nl test 364, Common Voice 17.0 nl test 11,266, Multilingual LibriSpeech Dutch test 3,075, VoxPopuli nl test 1,137.
