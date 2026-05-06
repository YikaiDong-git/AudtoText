# AudtoText

Local English (and multilingual) audio transcription on Windows + NVIDIA GPU using [`faster-whisper`](https://github.com/SYSTRAN/faster-whisper) with the `large-v3` model. One command in, four files out (`.txt`, `.srt`, `.vtt`, timestamped `.txt`).

![License](https://img.shields.io/badge/license-MIT-blue) ![Python](https://img.shields.io/badge/python-3.10%2B-blue) ![CUDA](https://img.shields.io/badge/CUDA-12.x-green) ![Platform](https://img.shields.io/badge/platform-Windows-lightgrey)

## Features

- **Whisper `large-v3` accuracy** with GPU acceleration via CTranslate2 (fp16).
- **Single command**, no cloud upload — your audio never leaves the machine.
- **Auto-emits four formats** from one pass: paragraphed `.txt`, segment-level timestamped `.txt`, SubRip `.srt`, WebVTT `.vtt`.
- **Auto device & precision** detection (CUDA fp16, falls back to CPU int8).
- **Bundled CUDA DLL discovery** — `nvidia-cublas-cu12` / `nvidia-cudnn-cu12` wheels are picked up automatically; no system-wide CUDA toolkit install needed.
- **VAD silence filtering** (Silero) so the decoder doesn't waste compute on empty stretches.

## Requirements

| Component | Tested | Minimum |
|-----------|--------|---------|
| OS        | Windows 11 | Windows 10 (Linux / WSL also fine; CUDA DLL auto-loader is Windows-only) |
| GPU       | RTX 2070 SUPER (8 GB) | Any CUDA 12-compatible NVIDIA GPU with ≥ 6 GB VRAM for `large-v3` fp16 |
| Driver    | 591.86 (CUDA 13.1 driver) | Driver supporting CUDA ≥ 12.0 |
| Python    | 3.11 | 3.10+ |
| FFmpeg    | 7.x | Any recent build, on `PATH` |
| Disk      | ~6 GB free | for env (~3 GB) + `large-v3` weights (~3 GB) |

CPU-only inference also works (set `--device cpu`) but is ~30× slower; expect ~4× real-time on a modern desktop CPU with the `medium` model, much worse on `large-v3`.

## Installation

> Install into an isolated env, not your base Python.

### Conda (recommended on Windows)

```powershell
conda create -p .\env python=3.11 ffmpeg -c conda-forge -y
.\env\Scripts\activate
pip install -r requirements.txt
```

### venv + system FFmpeg

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
# Then install FFmpeg separately and ensure `ffmpeg` is on PATH:
#   winget install Gyan.FFmpeg
```

Verify:

```powershell
ffmpeg -version | Select-String "ffmpeg version"
python -c "import ctranslate2; print('CUDA devices:', ctranslate2.get_cuda_device_count())"
```

The first run downloads `large-v3` weights (~3 GB) into the HuggingFace cache (`%USERPROFILE%\.cache\huggingface` by default; override with `--model-cache-dir` or `HF_HOME`).

## Quick start

```powershell
python audtotext.py path\to\seminar.m4a
```

Console output:

```
[load] large-v3 on cuda float16 ...
[load] done in 5.5s
[run]  transcribing path\to\seminar.m4a
[info] language=en prob=1.00 duration=4046.8s (67.4 min)
[prog]   495.8s / 4046.8s ( 12.3%) elapsed=  83.7s
...
[run]  finished in 532.3s (8.9 min); RTF=0.132
[save] txt         -> path\to\seminar.txt
[save] srt         -> path\to\seminar.srt
[save] vtt         -> path\to\seminar.vtt
[save] timestamped -> path\to\seminar.timestamped.txt
[done]
```

## Usage

```
python audtotext.py AUDIO [-o OUTPUT_DIR]
                          [--model {tiny|base|small|medium|large-v3|distil-large-v3}]
                          [--language en|zh|...|auto]
                          [--device {auto,cuda,cpu}]
                          [--compute-type {auto,float16,int8_float16,int8,float32}]
                          [--beam-size N]
                          [--no-vad] [--no-condition-prev]
                          [--formats txt,srt,vtt,timestamped]
                          [--model-cache-dir DIR]
```

Common recipes:

```powershell
# Best accuracy on a 1-hr seminar (slow): beam 10
python audtotext.py talk.m4a --beam-size 10

# 8 GB VRAM card running medium model
python audtotext.py talk.m4a --model medium --beam-size 5

# Mandarin lecture, autodetect language
python audtotext.py 讲座.m4a --language auto

# Subtitles only, skip the .txt
python audtotext.py movie.mkv --formats srt,vtt

# Force CPU (no GPU available)
python audtotext.py meeting.wav --device cpu --compute-type int8 --model small
```

## Benchmarks

Measured end-to-end on this repo's own pipeline. RTF = wall-clock / audio-duration; lower is faster.

| Model      | Precision | Beam | GPU                | Audio  | Wall   | RTF   | Peak VRAM |
|------------|-----------|------|--------------------|--------|--------|-------|-----------|
| `large-v3` | fp16      | 10   | RTX 2070 SUPER 8GB | 67.4 min | 8.9 min | **0.132** | ~4.2 GB   |

Caveats: batch size 1 (faster-whisper default), VAD on, condition-on-previous-text on, 6-temperature fallback. Larger GPUs (RTX 30/40 series) typically reach RTF 0.03–0.06 on `large-v3`; see [SYSTRAN/faster-whisper benchmarks](https://github.com/SYSTRAN/faster-whisper#benchmark) for cross-card numbers.

## Output formats

| File | Contents | Use for |
|------|----------|---------|
| `<stem>.txt`              | Paragraph-formatted prose, sentence-aware breaks every ~600 chars. | Reading, copy-paste into a doc. |
| `<stem>.timestamped.txt`  | One Whisper segment per line, prefixed with `[hh:mm:ss.ss]`. | Quick navigation; lightweight `grep`-able. |
| `<stem>.srt`              | SubRip subtitles, `00:00:00,000 --> 00:00:05,400` blocks. | Video subtitle tracks (`mkvtoolnix`, `ffmpeg -c:s mov_text`). |
| `<stem>.vtt`              | WebVTT, dot-separator timestamps, `WEBVTT` header. | HTML5 `<track>` elements, web players. |

Pick a subset with `--formats`, e.g. `--formats txt`.

## Troubleshooting

**`RuntimeError: Library cublas64_12.dll is not found or cannot be loaded`**
You're missing CUDA 12 runtime DLLs. The `nvidia-cublas-cu12` and `nvidia-cudnn-cu12` wheels in `requirements.txt` ship them; make sure they installed (`pip list | findstr nvidia`). The script auto-registers them on Windows via `os.add_dll_directory`.

**Download appears to start but `config.json` is empty / parse error**
Stale HuggingFace `xet` downloader. The script sets `HF_HUB_DISABLE_XET=1` automatically; if you bypassed it, re-export and retry:
```powershell
$env:HF_HUB_DISABLE_XET=1
```

**`CUDA out of memory` on `large-v3`**
Try `--compute-type int8_float16` (≈ half the VRAM, very small accuracy hit) or drop to `--model medium`. `large-v3` int8_float16 fits in ~5 GB.

**Output is just `[BLANK_AUDIO]` or empty**
Check VAD didn't eat your whole file: re-run with `--no-vad`. If the audio is very quiet, also try `--no-condition-prev` to suppress hallucinated repetition.

**Slow first run, fast later runs**
First invocation downloads `large-v3` weights (~3 GB) and warms the CUDA kernel cache. Subsequent runs reuse both.

**`.m4a` won't decode**
FFmpeg is not on `PATH`. `ffmpeg -version` must work. The conda recipe above installs it into the env automatically.

## How it works (one-line)

`pyav` → audio frames → Mel features → CTranslate2 `large-v3` encoder/decoder on GPU → segment list → write four format files.

## Acknowledgements

- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — the CTranslate2-backed Whisper wrapper this project is built on.
- [OpenAI Whisper](https://github.com/openai/whisper) — original `large-v3` weights.
- [OpenNMT/CTranslate2](https://github.com/OpenNMT/CTranslate2) — fast inference engine.

## License

MIT — see [LICENSE](LICENSE).
