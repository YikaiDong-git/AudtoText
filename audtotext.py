"""
AudtoText - local audio transcription with faster-whisper.

Usage:
    python audtotext.py INPUT_AUDIO [options]

Outputs (next to INPUT_AUDIO unless --output-dir given):
    <stem>.txt          paragraph-formatted plain text
    <stem>.srt          SubRip subtitles
    <stem>.vtt          WebVTT subtitles
    <stem>.timestamped.txt  one segment per line with [hh:mm:ss.ss]
"""
import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np


def register_cuda_dlls():
    """Add bundled NVIDIA wheels' DLL dirs to the search path on Windows.

    faster-whisper relies on CTranslate2, which dynamically loads cuBLAS,
    cuDNN, and nvrtc. The nvidia-*-cu12 pip wheels ship the DLLs but do
    not register them with the OS loader. Without this step, model encode
    fails with: "Library cublas64_12.dll is not found or cannot be loaded".
    """
    if sys.platform != "win32":
        return
    import importlib.util
    spec = importlib.util.find_spec("nvidia")
    if spec is None or not spec.submodule_search_locations:
        return
    nvidia_pkg_dir = Path(next(iter(spec.submodule_search_locations)))
    for sub in ("cublas/bin", "cudnn/bin", "cuda_nvrtc/bin"):
        d = nvidia_pkg_dir / sub
        if d.is_dir():
            os.add_dll_directory(str(d))
            os.environ["PATH"] = str(d) + os.pathsep + os.environ.get("PATH", "")


class BlockwiseFeatures:
    """faster-whisper's log-Mel features with the STFT done a block at a time.

    The stock FeatureExtractor runs one STFT over the whole file, and numpy
    does it in float64, so a 50-minute recording needs ~3 GB of RAM at once
    and fails with _ArrayMemoryError on a busy machine. STFT frames are
    independent, so the FFT and power spectrum run block by block; the mel
    projection and dynamic-range clamp then run once over the whole power
    spectrum, as in the stock extractor. The features are bitwise identical.
    """

    def __init__(self, stock, block_frames: int = 16384):
        self.stock = stock
        self.block_frames = block_frames

    def __getattr__(self, name):
        # sampling_rate, hop_length, nb_max_frames, time_per_frame, ...
        return getattr(self.stock, name)

    def __call__(self, waveform, padding=160, chunk_length=None):
        fe = self.stock
        if chunk_length is not None:
            fe.n_samples = chunk_length * fe.sampling_rate
            fe.nb_max_frames = fe.n_samples // fe.hop_length
        waveform = np.asarray(waveform, dtype=np.float32)
        if padding:
            waveform = np.pad(waveform, (0, padding))
        half = fe.n_fft // 2
        padded = np.pad(waveform, (half, half), mode="reflect")
        del waveform
        window = np.hanning(fe.n_fft + 1)[:-1].astype("float32")
        n_frames = 1 + (padded.shape[0] - fe.n_fft) // fe.hop_length
        frames = np.lib.stride_tricks.as_strided(
            padded, (n_frames, fe.n_fft), (fe.hop_length * padded.strides[0], padded.strides[0]))
        n_out = n_frames - 1  # the stock extractor drops the last frame
        # Same layout as the stock power spectrum (a transposed FFT), so the one
        # mel matmul below is the BLAS call the stock extractor makes. A matmul
        # split into column blocks rounds ~3% of values differently.
        magnitudes = np.empty((fe.n_fft // 2 + 1, n_out), dtype=np.float32, order="F")
        for start in range(0, n_out, self.block_frames):
            stop = min(start + self.block_frames, n_out)
            spec = np.fft.rfft(frames[start:stop] * window, n=fe.n_fft, axis=-1)
            magnitudes[:, start:stop] = np.abs(spec.T.astype("complex64")) ** 2
        del frames, padded
        mel_spec = fe.mel_filters @ magnitudes
        del magnitudes
        log_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None, out=mel_spec), out=mel_spec)
        np.maximum(log_spec, log_spec.max() - 8.0, out=log_spec)
        log_spec += 4.0
        log_spec /= 4.0
        return log_spec


def format_timestamp(seconds: float, sep: str = ",") -> str:
    """Format seconds as HH:MM:SS,mmm (SRT) or HH:MM:SS.mmm (VTT)."""
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


# Written without spaces between words; see write_txt_unspaced.
UNSPACED_LANGUAGES = {"zh", "ja", "yue"}
SENTENCE_END = tuple("。！？.!?")
FULLWIDTH_END = tuple("，。！？、；：）」』”")
ASCII_END = tuple(",.!?;:")


def write_txt(segments, path: Path, language: str) -> None:
    """Plain text with paragraph breaks every ~600 chars at sentence ends."""
    if language in UNSPACED_LANGUAGES:
        write_txt_unspaced(segments, path)
        return
    paragraph = " ".join(seg.text.strip() for seg in segments)
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    with open(path, "w", encoding="utf-8") as f:
        buf: list[str] = []
        for sent in sentences:
            buf.append(sent)
            if len(" ".join(buf)) > 600:
                f.write(" ".join(buf) + "\n\n")
                buf = []
        if buf:
            f.write(" ".join(buf) + "\n")


def write_txt_unspaced(segments, path: Path) -> None:
    """Paragraphs for Chinese, Japanese and Cantonese.

    Whisper often leaves these sentences unpunctuated, so segments join
    directly only after punctuation (ASCII punctuation followed by Latin text
    keeps its space) and keep a space elsewhere, the pause mark Whisper
    itself uses inside segments. Paragraphs break at segment ends: the first
    sentence end after ~250 characters, or any segment end after ~500.
    """
    paragraphs, buf = [], ""
    for seg in segments:
        text = re.sub(r"\s+", " ", seg.text).strip()
        if not text:
            continue
        joined = buf.endswith(FULLWIDTH_END) or (buf.endswith(ASCII_END) and not text[0].isascii())
        if buf and not joined:
            buf += " "
        buf += text
        if (len(buf) >= 250 and text.endswith(SENTENCE_END)) or len(buf) >= 500:
            paragraphs.append(buf)
            buf = ""
    if buf:
        paragraphs.append(buf)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n\n".join(paragraphs) + "\n")


def write_timestamped(segments, path: Path) -> None:
    """One segment per line, each prefixed with [hh:mm:ss.ss]."""
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            h = int(seg.start // 3600)
            m = int((seg.start % 3600) // 60)
            s = seg.start % 60
            f.write(f"[{h:02d}:{m:02d}:{s:05.2f}] {seg.text.strip()}\n")


def write_srt(segments, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(segments, 1):
            f.write(f"{i}\n")
            f.write(f"{format_timestamp(seg.start, ',')} --> "
                    f"{format_timestamp(seg.end, ',')}\n")
            f.write(seg.text.strip() + "\n\n")


def write_vtt(segments, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            f.write(f"{format_timestamp(seg.start, '.')} --> "
                    f"{format_timestamp(seg.end, '.')}\n")
            f.write(seg.text.strip() + "\n\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="audtotext",
        description="Transcribe audio locally with faster-whisper.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("audio", type=Path, help="input audio file (.m4a/.mp3/.wav/...)")
    p.add_argument("-o", "--output-dir", type=Path, default=None,
                   help="output directory (default: same dir as audio)")
    p.add_argument("--model", default="large-v3",
                   help="whisper model: tiny|base|small|medium|large-v3|distil-large-v3")
    p.add_argument("--language", default="en",
                   help="ISO-639-1 language code, or 'auto' to detect")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="inference device")
    p.add_argument("--compute-type", default="auto",
                   help="float16|int8_float16|int8|float32; 'auto' picks fp16 on cuda, int8 on cpu")
    p.add_argument("--beam-size", type=int, default=5,
                   help="beam search width (5 = standard, 10 = max accuracy, slower)")
    p.add_argument("--no-vad", action="store_true",
                   help="disable Silero VAD silence filter")
    p.add_argument("--no-condition-prev", action="store_true",
                   help="disable conditioning on previous text (less coherent, less hallucination)")
    p.add_argument("--formats", default="txt,srt,vtt,timestamped",
                   help="comma-separated subset of: txt, srt, vtt, timestamped")
    p.add_argument("--model-cache-dir", type=Path, default=None,
                   help="where to download model weights (default: HF_HOME or ~/.cache/huggingface)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the sampling used by temperature fallback; same seed + same audio "
                        "= same transcript")
    p.add_argument("--progress-every", type=float, default=10.0,
                   help="seconds between [prog] progress lines")
    return p.parse_args()


def main() -> int:
    # Windows defaults redirected stdout to the ANSI codepage (e.g. cp1252),
    # which cannot encode CJK paths or transcript text.
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    args = parse_args()

    if not args.audio.is_file():
        print(f"audtotext: input not found: {args.audio}", file=sys.stderr)
        return 2

    output_dir = args.output_dir or args.audio.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.audio.stem
    formats = {f.strip().lower() for f in args.formats.split(",") if f.strip()}
    valid_formats = {"txt", "srt", "vtt", "timestamped"}
    unknown = formats - valid_formats
    if unknown:
        print(f"audtotext: unknown format(s) {unknown}; valid: {valid_formats}",
              file=sys.stderr)
        return 2

    if args.model_cache_dir is not None:
        os.environ["HF_HOME"] = str(args.model_cache_dir)
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    register_cuda_dlls()

    import ctranslate2
    from faster_whisper import WhisperModel

    # Temperature fallback (t > 0) samples from the decoder; without a fixed
    # seed, a rerun of the same file can yield a different transcript.
    ctranslate2.set_random_seed(args.seed)

    if args.device == "auto":
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    else:
        device = args.device

    if args.compute_type == "auto":
        compute_type = "float16" if device == "cuda" else "int8"
    else:
        compute_type = args.compute_type

    print(f"[load] {args.model} on {device} {compute_type} ...", flush=True)
    t0 = time.time()
    model = WhisperModel(
        args.model,
        device=device,
        compute_type=compute_type,
        download_root=str(args.model_cache_dir) if args.model_cache_dir else None,
    )
    model.feature_extractor = BlockwiseFeatures(model.feature_extractor)
    print(f"[load] done in {time.time() - t0:.1f}s", flush=True)

    print(f"[run]  transcribing {args.audio}", flush=True)
    t0 = time.time()
    segments_iter, info = model.transcribe(
        str(args.audio),
        language=None if args.language == "auto" else args.language,
        beam_size=args.beam_size,
        best_of=5,
        temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        vad_filter=not args.no_vad,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=not args.no_condition_prev,
        compression_ratio_threshold=2.4,
        log_prob_threshold=-1.0,
        no_speech_threshold=0.6,
        word_timestamps=False,
    )
    print(f"[info] language={info.language} prob={info.language_probability:.2f} "
          f"duration={info.duration:.1f}s ({info.duration / 60:.1f} min)", flush=True)

    segments = []
    last_print = time.time()
    for seg in segments_iter:
        segments.append(seg)
        if time.time() - last_print > args.progress_every:
            pct = seg.end / info.duration * 100
            print(f"[prog] {seg.end:7.1f}s / {info.duration:.1f}s ({pct:5.1f}%) "
                  f"elapsed={time.time() - t0:6.1f}s", flush=True)
            last_print = time.time()

    elapsed = time.time() - t0
    rtf = elapsed / info.duration if info.duration else float("nan")
    print(f"[run]  finished in {elapsed:.1f}s ({elapsed / 60:.1f} min); RTF={rtf:.3f}",
          flush=True)

    writers = {
        "txt": (lambda segs, out: write_txt(segs, out, info.language), ".txt"),
        "srt": (write_srt, ".srt"),
        "vtt": (write_vtt, ".vtt"),
        "timestamped": (write_timestamped, ".timestamped.txt"),
    }
    for fmt in (f for f in writers if f in formats):
        writer, ext = writers[fmt]
        out = output_dir / f"{stem}{ext}"
        writer(segments, out)
        print(f"[save] {fmt:11s} -> {out}", flush=True)

    print("[done]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
