"""BlockwiseFeatures must reproduce faster-whisper's stock log-Mel features bitwise.

Uses a speech-like synthetic signal; set AUDTOTEXT_TEST_AUDIO to also check
a real recording.

    python -m unittest tests.test_features -v
"""
import os
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audtotext import BlockwiseFeatures  # noqa: E402
from faster_whisper.audio import decode_audio  # noqa: E402
from faster_whisper.feature_extractor import FeatureExtractor  # noqa: E402


def speechlike(seconds: float, seed: int = 0) -> np.ndarray:
    """Voiced harmonics on a moving pitch, gated into syllables, plus noise."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * 16000)) / 16000
    phase = 2 * np.pi * np.cumsum(120 + 30 * np.sin(2 * np.pi * 0.3 * t)) / 16000
    voiced = sum(np.sin(k * phase) / k for k in range(1, 30))
    syllables = np.sin(2 * np.pi * 3 * t) > -0.2
    return (0.1 * voiced * syllables + 0.01 * rng.standard_normal(t.size)).astype(np.float32)


class BlockwiseMatchesStock(unittest.TestCase):
    def check(self, audio):
        for n_mels in (80, 128):                 # large-v3 uses 128, the others 80
            stock = FeatureExtractor(feature_size=n_mels)
            ref = stock(audio, chunk_length=30)
            ref_lang = stock(audio[: 30 * 16000])   # language-detection call
            n_out = ref.shape[1]
            for block in (16384, 4096, 1000, 777, n_out):
                with self.subTest(n_mels=n_mels, block=block):
                    got = BlockwiseFeatures(FeatureExtractor(feature_size=n_mels), block)
                    self.assertTrue(np.array_equal(got(audio, chunk_length=30), ref))
                    self.assertTrue(np.array_equal(got(audio[: 30 * 16000]), ref_lang))
                    self.assertEqual((got.nb_max_frames, got.n_samples),
                                     (stock.nb_max_frames, stock.n_samples))

    def test_speechlike_signal(self):
        self.check(speechlike(200))              # 20,000 frames: two default blocks, ragged tail

    @unittest.skipUnless(os.environ.get("AUDTOTEXT_TEST_AUDIO"), "set AUDTOTEXT_TEST_AUDIO")
    def test_real_recording(self):
        self.check(decode_audio(os.environ["AUDTOTEXT_TEST_AUDIO"], sampling_rate=16000))

    def test_split_mel_matmul_is_detectable_here(self):
        """Control: the identity checks above can fail on this input.

        Splitting the mel matmul into column blocks (the first BlockwiseFeatures
        draft) changes float32 rounding with OpenBLAS; if this BLAS happens to
        be exact for split matmuls, the control cannot discriminate and skips.
        """
        fe = FeatureExtractor(feature_size=128)
        audio = np.pad(speechlike(200), (0, 160))
        padded = np.pad(audio, (200, 200), mode="reflect")
        n = 1 + (padded.size - 400) // 160 - 1
        frames = np.lib.stride_tricks.as_strided(padded, (n, 400), (640, 4))
        window = np.hanning(401)[:-1].astype("float32")
        mags = np.abs(np.fft.rfft(frames * window, axis=-1).T.astype("complex64")) ** 2
        whole = fe.mel_filters @ mags
        split = np.concatenate([fe.mel_filters @ mags[:, s:s + 1000] for s in range(0, n, 1000)], axis=1)
        if np.array_equal(whole, split):
            self.skipTest("this BLAS rounds split matmuls identically")
        self.assertFalse(np.array_equal(whole, split))


if __name__ == "__main__":
    unittest.main()
