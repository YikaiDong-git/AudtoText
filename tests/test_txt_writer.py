"""Tests for the .txt writer in audtotext.py.

English output must stay byte-identical to the writer as first released;
Chinese/Japanese/Cantonese output joins segments without spaces after
punctuation and breaks paragraphs at segment ends.

    python -m unittest tests.test_txt_writer -v
"""
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import audtotext  # noqa: E402

# Whisper (tiny) output for synthesized Mandarin speech: ASCII commas, almost
# no sentence-final punctuation, English terms inside, one "?" ending.
ZH_SEGMENTS = [
    "大家好,今天我們來討論電子健康記錄中的缺失數據問題",
    "在真實世界的研究裡,缺失數據幾乎無處不在",
    "如果我們直接刪除不完整的樣本,估計結果可能會產生偏差",
    "那們應該怎麼辦呢?",
    "一種常用的方法是多重插補,也就是Modipo Inputation",
    "它的基本思想是根據觀測道的數據,為每個缺失值生成多個合理的曲質",
    "另一種方法是你蓋綠家權,英文叫Inverse Probability Waiting",
    "這兩種方法都依賴於一個關鍵假設,那就是隨機缺失",
    "如果缺失的機制和未觀測的變量有關,問題就會變得更加複雜",
    "這時候我們需要做敏感性分析",
    "接下來,我會用一個具體的例子說明這些方法在英國推斷中的應用",
    "我們使用的數據來自一家大型醫院,包含了大約10萬名患者的舊診記錄",
    "其中,實驗是檢查結果的缺失比例超過了30%",
    "大家可以想一想,這種情況下應該選擇每一種方法",
]

EN_SEGMENTS = [
    " Thank you all for coming today.", " I want to start with a question that",
    " sounds simple but is not.", " How do we handle missing data in electronic health records?",
    " Most of the time, we just drop the incomplete rows.", " That works if the data are missing completely at random.",
    " In practice, they almost never are!", " Lab tests are ordered because a doctor is worried,",
    " so a missing value tells you something.", " Multiple imputation is one answer.",
    " Inverse probability weighting is another.", " Both rest on an assumption we cannot test from the data alone.",
    " So today I will show three examples.", " The first comes from a large hospital system,",
    " roughly one hundred thousand patients.", " The second is a registry.", " The third is a trial.",
    " Each one breaks the usual assumptions in a different way.", " Let's begin with the hospital data.",
    " Here the lab values are missing for about thirty percent of visits.", " Why would that be?",
]


def first_release_write_txt(segments, path):
    """The .txt writer exactly as released in 568199e (reference for English)."""
    paragraph = " ".join(seg.text.strip() for seg in segments)
    paragraph = re.sub(r"\s+", " ", paragraph).strip()
    sentences = re.split(r"(?<=[.!?])\s+", paragraph)
    with open(path, "w", encoding="utf-8") as f:
        buf = []
        for sent in sentences:
            buf.append(sent)
            if len(" ".join(buf)) > 600:
                f.write(" ".join(buf) + "\n\n")
                buf = []
        if buf:
            f.write(" ".join(buf) + "\n")


def segs(texts):
    return [SimpleNamespace(text=t) for t in texts]


class TxtWriter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, texts, language):
        out = self.dir / f"{language}.txt"
        audtotext.write_txt(segs(texts), out, language)
        return out

    def test_english_and_korean_match_first_release(self):
        long_talk = EN_SEGMENTS * 3
        ends = set()
        for k in range(len(long_talk) + 1):        # every prefix: both final-flush branches
            texts = long_talk[:k]
            ref = self.dir / "ref.txt"
            first_release_write_txt(segs(texts), ref)
            ends.add(ref.read_text(encoding="utf-8").endswith("\n\n"))
            for language in ("en", "ko", "fr"):
                with self.subTest(language=language, k=k):
                    self.assertEqual(self.write(texts, language).read_bytes(), ref.read_bytes())
        self.assertEqual(ends, {True, False}, "fixtures must reach both final-flush branches")

    def test_chinese_joins_after_punctuation_only(self):
        text = self.write(ZH_SEGMENTS, "zh").read_text(encoding="utf-8")
        self.assertIn("怎麼辦呢?一種", text)                     # after "?" -> joined
        self.assertIn("數據問題 在真實世界", text)                 # unpunctuated pause -> space kept
        self.assertIn("Probability Waiting 這兩種", text)
        self.assertEqual(text.count(" "), 12 + 3)    # 13 joins - 1 after "?", + 3 inside English terms
        self.assertTrue(text.endswith("每一種方法\n"))

    def test_chinese_paragraphs_break_at_segment_ends(self):
        texts = ZH_SEGMENTS * 6                                      # ~2,300 characters
        paragraphs = self.write(texts, "zh").read_text(encoding="utf-8").rstrip("\n").split("\n\n")
        self.assertGreaterEqual(len(paragraphs), 4)
        longest = max(len(t) for t in ZH_SEGMENTS)
        for p in paragraphs[:-1]:
            self.assertGreaterEqual(len(p), 250)
            self.assertLess(len(p), 500 + longest + 1)
            self.assertTrue(any(p.endswith(t) for t in ZH_SEGMENTS), "break inside a segment")
        self.assertEqual("".join(paragraphs).replace(" ", ""), "".join(texts).replace(" ", ""))

    def test_sentence_end_after_250_closes_paragraph(self):
        texts = ["这是一个很长的句子没有标点" * 3] * 6 + ["所以我们今天讨论这个问题。", "下一段从这里开始"]
        paragraphs = self.write(texts, "zh").read_text(encoding="utf-8").rstrip("\n").split("\n\n")
        self.assertTrue(paragraphs[0].endswith("讨论这个问题。"))
        self.assertEqual(paragraphs[1], "下一段从这里开始")

    def test_ascii_punctuation_before_latin_keeps_space(self):
        text = self.write(["我们用 Python.", "Next we test it", "。", "Python 很好用"], "ja").read_text(encoding="utf-8")
        self.assertEqual(text, "我们用 Python. Next we test it 。Python 很好用\n")


if __name__ == "__main__":
    unittest.main()
