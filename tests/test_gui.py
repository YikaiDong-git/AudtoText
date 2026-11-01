"""Tests for audtotext_gui.py.

Fast tests (no GPU) check the pieces that turn CLI output into window state,
using lines copied from real runs. The end-to-end tests drive the window
offscreen, so they need a model cache and an audio file:

    set AUDTOTEXT_MODEL_DIR=D:\\transcribe_workspace\\models
    set AUDTOTEXT_TEST_AUDIO=D:\\transcribe_workspace\\smoketest.m4a
    set AUDTOTEXT_TEST_MODEL=tiny            (optional; default tiny)
    python -m unittest tests.test_gui -v

Close any open AudtoText window first: the forwarding test needs the
single-window channel for itself.
"""
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import audtotext_gui as gui  # noqa: E402
from PySide6.QtCore import QEventLoop, QMimeData, QPointF, QSettings, Qt, QUrl  # noqa: E402
from PySide6.QtGui import QDragEnterEvent, QDropEvent  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

# Lines copied verbatim from run_0924.log and run_call.log (Oct 2026 runs).
REAL_LINES = {
    "[load] large-v3 on cuda float16 ...": {"event": "loading"},
    "[load] done in 5.6s": None,
    "[run]  transcribing C:\\Users\\59827\\Downloads\\9月24日 下午12点17分.m4a": {"event": "started"},
    "[info] language=en prob=1.00 duration=2576.6s (42.9 min)":
        {"event": "info", "language": "en", "prob": 1.0, "duration": 2576.6},
    "[prog]    99.3s / 2576.6s (  3.9%) elapsed=  35.2s":
        {"event": "progress", "position": 99.3, "duration": 2576.6, "pct": 3.9, "elapsed": 35.2},
    "[prog]  2552.6s / 2576.6s ( 99.1%) elapsed= 437.2s":
        {"event": "progress", "position": 2552.6, "duration": 2576.6, "pct": 99.1, "elapsed": 437.2},
    "[run]  finished in 437.2s (7.3 min); RTF=0.170": {"event": "finished", "seconds": 437.2},
    "[save] timestamped -> C:\\Users\\59827\\Downloads\\9月24日 下午12点17分.timestamped.txt":
        {"event": "saved", "format": "timestamped",
         "path": "C:\\Users\\59827\\Downloads\\9月24日 下午12点17分.timestamped.txt"},
    "[save] txt         -> C:\\Users\\59827\\Downloads\\(256)749-0431-2610011715.txt":
        {"event": "saved", "format": "txt",
         "path": "C:\\Users\\59827\\Downloads\\(256)749-0431-2610011715.txt"},
    "[done]": {"event": "done"},
}


class ParseAndCommand(unittest.TestCase):
    def test_real_cli_lines(self):
        for line, expected in REAL_LINES.items():
            with self.subTest(line=line):
                self.assertEqual(gui.parse_line(line), expected)

    def test_command_matches_cli_flags(self):
        s = gui.Settings()
        cmd = gui.build_command(Path("py.exe"), Path("a b.m4a"), s, Path("M"))
        self.assertEqual(cmd, ["py.exe", str(gui.CLI_SCRIPT), "a b.m4a", "--model", "large-v3",
                               "--language", "auto", "--beam-size", "10",
                               "--formats", "txt,srt,vtt,timestamped", "--progress-every", "2",
                               "--model-cache-dir", "M"])
        s = gui.Settings(model="tiny", language="zh", beam_size=5, formats=("txt",),
                         phone_mode=True, next_to_audio=False, output_dir="D:\\out")
        cmd = gui.build_command(Path("py.exe"), Path("x.mp3"), s, None)
        self.assertEqual(cmd[3:], ["--model", "tiny", "--language", "zh", "--beam-size", "5",
                                   "--formats", "txt", "--progress-every", "2",
                                   "--no-condition-prev", "-o", "D:\\out"])

    def test_expand_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub").mkdir()
            for name in ("b.MP3", "a.m4a", "notes.docx", "sub/c.wav", "sub/d.txt"):
                (root / name).write_bytes(b"")
            files, skipped = gui.expand_inputs([root / "a.m4a", root / "b.MP3",
                                                root / "notes.docx", root / "sub"])
            self.assertEqual([f.name for f in files], ["a.m4a", "b.MP3", "c.wav"])
            self.assertEqual(skipped, ["notes.docx"])
            files, skipped = gui.expand_inputs([root / "notes.docx"], check_ext=False)
            self.assertEqual(([f.name for f in files], skipped), (["notes.docx"], []))

    def test_settings_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            qs = QSettings(str(Path(tmp) / "s.ini"), QSettings.Format.IniFormat)
            s = gui.Settings(model="tiny", language="zh", beam_size=5, formats=("srt", "txt"),
                             phone_mode=True, next_to_audio=False, output_dir="D:\\out")
            s.save(qs)
            qs.sync()
            again = gui.Settings.load(QSettings(str(Path(tmp) / "s.ini"), QSettings.Format.IniFormat))
            self.assertEqual(again, s)

    def test_clock(self):
        self.assertEqual([gui.fmt_clock(x) for x in (0, 59.6, 582.1, 4046.8)],
                         ["0:00", "1:00", "9:42", "1:07:27"])


class WindowIdle(unittest.TestCase):
    def test_fresh_window_has_no_transcript_actions(self):
        app = QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as tmp:
            gui.SETTINGS_PATH = Path(tmp) / "settings.ini"
            win = gui.MainWindow(None)
            for button in (win.copy_btn, win.open_btn, win.reveal_btn, win.cancel_btn):
                self.assertFalse(button.isEnabled(), button.text())
            self.assertIn("next to each file", win.drop_zone.hint.text())
            win.next_to_audio.setChecked(False)
            self.assertTrue(win.output_dir.isEnabled())
            self.assertIn("folder you choose", win.drop_zone.hint.text())
            win.server.close()
            win.close()
            app.processEvents()


@unittest.skipUnless(os.environ.get("AUDTOTEXT_MODEL_DIR") and os.environ.get("AUDTOTEXT_TEST_AUDIO"),
                     "set AUDTOTEXT_MODEL_DIR and AUDTOTEXT_TEST_AUDIO to run the window tests")
class WindowEndToEnd(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])
        cls.model_dir = Path(os.environ["AUDTOTEXT_MODEL_DIR"])
        cls.source = Path(os.environ["AUDTOTEXT_TEST_AUDIO"])
        cls.model = os.environ.get("AUDTOTEXT_TEST_MODEL", "tiny")

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="audtotext_gui_test_"))
        gui.SETTINGS_PATH = self.tmp / "settings.ini"
        # Same awkward name shape as real recordings: CJK, spaces, parentheses.
        self.audio = self.tmp / f"测试 录音 (1){self.source.suffix}"
        shutil.copyfile(self.source, self.audio)
        self.win = gui.MainWindow(self.model_dir)
        self.win.model.setCurrentIndex(self.win.model.findData(self.model))

    def tearDown(self):
        self.assertIsNone(self.win.proc, "a transcription was still running at teardown")
        self.win.server.close()
        self.win.close()
        shutil.rmtree(self.tmp)

    def wait_until(self, condition, timeout_s):
        deadline = time.monotonic() + timeout_s
        while not condition():
            if time.monotonic() > deadline:
                self.fail(f"timed out after {timeout_s}s")
            self.app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 100)
            time.sleep(0.02)

    def drop(self, paths):
        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile(str(p)) for p in paths])
        enter = QDragEnterEvent(QPointF(20, 20), Qt.DropAction.CopyAction, mime,
                                Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(self.win, enter)
        self.assertTrue(enter.isAccepted(), "window refused the drag")
        drop = QDropEvent(QPointF(20, 20), Qt.DropAction.CopyAction, mime,
                          Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier)
        QApplication.sendEvent(self.win, drop)

    def test_drop_writes_the_same_files_as_the_cli(self):
        stray = self.tmp / "notes.docx"
        stray.write_bytes(b"")
        self.drop([self.audio, stray])
        self.assertEqual([j.audio for j in self.win.jobs], [self.audio])
        job = self.win.jobs[0]
        self.wait_until(lambda: job.state not in ("queued", "running"), 900)
        self.assertEqual(job.state, "done", "\n".join(job.log[-30:]))
        self.assertIn("[exit] code=0x00000000 status=NormalExit", job.log)
        self.assertEqual(set(job.outputs), {"txt", "srt", "vtt", "timestamped"})

        reference = self.tmp / "cli"
        reference.mkdir()
        cmd = gui.build_command(gui.engine_python(), self.audio, job.settings, self.model_dir)
        proc = subprocess.run(cmd + ["-o", str(reference)], capture_output=True,
                              text=True, encoding="utf-8")
        self.assertEqual(proc.returncode, 0, proc.stdout[-2000:] + proc.stderr[-2000:])
        for fmt, path in job.outputs.items():
            with self.subTest(fmt=fmt):
                self.assertEqual(path.parent, self.audio.parent)
                self.assertEqual(path.read_bytes(), (reference / path.name).read_bytes())
        self.assertEqual(self.win.transcript.toPlainText(),
                         job.outputs["txt"].read_text(encoding="utf-8"))

    def test_second_launch_forwards_files_and_cancel_stops_the_run(self):
        self.assertTrue(self.win.server.isListening(),
                        "another AudtoText window owns the channel; close it and rerun")
        env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
        second = subprocess.Popen([str(gui.engine_python()), str(gui.APP_DIR / "audtotext_gui.py"),
                                   str(self.audio)], env=env)
        try:
            self.wait_until(lambda: second.poll() is not None and self.win.jobs, 60)
        finally:
            if second.poll() is None:
                second.kill()
        self.assertEqual(second.returncode, 0)
        job = self.win.jobs[0]
        self.assertEqual(job.audio.resolve(), self.audio.resolve())
        self.wait_until(lambda: job.state == "running" and job.pct >= 0, 120)
        self.win._cancel_current()
        self.wait_until(lambda: job.state != "running", 60)
        self.assertEqual(job.state, "cancelled")
        self.assertEqual(job.outputs, {})


if __name__ == "__main__":
    unittest.main()
