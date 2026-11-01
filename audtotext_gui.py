"""
AudtoText desktop app.

Drop audio or video files onto the window, or onto its desktop shortcut, and
get the same transcript files the command line writes (.txt, .srt, .vtt,
.timestamped.txt), saved next to each file by default.

Every file runs audtotext.py in its own child process, so the output is
exactly what the CLI produces and the GPU is released between files.

Usage:
    pythonw audtotext_gui.py [--model-cache-dir DIR] [AUDIO ...]
"""
import argparse
import codecs
import ctypes
import getpass
import os
import re
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path

if sys.platform == "win32":
    # Qt6Core.dll links against Windows' own ICU (System32\icuuc.dll). Inside
    # a conda env, Library\bin\icuuc.dll (conda-forge ICU, pulled in by
    # ffmpeg) comes first on the DLL search path, and importing QtCore fails
    # with "The specified procedure could not be found". Loading the system
    # copy by full path first makes the loader reuse it for Qt.
    ctypes.WinDLL(os.path.join(os.environ["SystemRoot"], "System32", "icuuc.dll"))
    # Own taskbar identity, so Windows shows this app's icon instead of Python's.
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("AudtoText.Desktop")

from PySide6.QtCore import QDir, QProcess, QProcessEnvironment, QSettings, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QFontDatabase, QIcon, QPalette
from PySide6.QtNetwork import QLocalServer, QLocalSocket
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QHeaderView, QLabel, QLineEdit, QMainWindow, QMessageBox, QPlainTextEdit,
    QProgressBar, QPushButton, QTabWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

APP_DIR = Path(__file__).resolve().parent
CLI_SCRIPT = APP_DIR / "audtotext.py"
ICON_PATH = APP_DIR / "assets" / "audtotext.ico"
SETTINGS_PATH = APP_DIR / "audtotext_gui.ini"
SERVER_NAME = "AudtoText-" + re.sub(r"\W", "_", getpass.getuser())

MEDIA_EXTS = (".m4a", ".mp3", ".wav", ".flac", ".ogg", ".opus", ".aac", ".wma", ".amr",
              ".aif", ".aiff", ".mp4", ".m4v", ".mov", ".mkv", ".webm", ".avi", ".wmv",
              ".3gp", ".mpg", ".mpeg", ".ts")
LANGUAGES = [("auto", "Auto-detect"), ("en", "English"), ("zh", "Chinese"), ("es", "Spanish"),
             ("fr", "French"), ("de", "German"), ("ja", "Japanese"), ("ko", "Korean")]
ACCURACY = [(10, "Max accuracy (beam 10)"), (5, "Standard (beam 5)")]
MODELS = ["large-v3", "large-v3-turbo", "medium", "small", "tiny"]
FORMATS = [("txt", "Text .txt"), ("srt", "Subtitles .srt"),
           ("vtt", "Web subtitles .vtt"), ("timestamped", "Timestamped .txt")]
FAILED_COLOR = QColor("#c42b1c")

STYLE = """
#dropZone { border: 2px dashed palette(mid); border-radius: 12px; background: palette(base); }
#dropZone[hover="true"] { border: 2px solid palette(highlight); background: palette(alternate-base); }
#dropTitle { font-size: 15pt; font-weight: 600; }
"""


@dataclass
class Settings:
    """The options shown in the window; persisted in audtotext_gui.ini."""
    model: str = "large-v3"
    language: str = "auto"
    beam_size: int = 10
    formats: tuple = ("txt", "srt", "vtt", "timestamped")
    phone_mode: bool = False
    next_to_audio: bool = True
    output_dir: str = ""

    @classmethod
    def load(cls, qs: QSettings) -> "Settings":
        d = cls()
        formats = qs.value("formats", ",".join(d.formats), type=str)
        return cls(
            model=qs.value("model", d.model, type=str),
            language=qs.value("language", d.language, type=str),
            beam_size=qs.value("beam_size", d.beam_size, type=int),
            formats=tuple(f for f in formats.split(",") if f in dict(FORMATS)),
            phone_mode=qs.value("phone_mode", d.phone_mode, type=bool),
            next_to_audio=qs.value("next_to_audio", d.next_to_audio, type=bool),
            output_dir=qs.value("output_dir", d.output_dir, type=str),
        )

    def save(self, qs: QSettings) -> None:
        for key in ("model", "language", "beam_size", "phone_mode", "next_to_audio", "output_dir"):
            qs.setValue(key, getattr(self, key))
        qs.setValue("formats", ",".join(self.formats))


@dataclass
class Job:
    """One queued file and everything learned from its CLI output."""
    audio: Path
    settings: Settings
    item: QTreeWidgetItem
    state: str = "queued"          # queued | running | done | failed | cancelled
    stage: str = "Waiting"
    language: str = ""
    language_prob: float = 0.0
    duration: float = 0.0
    pct: float = -1.0              # -1 until the transcription itself starts
    run_seconds: float = 0.0
    outputs: dict = field(default_factory=dict)
    log: list = field(default_factory=list)
    saw_done: bool = False
    cancel_requested: bool = False


def engine_python() -> Path:
    """Console python.exe beside the running interpreter (the GUI may run under pythonw.exe)."""
    exe = Path(sys.executable)
    return exe.with_name("python.exe") if exe.name.lower() == "pythonw.exe" else exe


def build_command(python_exe: Path, audio: Path, s: Settings, model_cache_dir) -> list:
    """The exact audtotext.py command line for one file."""
    cmd = [str(python_exe), str(CLI_SCRIPT), str(audio),
           "--model", s.model, "--language", s.language,
           "--beam-size", str(s.beam_size), "--formats", ",".join(s.formats),
           "--progress-every", "2"]
    if s.phone_mode:
        cmd.append("--no-condition-prev")
    if not s.next_to_audio:
        cmd += ["-o", s.output_dir]
    if model_cache_dir is not None:
        cmd += ["--model-cache-dir", str(model_cache_dir)]
    return cmd


def expand_inputs(paths, check_ext: bool = True):
    """Split dropped paths into media files to queue and names to skip.

    A dropped folder contributes the media files directly inside it.
    """
    files, skipped = [], []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            files += sorted(c for c in p.iterdir() if c.is_file() and c.suffix.lower() in MEDIA_EXTS)
        elif p.is_file() and (not check_ext or p.suffix.lower() in MEDIA_EXTS):
            files.append(p)
        else:
            skipped.append(p.name)
    return files, skipped


RE_INFO = re.compile(r"^\[info\] language=(\S+) prob=([\d.]+) duration=([\d.]+)s")
RE_PROG = re.compile(r"^\[prog\]\s+([\d.]+)s / ([\d.]+)s \(\s*([\d.]+)%\) elapsed=\s*([\d.]+)s")
RE_FINISHED = re.compile(r"^\[run\]\s+finished in ([\d.]+)s")
RE_SAVE = re.compile(r"^\[save\] (\w+)\s+-> (.+)$")


def parse_line(line: str):
    """Turn one line of audtotext.py output into an event dict (None if it carries no state)."""
    if line.startswith("[load]") and line.endswith("..."):
        return {"event": "loading"}
    if line.startswith("[run]  transcribing "):
        return {"event": "started"}
    if m := RE_INFO.match(line):
        return {"event": "info", "language": m[1], "prob": float(m[2]), "duration": float(m[3])}
    if m := RE_PROG.match(line):
        return {"event": "progress", "position": float(m[1]), "duration": float(m[2]),
                "pct": float(m[3]), "elapsed": float(m[4])}
    if m := RE_FINISHED.match(line):
        return {"event": "finished", "seconds": float(m[1])}
    if m := RE_SAVE.match(line):
        return {"event": "saved", "format": m[1], "path": m[2]}
    if line == "[done]":
        return {"event": "done"}
    return None


def fmt_clock(seconds: float) -> str:
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def forward_to_running_window(paths) -> bool:
    """Hand files to an AudtoText window that is already open; False if none is."""
    sock = QLocalSocket()
    sock.connectToServer(SERVER_NAME)
    if not sock.waitForConnected(500):
        return False
    sock.write("\n".join(paths).encode("utf-8"))
    sock.waitForBytesWritten(3000)
    sock.disconnectFromServer()
    if sock.state() != QLocalSocket.LocalSocketState.UnconnectedState:
        sock.waitForDisconnected(3000)
    return True


class DropZone(QFrame):
    """Dashed drop target; a click opens the file chooser."""
    clicked = Signal()

    def __init__(self, icon: QIcon):
        super().__init__()
        self.setObjectName("dropZone")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(140)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        art = QLabel()
        art.setPixmap(icon.pixmap(48, 48))
        title = QLabel("Drop audio or video files here")
        title.setObjectName("dropTitle")
        self.hint = QLabel()
        self.hint.setForegroundRole(QPalette.ColorRole.PlaceholderText)
        for w in (art, title, self.hint):
            w.setAlignment(Qt.AlignmentFlag.AlignCenter)
            layout.addWidget(w)
        self.set_hover(False)

    def set_hover(self, on: bool) -> None:
        self.setProperty("hover", on)
        self.style().unpolish(self)
        self.style().polish(self)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()


class MainWindow(QMainWindow):
    def __init__(self, model_cache_dir):
        super().__init__()
        self.model_cache_dir = model_cache_dir
        self.qsettings = QSettings(str(SETTINGS_PATH), QSettings.Format.IniFormat)
        self.jobs: list[Job] = []
        self.current: Job | None = None
        self.proc: QProcess | None = None
        self.closing = False
        self._partial = ""
        self._decoder = None
        self.icon = QIcon(str(ICON_PATH))
        self.setWindowTitle("AudtoText")
        self.setWindowIcon(self.icon)
        self.setAcceptDrops(True)
        self.resize(780, 860)
        self._build_ui()
        self._apply_settings(Settings.load(self.qsettings))
        self._connect_settings()
        self._refresh_controls()
        self._show_selected()

        self.server = QLocalServer(self)
        self.server.newConnection.connect(self._on_forwarded_connection)
        if not self.server.listen(SERVER_NAME):
            self.statusBar().showMessage(
                f"Files dropped on the desktop icon will open a second window "
                f"({self.server.errorString()})")

    # ---- layout -------------------------------------------------------
    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(16, 16, 16, 12)
        root.setSpacing(12)

        self.drop_zone = DropZone(self.icon)
        self.drop_zone.clicked.connect(self._choose_files)
        root.addWidget(self.drop_zone)

        box = QGroupBox("Options")
        grid = QGridLayout(box)
        self.language = QComboBox()
        for code, label in LANGUAGES:
            self.language.addItem(label, code)
        self.accuracy = QComboBox()
        for beam, label in ACCURACY:
            self.accuracy.addItem(label, beam)
        self.model = QComboBox()
        for name in MODELS:
            self.model.addItem(name, name)
        self.model.setToolTip("Models other than large-v3 download on first use.")
        for col, (label, widget) in enumerate((("Language", self.language),
                                               ("Accuracy", self.accuracy),
                                               ("Model", self.model))):
            grid.addWidget(QLabel(label), 0, 2 * col)
            grid.addWidget(widget, 0, 2 * col + 1)

        fmt_row = QHBoxLayout()
        self.format_boxes = {}
        for key, label in FORMATS:
            self.format_boxes[key] = QCheckBox(label)
            fmt_row.addWidget(self.format_boxes[key])
        fmt_row.addStretch(1)
        grid.addWidget(QLabel("Files"), 1, 0)
        grid.addLayout(fmt_row, 1, 1, 1, 5)

        self.phone_mode = QCheckBox("Phone call or noisy recording")
        self.phone_mode.setToolTip(
            "Turns off conditioning on previous text (--no-condition-prev). Use it when a "
            "transcript comes out all lowercase with no punctuation, or repeats itself.")
        grid.addWidget(self.phone_mode, 2, 1, 1, 5)

        out_row = QHBoxLayout()
        self.next_to_audio = QCheckBox("Next to the audio file")
        self.output_dir = QLineEdit()
        self.output_dir.setPlaceholderText("Output folder")
        self.output_dir.setAcceptDrops(False)
        self.browse_btn = QPushButton("Browse…")
        self.browse_btn.clicked.connect(self._choose_output_dir)
        out_row.addWidget(self.next_to_audio)
        out_row.addWidget(self.output_dir, 1)
        out_row.addWidget(self.browse_btn)
        grid.addWidget(QLabel("Save to"), 3, 0)
        grid.addLayout(out_row, 3, 1, 1, 5)
        for col in (1, 3, 5):
            grid.setColumnStretch(col, 1)
        root.addWidget(box)

        self.queue = QTreeWidget()
        self.queue.setHeaderLabels(["File", "Length", "Language", "Status"])
        self.queue.setRootIsDecorated(False)
        self.queue.setUniformRowHeights(True)
        header = self.queue.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.queue.setColumnWidth(3, 230)
        self.queue.currentItemChanged.connect(self._show_selected)
        root.addWidget(self.queue, 2)

        progress_row = QHBoxLayout()
        self.progress = QProgressBar()
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setToolTip("Stop the file that is being transcribed")
        self.cancel_btn.clicked.connect(self._cancel_current)
        self.clear_btn = QPushButton("Clear list")
        self.clear_btn.setToolTip("Remove finished and waiting files; the running one keeps going")
        self.clear_btn.clicked.connect(self._clear_list)
        progress_row.addWidget(self.progress, 1)
        progress_row.addWidget(self.cancel_btn)
        progress_row.addWidget(self.clear_btn)
        root.addLayout(progress_row)

        self.tabs = QTabWidget()
        page = QWidget()
        page_layout = QVBoxLayout(page)
        self.transcript = QPlainTextEdit()
        self.transcript.setReadOnly(True)
        self.transcript.setAcceptDrops(False)
        self.transcript.setPlaceholderText(
            "The transcript of the selected file appears here when it is done.")
        buttons = QHBoxLayout()
        self.copy_btn = QPushButton("Copy text")
        self.copy_btn.clicked.connect(self._copy_text)
        self.open_btn = QPushButton("Open file")
        self.open_btn.clicked.connect(self._open_transcript)
        self.reveal_btn = QPushButton("Show in folder")
        self.reveal_btn.clicked.connect(self._reveal_transcript)
        for b in (self.copy_btn, self.open_btn, self.reveal_btn):
            buttons.addWidget(b)
        buttons.addStretch(1)
        page_layout.addWidget(self.transcript, 1)
        page_layout.addLayout(buttons)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setAcceptDrops(False)
        self.log_view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.log_view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        self.tabs.addTab(page, "Transcript")
        self.tabs.addTab(self.log_view, "Log")
        root.addWidget(self.tabs, 3)

        self.setCentralWidget(central)
        cache = str(self.model_cache_dir) if self.model_cache_dir else "Hugging Face default cache"
        self.statusBar().addPermanentWidget(QLabel(f"Models: {cache}"))

    # ---- settings -----------------------------------------------------
    def _apply_settings(self, s: Settings) -> None:
        for combo, value in ((self.language, s.language), (self.accuracy, s.beam_size),
                             (self.model, s.model)):
            if combo.findData(value) < 0:
                combo.addItem(str(value), value)
            combo.setCurrentIndex(combo.findData(value))
        for key, box in self.format_boxes.items():
            box.setChecked(key in s.formats)
        self.phone_mode.setChecked(s.phone_mode)
        self.next_to_audio.setChecked(s.next_to_audio)
        self.output_dir.setText(s.output_dir)

    def _connect_settings(self) -> None:
        for combo in (self.language, self.accuracy, self.model):
            combo.currentIndexChanged.connect(self._settings_changed)
        for box in (*self.format_boxes.values(), self.phone_mode, self.next_to_audio):
            box.toggled.connect(self._settings_changed)
        self.output_dir.editingFinished.connect(self._settings_changed)

    def current_settings(self) -> Settings:
        return Settings(
            model=self.model.currentData(),
            language=self.language.currentData(),
            beam_size=self.accuracy.currentData(),
            formats=tuple(k for k, box in self.format_boxes.items() if box.isChecked()),
            phone_mode=self.phone_mode.isChecked(),
            next_to_audio=self.next_to_audio.isChecked(),
            output_dir=self.output_dir.text().strip(),
        )

    def _settings_changed(self, *_) -> None:
        self.current_settings().save(self.qsettings)
        self._refresh_controls()

    @staticmethod
    def _settings_problem(s: Settings):
        if not s.formats:
            return "Tick at least one file type to save."
        if not s.next_to_audio and not s.output_dir:
            return "Choose an output folder, or tick 'Next to the audio file'."
        return None

    # ---- queue --------------------------------------------------------
    def add_paths(self, paths, check_ext: bool = True) -> None:
        s = self.current_settings()
        problem = self._settings_problem(s)
        if problem:
            QMessageBox.warning(self, "AudtoText", problem)
            return
        files, skipped = expand_inputs(paths, check_ext)
        active = {j.audio.resolve() for j in self.jobs if j.state in ("queued", "running")}
        added = 0
        for f in files:
            if f.resolve() in active:
                skipped.append(f"{f.name} (already in the list)")
                continue
            item = QTreeWidgetItem([f.name, "", "", "Waiting"])
            item.setToolTip(0, str(f))
            self.queue.addTopLevelItem(item)
            self.jobs.append(Job(audio=f, settings=s, item=item))
            active.add(f.resolve())
            added += 1
        notes = [f"Added {added} file(s)."] if added else []
        if skipped:
            notes.append("Skipped: " + ", ".join(skipped))
        self.statusBar().showMessage(" ".join(notes), 15000)
        self._start_next()

    def _start_next(self) -> None:
        if self.proc is not None or self.closing:
            return
        job = next((j for j in self.jobs if j.state == "queued"), None)
        if job is None:
            self._refresh_controls()
            return
        job.state, job.stage = "running", "Loading model…"
        self.current = job
        self._partial = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        proc = QProcess(self)
        proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        env = QProcessEnvironment.systemEnvironment()
        env.insert("PYTHONIOENCODING", "utf-8")
        env.insert("PYTHONUNBUFFERED", "1")
        proc.setProcessEnvironment(env)
        proc.readyReadStandardOutput.connect(self._on_output)
        proc.finished.connect(self._on_finished)
        proc.errorOccurred.connect(self._on_process_error)
        cmd = build_command(engine_python(), job.audio, job.settings, self.model_cache_dir)
        job.log.append("$ " + subprocess.list2cmdline(cmd))
        self.proc = proc
        proc.start(cmd[0], cmd[1:])
        if self.queue.currentItem() is None:
            self.queue.setCurrentItem(job.item)
        self._update_job(job)
        self._refresh_controls()

    def _on_output(self) -> None:
        text = self._decoder.decode(bytes(self.proc.readAllStandardOutput()))
        lines = (self._partial + text).split("\n")
        self._partial = lines.pop()
        for raw in lines:
            # Download bars redraw with \r; keep only the latest state of the line.
            self._handle_line(self.current, raw.rstrip("\r").split("\r")[-1])

    def _handle_line(self, job: Job, line: str) -> None:
        job.log.append(line)
        if self.queue.currentItem() is job.item:
            self.log_view.appendPlainText(line)
        ev = parse_line(line)
        if ev is None:
            return
        kind = ev["event"]
        if kind == "loading":
            job.stage = "Loading model…"
        elif kind == "started":
            job.stage = "Detecting language…" if job.settings.language == "auto" else "Reading audio…"
        elif kind == "info":
            job.language, job.language_prob, job.duration = ev["language"], ev["prob"], ev["duration"]
            job.stage, job.pct = "Transcribing…", 0.0
        elif kind == "progress":
            job.pct = ev["pct"]
            left = ev["elapsed"] * (100 - job.pct) / job.pct if job.pct > 0 else 0.0
            job.stage = f"Transcribing {job.pct:.0f}% · {fmt_clock(left)} left"
        elif kind == "finished":
            job.run_seconds, job.pct, job.stage = ev["seconds"], 100.0, "Saving…"
        elif kind == "saved":
            job.outputs[ev["format"]] = Path(ev["path"])
        elif kind == "done":
            job.saw_done = True
        self._update_job(job)

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        job = self.current
        tail = self._partial + self._decoder.decode(b"", final=True)
        if tail:
            self._handle_line(job, tail.rstrip("\r").split("\r")[-1])
        code = exit_code & 0xFFFFFFFF
        job.log.append(f"[exit] code=0x{code:08X} status={exit_status.name}")
        if job.cancel_requested:
            job.state, job.stage = "cancelled", "Cancelled"
        elif exit_status == QProcess.ExitStatus.NormalExit and code == 0 and job.saw_done:
            job.state, job.stage = "done", f"Done in {fmt_clock(job.run_seconds)}"
        else:
            shown = str(code) if code < 256 else f"0x{code:08X}"
            job.state, job.stage = "failed", f"Failed (exit code {shown}), see Log"
        self._finish_job(job)

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        # Crashes and kills also arrive through finished(); only a failed
        # start has no finished() to follow it.
        if error != QProcess.ProcessError.FailedToStart:
            return
        job = self.current
        job.log.append(f"[error] could not start {engine_python()}: {self.proc.errorString()}")
        job.state, job.stage = "failed", "Failed to start Python, see Log"
        self._finish_job(job)

    def _finish_job(self, job: Job) -> None:
        self.proc.deleteLater()
        self.proc = None
        self.current = None
        self._update_job(job)
        if job.state in ("done", "failed"):
            self.queue.setCurrentItem(job.item)
        self._show_selected()
        self._refresh_controls()
        QTimer.singleShot(0, self._start_next)

    def _cancel_current(self) -> None:
        if self.proc is None:
            return
        self.current.cancel_requested = True
        self.current.stage = "Cancelling…"
        self._update_job(self.current)
        self.proc.kill()

    def _clear_list(self) -> None:
        for job in [j for j in self.jobs if j.state != "running"]:
            self.queue.takeTopLevelItem(self.queue.indexOfTopLevelItem(job.item))
            self.jobs.remove(job)
        self._show_selected()

    # ---- display ------------------------------------------------------
    def _update_job(self, job: Job) -> None:
        item = job.item
        item.setText(1, fmt_clock(job.duration) if job.duration else "")
        item.setText(2, f"{job.language} ({job.language_prob:.2f})" if job.language else "")
        item.setText(3, job.stage)
        if job.state == "failed":
            item.setForeground(3, QBrush(FAILED_COLOR))
        else:
            item.setData(3, Qt.ItemDataRole.ForegroundRole, None)
        if job is self.current:
            if job.pct < 0:
                self.progress.setRange(0, 0)  # busy until the first progress line
            else:
                self.progress.setRange(0, 1000)
                self.progress.setValue(int(job.pct * 10))
                self.progress.setFormat(f"{job.audio.name}  ·  {job.pct:.0f}%")

    def _refresh_controls(self) -> None:
        s = self.current_settings()
        self.output_dir.setEnabled(not s.next_to_audio)
        self.browse_btn.setEnabled(not s.next_to_audio)
        where = ("next to each file" if s.next_to_audio
                 else f"to {s.output_dir}" if s.output_dir else "to the folder you choose below")
        self.drop_zone.hint.setText(f"or click to choose files  ·  transcripts are saved {where}")
        self.cancel_btn.setEnabled(self.proc is not None)
        if self.proc is None:
            done = sum(j.state == "done" for j in self.jobs)
            self.progress.setRange(0, 1)
            self.progress.setValue(1 if done else 0)
            self.progress.setFormat(f"{done} done" if done else "Ready")

    def _selected_job(self):
        item = self.queue.currentItem()
        return next((j for j in self.jobs if j.item is item), None)

    @staticmethod
    def _transcript_path(job):
        if job is None:
            return None
        for fmt in ("txt", "timestamped"):
            path = job.outputs.get(fmt)
            if path is not None and path.is_file():
                return path
        return None

    def _show_selected(self, *_) -> None:
        job = self._selected_job()
        self.log_view.setPlainText("\n".join(job.log) if job else "")
        path = self._transcript_path(job)
        if path is None:
            self.transcript.clear()
        else:
            self.transcript.setPlainText(path.read_text(encoding="utf-8"))
        for b in (self.copy_btn, self.open_btn, self.reveal_btn):
            b.setEnabled(path is not None)

    def _copy_text(self) -> None:
        text = self.transcript.toPlainText()
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage(f"Copied {len(text.split()):,} words.", 5000)

    def _open_transcript(self) -> None:
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._transcript_path(self._selected_job()))))

    def _reveal_transcript(self) -> None:
        path = self._transcript_path(self._selected_job())
        if sys.platform == "win32":
            subprocess.Popen(f'explorer /select,"{path}"')
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))

    # ---- input --------------------------------------------------------
    def _choose_files(self) -> None:
        patterns = " ".join(f"*{e}" for e in MEDIA_EXTS)
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Choose audio or video files", "",
            f"Audio and video ({patterns});;All files (*)")
        if paths:
            self.add_paths(paths, check_ext=False)

    def _choose_output_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Choose output folder", self.output_dir.text())
        if folder:
            self.output_dir.setText(QDir.toNativeSeparators(folder))
            self._settings_changed()

    def dragEnterEvent(self, event):
        if any(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()
            self.drop_zone.set_hover(True)

    def dragMoveEvent(self, event):
        if any(u.isLocalFile() for u in event.mimeData().urls()):
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self.drop_zone.set_hover(False)

    def dropEvent(self, event):
        self.drop_zone.set_hover(False)
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        event.acceptProposedAction()
        self.add_paths(paths)

    def _on_forwarded_connection(self) -> None:
        while self.server.hasPendingConnections():
            sock = self.server.nextPendingConnection()
            buf = bytearray()
            sock.readyRead.connect(lambda s=sock, b=buf: b.extend(bytes(s.readAll())))
            sock.disconnected.connect(lambda s=sock, b=buf: self._receive_forwarded(s, b))

    def _receive_forwarded(self, sock: QLocalSocket, buf: bytearray) -> None:
        buf.extend(bytes(sock.readAll()))
        sock.deleteLater()
        paths = [p for p in bytes(buf).decode("utf-8").split("\n") if p]
        self.showNormal()
        self.raise_()
        self.activateWindow()
        if paths:
            self.add_paths(paths)

    def closeEvent(self, event):
        if self.proc is not None:
            answer = QMessageBox.question(
                self, "AudtoText",
                f"{self.current.audio.name} is still being transcribed. Stop it and quit?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.closing = True
            self.current.cancel_requested = True
            self.proc.kill()
            self.proc.waitForFinished(5000)
        event.accept()


def show_exception(exc_type, exc, tb) -> None:
    """pythonw.exe has no console, so uncaught errors are shown in a dialog."""
    QMessageBox.critical(None, "AudtoText error",
                         "".join(traceback.format_exception(exc_type, exc, tb)))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="audtotext_gui",
                                description="Drag-and-drop window for audtotext.py.")
    p.add_argument("files", nargs="*", help="audio/video files to queue at start")
    p.add_argument("--model-cache-dir", type=Path, default=None,
                   help="passed to audtotext.py --model-cache-dir")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    app = QApplication(sys.argv[:1])
    app.setApplicationName("AudtoText")
    files = [str(Path(f).resolve()) for f in args.files]
    if forward_to_running_window(files):
        return 0
    sys.excepthook = show_exception
    app.setStyleSheet(STYLE)
    app.setWindowIcon(QIcon(str(ICON_PATH)))
    window = MainWindow(args.model_cache_dir)
    window.show()
    if files:
        window.add_paths(files)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
