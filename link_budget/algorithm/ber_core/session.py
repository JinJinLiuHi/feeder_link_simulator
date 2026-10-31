"""Qt-owned process lifecycle shared by simulation and code validation."""

import multiprocessing as mp
import queue
import time
from PySide6.QtCore import QObject, QTimer, Signal


class ProcessSession(QObject):
    message = Signal(object)
    ended = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.process = None
        self.messages = None
        self.cancel = None
        self.dead_since = None
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.poll)

    def start(self, action, request):
        from .simulation import worker

        if self.process is not None:
            raise RuntimeError("已有任务运行中")
        ctx = mp.get_context("spawn")
        self.messages = ctx.Queue()
        self.cancel = ctx.Event()
        self.process = ctx.Process(
            target=worker, args=(action, request, self.messages, self.cancel), daemon=True
        )
        self.dead_since = None
        try:
            self.process.start()
        except Exception:
            self.dispose()
            raise
        self.timer.start()

    def stop(self):
        if self.cancel is not None:
            self.cancel.set()

    def poll(self):
        if self.messages is None:
            return
        terminal = False
        for _ in range(1000):
            try:
                message = self.messages.get_nowait()
            except queue.Empty:
                break
            self.message.emit(message)
            terminal |= message["type"] in ("done", "validated", "error")
        if terminal:
            self.dispose()
            self.ended.emit()
            return
        if self.process is not None and not self.process.is_alive():
            self.dead_since = self.dead_since or time.monotonic()
            if time.monotonic() - self.dead_since > 1:
                self.message.emit(
                    {
                        "type": "error",
                        "message": f"仿真进程意外结束（退出码 {self.process.exitcode}）",
                    }
                )
                self.dispose()
                self.ended.emit()

    def dispose(self):
        self.timer.stop()
        if self.cancel is not None:
            self.cancel.set()
        if self.process is not None and self.process.pid is not None:
            self.process.join(timeout=0.3)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=1)
            self.process.close()
        if self.messages is not None:
            self.messages.close()
            self.messages.cancel_join_thread()
        self.process = self.messages = self.cancel = None
