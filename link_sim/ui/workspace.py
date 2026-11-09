"""Embedded full pages for channel simulation and model library."""
from importlib import import_module
from concurrent.futures import ThreadPoolExecutor
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel,QProgressBar,QSizePolicy,QTabWidget,QVBoxLayout,QWidget
from link_sim.ui.styles import APP_STYLE

def _configure_charts():
    try:
        from matplotlib import rcParams
        rcParams.update({"figure.facecolor":"#07131f","axes.facecolor":"#091b29","axes.edgecolor":"#39728a","axes.labelcolor":"#d9ecf7","text.color":"#d9ecf7","xtick.color":"#91afc2","ytick.color":"#91afc2","grid.color":"#17384a","legend.facecolor":"#091b29","legend.edgecolor":"#39728a"})
    except Exception:pass

# Serialize imports; Qt widgets are always constructed on the GUI thread.
_import_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="link-import")

def _load_class(module_name, class_name):
    cls = getattr(import_module(module_name), class_name)
    # These are otherwise imported while constructing the first PA editor.
    if module_name == "link_sim.algorithm.pa_model":
        import_module("link_sim.algorithm.pa_simulation")
        import_module("link_sim.algorithm.pa_dpd")
    if module_name.startswith("link_sim.algorithm."):
        import_module("link_sim.ui.model_library")
    _configure_charts()
    return cls

EMBEDDED_STYLE=APP_STYLE+"""
QDialog,QWidget#linkModulePage,QFrame#moduleContent{background:#07131f;color:#d9ecf7}
QSplitter::handle{background:#17384a;width:2px}
QTableWidget{gridline-color:#17384a;alternate-background-color:#0b2638}
QHeaderView::section{background:#0b2b3d;color:#bfe8f5;border:0;border-right:1px solid #17384a;padding:6px}
QScrollArea{border:0;background:#07131f} QScrollArea>QWidget>QWidget{background:#07131f}
QTabWidget::pane{border:1px solid #17384a;background:#07131f}
QTabBar::tab{min-width:72px;padding:7px 15px;background:#0a2434;color:#8baebe;border:0}
QTabBar::tab:selected{background:#0d6078;color:white;border-bottom:2px solid #42dff8}
"""

class LazyDialogPage(QWidget):
    import_finished = Signal(object, object)
    load_finished = Signal(bool, str)

    def __init__(self, module_name, class_name, parent=None):
        super().__init__(parent)
        self.setObjectName("linkModulePage")
        self.setStyleSheet(EMBEDDED_STYLE)
        self.module_name, self.class_name = module_name, class_name
        self.loaded = self.loading = False
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.placeholder = QLabel("正在准备模块…")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.placeholder, 1)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setTextVisible(False)
        self.layout.addWidget(self.progress)
        self.import_finished.connect(self._finish_load, Qt.ConnectionType.QueuedConnection)

    def ensure_loaded(self, *, background=False):
        if self.loaded or self.loading:
            return
        self.loading = True
        self.progress.show()
        if background:
            future = _import_pool.submit(_load_class, self.module_name, self.class_name)
            def completed(future):
                try:
                    cls, error = future.result(), None
                except Exception as exc:
                    cls, error = None, exc
                try:
                    self.import_finished.emit(cls, error)
                except RuntimeError:
                    pass  # Page closed while the import was in progress.
            future.add_done_callback(completed)
        else:
            try:
                cls = _load_class(self.module_name, self.class_name)
            except Exception as exc:
                self._finish_load(None, exc)
            else:
                self._finish_load(cls, None)

    def _finish_load(self, cls, error):
        self.loading = False
        self.progress.hide()
        message = ""
        try:
            if error is not None:
                raise error
            dialog = cls(parent=self)
            dialog.setWindowFlags(Qt.WindowType.Widget)
            dialog.setWindowState(Qt.WindowState.WindowNoState)
            dialog.setModal(False)
            dialog.setMinimumSize(0, 0)
            dialog.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            dialog.setStyleSheet(EMBEDDED_STYLE)
            self.layout.replaceWidget(self.placeholder, dialog)
            self.placeholder.hide()
            self.dialog = dialog
            self.loaded = True
            dialog.show()
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            self.placeholder.setText(f"模块加载失败\n{message}")
            self.placeholder.setStyleSheet("color:#ff7b89;font-size:14px")
        self.load_finished.emit(self.loaded, message)

    def showEvent(self, event):
        super().showEvent(event)
        self.ensure_loaded(background=True)

class ChannelWorkspace(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent);self.setStyleSheet(EMBEDDED_STYLE);layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0);tabs=QTabWidget();tabs.setDocumentMode(True);layout.addWidget(tabs)
        definitions=(("功放","link_sim.algorithm.pa_model","PAModelDialog"),
                     ("滤波器","link_sim.algorithm.filter_model","FilterModelDialog"),
                     ("混频器","link_sim.algorithm.mixer_model","MixerModelDialog"),
                     ("ADC/DAC","link_sim.algorithm.ad_model","ADDAModelDialog"),
                     ("底噪模型","link_sim.algorithm.noise_model","NoiseModelDialog"))
        for title,module,cls in definitions:tabs.addTab(LazyDialogPage(module,cls,tabs),title)

def simulation_pages(parent=None):
    return [("信道仿真",ChannelWorkspace(parent)),
            ("模型库",LazyDialogPage("link_sim.ui.model_library","ModelLibraryDialog",parent)),
            ("全链路仿真",LazyDialogPage("link_sim.ui.waveform_workspace","WaveformWorkspace",parent))]
