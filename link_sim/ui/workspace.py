"""Embedded full pages for channel simulation and model library."""
from importlib import import_module
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame,QLabel,QSizePolicy,QTabWidget,QVBoxLayout,QWidget
from link_sim.ui.styles import APP_STYLE

def _configure_charts():
    try:
        from matplotlib import rcParams
        rcParams.update({"figure.facecolor":"#07131f","axes.facecolor":"#091b29","axes.edgecolor":"#39728a","axes.labelcolor":"#d9ecf7","text.color":"#d9ecf7","xtick.color":"#91afc2","ytick.color":"#91afc2","grid.color":"#17384a","legend.facecolor":"#091b29","legend.edgecolor":"#39728a"})
    except Exception:pass

_configure_charts()

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
    def __init__(self,module_name,class_name,parent=None):
        super().__init__(parent);self.setObjectName("linkModulePage");self.setStyleSheet(EMBEDDED_STYLE);self.module_name=module_name;self.class_name=class_name;self.loaded=False;self.layout=QVBoxLayout(self);self.layout.setContentsMargins(0,0,0,0);self.placeholder=QLabel("正在准备模块…");self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter);self.layout.addWidget(self.placeholder,1)
    def ensure_loaded(self):
        if self.loaded:return
        self.loaded=True
        try:
            cls=getattr(import_module(self.module_name),self.class_name);dialog=cls(parent=self);dialog.setWindowFlags(Qt.WindowType.Widget);dialog.setWindowState(Qt.WindowState.WindowNoState);dialog.setModal(False);dialog.setMinimumSize(0,0);dialog.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding);dialog.setStyleSheet(EMBEDDED_STYLE)
            wrapper=dialog.findChild(QFrame,"moduleContent")
            if wrapper:wrapper.setStyleSheet("QFrame#moduleContent{background:#07131f;color:#d9ecf7}")
            self.layout.replaceWidget(self.placeholder,dialog);self.placeholder.deleteLater();self.dialog=dialog;dialog.show()
        except Exception as exc:self.placeholder.setText(f"模块加载失败\n{type(exc).__name__}: {exc}");self.placeholder.setStyleSheet("color:#ff7b89;font-size:14px")
    def showEvent(self,event):super().showEvent(event);self.ensure_loaded()

class ChannelWorkspace(QWidget):
    def __init__(self,parent=None):
        super().__init__(parent);self.setStyleSheet(EMBEDDED_STYLE);layout=QVBoxLayout(self);layout.setContentsMargins(0,0,0,0);tabs=QTabWidget();tabs.setDocumentMode(True);layout.addWidget(tabs)
        definitions=(("功放","link_sim.algorithm.pa_model","PAModelDialog"),("ADC/DAC","link_sim.algorithm.ad_model","ADDAModelDialog"),("滤波器","link_sim.algorithm.filter_model","FilterModelDialog"),("混频器","link_sim.algorithm.mixer_model","MixerModelDialog"))
        for title,module,cls in definitions:tabs.addTab(LazyDialogPage(module,cls,tabs),title)

def simulation_pages(parent=None):
    return [("信道仿真",ChannelWorkspace(parent)),("模型库",LazyDialogPage("link_sim.ui.model_library","ModelLibraryDialog",parent))]
