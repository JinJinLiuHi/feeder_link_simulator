"""Embedded full pages for link calculation."""
from importlib import import_module
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor,QPalette
from PySide6.QtWidgets import QDialog,QFrame,QLabel,QScrollArea,QSizePolicy,QVBoxLayout,QWidget
from link_budget.ui.styles import APP_STYLE

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
"""

class LazyDialogPage(QWidget):
    def __init__(self,module_name,class_name,parent=None):
        super().__init__(parent);self.setObjectName("linkModulePage");self.setStyleSheet(EMBEDDED_STYLE);self.module_name=module_name;self.class_name=class_name;self.loaded=False
        self.layout=QVBoxLayout(self);self.layout.setContentsMargins(0,0,0,0);self.placeholder=QLabel("正在准备模块…");self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter);self.layout.addWidget(self.placeholder,1)
    def ensure_loaded(self):
        if self.loaded:return
        self.loaded=True
        try:
            cls=getattr(import_module(self.module_name),self.class_name);dialog=cls(parent=self);dialog.setWindowFlags(Qt.WindowType.Widget);dialog.setWindowState(Qt.WindowState.WindowNoState);dialog.setModal(False);dialog.setMinimumSize(0,0);dialog.setSizePolicy(QSizePolicy.Policy.Expanding,QSizePolicy.Policy.Expanding);dialog.setStyleSheet(EMBEDDED_STYLE)
            wrapper=dialog.findChild(QFrame,"moduleContent")
            if wrapper:wrapper.setStyleSheet("QFrame#moduleContent{background:#07131f;color:#d9ecf7}")
            self.content_scroll=QScrollArea();self.content_scroll.setWidgetResizable(True);self.content_scroll.setFrameShape(QScrollArea.Shape.NoFrame);self.content_scroll.setWidget(dialog)
            self.layout.replaceWidget(self.placeholder,self.content_scroll);self.placeholder.deleteLater();self.dialog=dialog;dialog.show()
        except Exception as exc:self.placeholder.setText(f"模块加载失败\n{type(exc).__name__}: {exc}");self.placeholder.setStyleSheet("color:#ff7b89;font-size:14px")
    def showEvent(self,event):super().showEvent(event);self.ensure_loaded()

def calculation_pages(parent=None):
    definitions=(("链路预算","link_budget.algorithm.Link_budget","LinkBudgetDialog"),("可用性分析","link_budget.algorithm.channel_model","ChannelModelDialog"),("雨衰序列","link_budget.algorithm.rain_sequence_widget","RainSequenceDialog"),("调制编码","link_budget.algorithm.ber_analysis","BERAnalysisDialog"))
    return [(title,LazyDialogPage(module,cls,parent)) for title,module,cls in definitions]
