"""
全局样式表
Global Stylesheet
"""


def light_palette():
    """Keep unstyled containers and native controls consistent with APP_STYLE."""
    from PySide6.QtGui import QColor, QPalette

    palette = QPalette()
    colors = {
        QPalette.ColorRole.Window: "#07131F",
        QPalette.ColorRole.WindowText: "#C9E6F2",
        QPalette.ColorRole.Base: "#091B29",
        QPalette.ColorRole.AlternateBase: "#07131F",
        QPalette.ColorRole.Text: "#DFF8FF",
        QPalette.ColorRole.Button: "#091B29",
        QPalette.ColorRole.ButtonText: "#C9E6F2",
        QPalette.ColorRole.Highlight: "#12617B",
        QPalette.ColorRole.HighlightedText: "#DFF8FF",
        QPalette.ColorRole.ToolTipBase: "#091B29",
        QPalette.ColorRole.ToolTipText: "#DFF8FF",
        QPalette.ColorRole.PlaceholderText: "#648A9C",
        QPalette.ColorRole.Link: "#00CFE8",
        QPalette.ColorRole.Light: "#091B29",
        QPalette.ColorRole.Midlight: "#103C50",
        QPalette.ColorRole.Mid: "#1C4961",
        QPalette.ColorRole.Dark: "#39728A",
        QPalette.ColorRole.Shadow: "#648A9C",
    }
    for role, color in colors.items():
        palette.setColor(role, QColor(color))
    for role in (QPalette.ColorRole.Text, QPalette.ColorRole.WindowText,
                 QPalette.ColorRole.ButtonText):
        palette.setColor(QPalette.ColorGroup.Disabled, role, QColor("#648A9C"))
    return palette


APP_STYLE = """
/* Parameter containers must keep a light background under a dark system theme. */
QWidget#berParameterContent, QWidget#berParameterPanel,
QWidget#berResults, QWidget#berScrollViewport,
QScrollArea#berParameterScroll,
QStackedWidget#codingPages, QWidget#codingTrellisPage, QWidget#codingMatrixPage,
QWidget#codingParameterContent {
    background-color: #07131F;
    color: #C9E6F2;
}

QMainWindow {
    background-color: #07131F;
}

/* ── 菜单栏 ── */
QMenuBar {
    background-color: #091B29;
    border-bottom: 1px solid #193F55;
    padding: 2px 8px;
    font-size: 10pt;
    color: #C9E6F2;
    spacing: 2px;
}
QMenuBar::item {
    background: transparent;
    padding: 6px 14px;
    border-radius: 5px;
    color: #91AFC2;
}
QMenuBar::item:selected {
    background-color: #103C50;
    color: #DFF8FF;
}
QMenuBar::item:pressed {
    background-color: #0D3042;
    color: #DFF8FF;
}

/* ── 下拉菜单 ── */
QMenu {
    background-color: #091B29;
    border: 1px solid #1C4961;
    border-radius: 8px;
    padding: 6px;
    font-size: 10pt;
    color: #C9E6F2;
}
QMenu::item {
    padding: 8px 16px 8px 12px;
    border-radius: 5px;
    color: #91AFC2;
}
QMenu::item:selected {
    background-color: #103C50;
    color: #DFF8FF;
}
QMenu::separator {
    height: 1px;
    background: #193F55;
    margin: 4px 8px;
}

/* ── 状态栏 ── */
QStatusBar {
    background-color: #091B29;
    border-top: 1px solid #193F55;
    font-size: 8pt;
    color: #648A9C;
    padding: 0 8px;
}
QStatusBar::item {
    border: none;
}

/* ── 通用对话框 ── */
QDialog {
    background-color: #091B29;
    border-radius: 10px;
}

/* ── 标签 ── */
QLabel {
    color: #C9E6F2;
    font-size: 10pt;
}

/* ── 输入框 ── */
QLineEdit {
    background-color: #091B29;
    border: 1px solid #1C4961;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 10pt;
    color: #DFF8FF;
    selection-background-color: #12617B;
    selection-color: #DFF8FF;
}
QLineEdit:focus {
    border: 1.5px solid #00CFE8;
    outline: none;
}
QLineEdit:hover {
    border-color: #39728A;
}

QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {
    background-color: #091B29;
    color: #DFF8FF;
    border: 1px solid #1C4961;
    border-radius: 6px;
    padding: 5px 8px;
    selection-background-color: #12617B;
    selection-color: #DFF8FF;
}
QCheckBox, QRadioButton {
    color: #C9E6F2;
}

/* ── 按钮 ── */
QPushButton {
    background-color: #091B29;
    border: 1px solid #1C4961;
    border-radius: 6px;
    padding: 7px 16px;
    font-size: 10pt;
    color: #C9E6F2;
}
QPushButton:hover {
    background-color: #103C50;
    border-color: #39728A;
}
QPushButton:pressed {
    background-color: #0D3042;
}
QPushButton.primary {
    background-color: #00CFE8;
    border-color: #00CFE8;
    color: #091B29;
}
QPushButton.primary:hover {
    background-color: #185FA5;
    border-color: #185FA5;
}

/* ── 分组框 ── */
QGroupBox {
    background-color: #091B29;
    border: 1px solid #193F55;
    border-radius: 8px;
    margin-top: 12px;
    padding: 12px;
    font-size: 10pt;
    font-weight: 500;
    color: #C9E6F2;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    left: 12px;
    color: #91AFC2;
    font-size: 9pt;
    font-weight: 500;
}

/* ── 下拉框 ── */
QComboBox {
    background-color: #091B29;
    border: 1px solid #1C4961;
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 10pt;
    color: #DFF8FF;
    min-width: 120px;
}
QComboBox:focus {
    border-color: #00CFE8;
}
QComboBox::drop-down {
    border: none;
    width: 20px;
}
QComboBox QFrame {
    background-color: #091B29;
    border: 1px solid #1C4961;
}
QComboBox QAbstractItemView {
    background-color: #091B29;
    color: #DFF8FF;
    border: 1px solid #1C4961;
    border-radius: 6px;
    selection-background-color: #12617B;
    selection-color: #DFF8FF;
    padding: 4px;
    outline: 0;
}
QComboBox QAbstractItemView::item {
    background-color: #091B29;
    color: #DFF8FF;
    min-height: 26px;
    padding: 3px 8px;
}
QComboBox QAbstractItemView::item:selected,
QComboBox QAbstractItemView::item:hover {
    background-color: #12617B;
    color: #DFF8FF;
}
QComboBox QAbstractItemView::item:disabled {
    color: #648A9C;
}

/* ── 表格 ── */
QTableWidget {
    background-color: #091B29;
    alternate-background-color: #07131F;
    border: 1px solid #193F55;
    border-radius: 6px;
    gridline-color: #103C50;
    font-size: 10pt;
    color: #DFF8FF;
    selection-background-color: #12617B;
    selection-color: #FFFFFF;
}
QTableWidget::item {
    padding: 6px 10px;
}
QTableWidget::item:selected, QTableWidget::item:selected:!active {
    background-color: #12617B;
    color: #FFFFFF;
}
QHeaderView::section {
    background-color: #0B2B3D;
    border: none;
    border-bottom: 1px solid #193F55;
    border-right: 1px solid #193F55;
    padding: 6px 10px;
    font-size: 9pt;
    font-weight: 500;
    color: #BFE8F5;
}

/* ── 滚动条 ── */
QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 0;
}
QScrollBar::handle:vertical {
    background: #1C4961;
    border-radius: 4px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: #39728A;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}
QScrollBar:horizontal {
    background: transparent;
    height: 8px;
}
QScrollBar::handle:horizontal {
    background: #1C4961;
    border-radius: 4px;
    min-width: 24px;
}
QScrollBar::handle:horizontal:hover {
    background: #39728A;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* ── Tab 组件 ── */
QTabWidget::pane {
    background-color: #091B29;
    border: 1px solid #193F55;
    border-radius: 8px;
    top: -1px;
}
QTabBar::tab {
    background-color: transparent;
    border: none;
    padding: 8px 16px;
    font-size: 10pt;
    color: #648A9C;
    border-bottom: 2px solid transparent;
}
QTabBar::tab:selected {
    color: #DFF8FF;
    border-bottom: 2px solid #00CFE8;
    font-weight: 500;
}
QTabBar::tab:hover:!selected {
    color: #C9E6F2;
    background-color: #103C50;
    border-radius: 6px 6px 0 0;
}

/* ── Splitter ── */
QSplitter::handle {
    background-color: #193F55;
}
QSplitter::handle:horizontal {
    width: 1px;
}
QSplitter::handle:vertical {
    height: 1px;
}
"""
