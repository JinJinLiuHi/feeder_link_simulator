"""嵌入误码率分析对话框的雨衰序列生成功能页。"""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QPushButton, QScrollArea, QSplitter, QGridLayout,
    QVBoxLayout, QWidget,
)

from .rain_sequence_core.analysis import acf_curve, ccdf_curve
from .rain_sequence_core.config import HYDROMETEOR_PHASE_MODES, ModelConfig, OutputConfig, SimulationConfig
from .rain_sequence_core.engine import SimulationResult, simulate_channel
from .rain_sequence_core.exporter import export_result
from .rain_sequence_core.station_store import StationStore
from link_budget.ui.base_dialog import ModuleDialog


RAIN_MODELS = (
    "ONERA双过程", "ITU-R P.1853", "Maseng–Bakken", "SST/E-SST",
    "EXCELL/Multi-EXCELL", "Gamma相关过程", "融合模型",
)

MODEL_PARAM_DEFS = {
    "ONERA双过程": (("fast_tau_s", "快过程时间/s", 2.5), ("slow_tau_s", "慢过程时间/s", 75.0), ("onera_fast_sigma", "快过程σ", .28), ("onera_slow_shape", "慢过程Gamma形状", 2.2), ("onera_peak_scale_db", "慢过程尺度/dB", 4.2), ("event_mean_s", "事件平均持续/s", 90.0), ("event_gap_s", "事件平均间隔/s", 180.0), ("onera_rise_fraction", "峰值相对位置", .42), ("onera_cell_length_km", "等效雨胞长度/km", 1.0)),
    "ITU-R P.1853": (("itu_tau_s", "相关时间/s", 12.0), ("itu_gamma_shape", "Gamma形状", 1.8), ("itu_marginal_scale", "边缘尺度", 2.15), ("itu_fast_tau_s", "快过程时间/s", 1.8), ("itu_fast_sigma", "快过程σ", .18), ("event_mean_s", "事件平均持续/s", 120.0), ("event_gap_s", "事件平均间隔/s", 240.0), ("itu_rise_fraction", "上升段占比", .34), ("itu_cell_length_km", "等效雨胞长度/km", 1.0)),
    "Maseng–Bakken": (("mb_beta_s_inv", "回归系数β/(1/s)", .055), ("mb_log_mean", "对数域均值", 1.05), ("mb_log_sigma", "对数域σ", .72), ("mb_floor_db", "衰减下限/dB", .15), ("mb_intermittency", "间歇系数", .20), ("mb_cell_length_km", "等效雨胞长度/km", 1.0)),
    "SST/E-SST": (("sst_wind_speed_m_s", "平流风速/(m/s)", 12.0), ("sst_cell_length_km", "雨胞相关长度/km", 2.4), ("sst_spectral_slope", "空间谱斜率", 1.67), ("sst_convective_ratio", "对流雨占比", .68), ("sst_melting_factor", "融化层增强", 1.08), ("event_mean_s", "事件平均持续/s", 150.0), ("event_gap_s", "事件平均间隔/s", 190.0)),
    "EXCELL/Multi-EXCELL": (("excell_cell_count", "雨胞数量", 5.0), ("excell_radius_km", "雨胞半径/km", 1.1), ("excell_wind_speed_m_s", "平流风速/(m/s)", 12.0), ("excell_peak_shape", "峰值Gamma形状", 2.2), ("excell_peak_scale_db", "峰值尺度/dB", 4.0), ("excell_radial_power", "径向衰减指数", 1.7), ("excell_background_db", "层状雨背景/dB", 0.0)),
    "Gamma相关过程": (("gamma_tau_s", "相关时间/s", 45.0), ("gamma_shape", "Gamma形状", 2.2), ("gamma_scale_db", "Gamma尺度/dB", 3.8), ("gamma_dry_probability", "无雨概率", .45), ("gamma_cell_length_km", "等效雨胞长度/km", 1.0)),
    "融合模型": (("fusion_dynamic_weight", "ONERA动态权重", .65), ("fusion_event_weight", "EXCELL事件权重", .35), ("fusion_tail_boost", "强衰减尾部增强", 1.15), ("fusion_target_p99_db", "验证目标P99/dB", 25.0), ("fast_tau_s", "快过程时间/s", 2.5), ("slow_tau_s", "慢过程时间/s", 75.0), ("excell_cell_count", "事件核雨胞数量", 5.0), ("excell_radius_km", "事件核雨胞半径/km", 1.1), ("fusion_cell_length_km", "融合雨胞长度/km", 1.2)),
}


_EDIT_STYLE = (
    "QLineEdit{background:#091B29;border:1px solid #1C4961;border-radius:3px;"
    "padding:3px 6px;font-size:10pt;color:#D9ECF7;}"
    "QLineEdit:focus{border:1.5px solid #1D9E75;}"
)
_COMBO_STYLE = (
    "QComboBox{background:#091B29;border:1px solid #1C4961;border-radius:3px;"
    "padding:3px 8px;font-size:10pt;color:#D9ECF7;}"
    "QComboBox QAbstractItemView{background:#091B29;color:#D9ECF7;selection-background-color:#12617B;}"
)
_GROUP_STYLE = (
    "QGroupBox{background:#091B29;border:1px solid #1C4961;border-radius:6px;"
    "margin-top:8px;padding:6px 8px;}"
    "QGroupBox::title{subcontrol-origin:margin;left:10px;padding:0 4px;"
    "color:#1D9E75;font-size:9pt;font-weight:bold;}"
)


def _group(title: str) -> QGroupBox:
    group = QGroupBox(title)
    group.setStyleSheet(_GROUP_STYLE)
    layout = QVBoxLayout(group)
    layout.setContentsMargins(6, 4, 6, 6)
    layout.setSpacing(4)
    return group


class RainSequenceWidget(QWidget):
    """雨衰时序的原生 PyQt 页面，不启动外部程序。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: SimulationResult | None = None
        self._results: list[SimulationResult] = []
        self._station_store = StationStore(
            Path(__file__).resolve().parents[1] / "config" / "rain_sequence_stations.json"
        )
        self._fields: dict[str, QWidget] = {}
        self._build()

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(0)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle{background:#17384A;}")
        splitter.addWidget(self._build_settings())
        splitter.addWidget(self._build_results())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    def _build_settings(self) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(560)
        scroll.setMaximumWidth(760)
        scroll.setStyleSheet("QScrollArea{border:none;background:transparent;}")
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(6)

        basic = _group("基本场景")
        grid = QGridLayout(); grid.setSpacing(5); basic.layout().addLayout(grid)
        self._orbit = self._combo(("LEO", "GEO", "外部星历（固定仰角）"))
        self._leo_alt = self._edit("500")
        self._geo_lon = self._edit("110.5")
        self._fixed_el = self._edit("45")
        self._emin = self._edit("10")
        self._emax = self._edit("70")
        self._uplink = self._edit("50")
        self._downlink = self._edit("40")
        self._direction = self._combo(("上下行同时输出", "仅上行", "仅下行"))
        self._polarization = self._combo(("双线极化 H/V", "单极化 H", "单极化 V", "单极化 RHCP", "单极化 LHCP"))
        def row(index, left_label, left, right_label=None, right=None):
            grid.addWidget(QLabel(left_label), index, 0); grid.addWidget(left, index, 1)
            if right is not None: grid.addWidget(QLabel(right_label), index, 2); grid.addWidget(right, index, 3)
        row(0, "轨道场景:", self._orbit)
        row(1, "轨道高度/km", self._leo_alt, "最小仰角/°", self._emin)
        row(2, "上行/GHz", self._uplink, "下行/GHz", self._downlink)
        row(3, "极化复用", self._polarization, "链路方向", self._direction)
        row(4, "GEO经度/°", self._geo_lon, "固定仰角/°", self._fixed_el)
        layout.addWidget(basic)

        station = _group("信关站与雨层")
        grid = QGridLayout(); grid.setSpacing(5); station.layout().addLayout(grid)
        self._station = self._edit("上海站")
        self._station_picker = self._combo(self._station_store.names())
        self._station_picker.currentTextChanged.connect(self._apply_station)
        self._lat = self._edit("31.23")
        self._lon = self._edit("121.47")
        self._alt = self._edit("0.02")
        self._height_mode = self._combo(("ITU-R P.839-4自动计算", "手动设置"))
        self._rain_height = self._edit("5.0")
        self._rate = self._edit("35")
        self._probability = self._edit("0.32")
        self._phase = self._combo(HYDROMETEOR_PHASE_MODES)
        grid.addWidget(QLabel("信关站:"), 0, 0); grid.addWidget(self._station_picker, 0, 1, 1, 3)
        grid.addWidget(QLabel("站名:"), 1, 0); grid.addWidget(self._station, 1, 1)
        grid.addWidget(QLabel("纬度/经度/海拔"), 1, 2); grid.addWidget(QLabel("可保存自定义站址"), 1, 3)
        row = lambda r, a, b, c, d: (grid.addWidget(QLabel(a), r, 0), grid.addWidget(b, r, 1), grid.addWidget(QLabel(c), r, 2), grid.addWidget(d, r, 3))
        row(2, "纬度/°", self._lat, "经度/°", self._lon)
        row(3, "海拔/km", self._alt, "手动雨高/km", self._rain_height)
        grid.addWidget(QLabel("雨层高度:"), 4, 0); grid.addWidget(self._height_mode, 4, 1, 1, 3)
        layout.addWidget(station)
        station_buttons = QHBoxLayout()
        save_station = self._button("保存当前站址")
        save_station.clicked.connect(self._save_station)
        station_buttons.addWidget(save_station); station_buttons.addStretch()
        station.layout().addLayout(station_buttons)

        components = _group("传播影响")
        self._checks: dict[str, QCheckBox] = {}
        for key, text, checked in (
            ("enable_rain", "降雨", True), ("enable_gas", "气体吸收", True),
            ("enable_cloud", "云雾", True), ("enable_scintillation", "对流层闪烁", True),
            ("enable_ice_crystal", "冰晶", False), ("enable_snow", "降雪", False),
            ("enable_polarization", "双极化 Jones 矩阵", True),
        ):
            check = QCheckBox(text); check.setChecked(checked)
            check.setStyleSheet("font-size:10pt;color:#C9E6F2;")
            self._checks[key] = check
            components.layout().addWidget(check)
        self._terminal_isolation = self._edit("32")
        isolation_form = QFormLayout(); isolation_form.setSpacing(4)
        isolation_form.addRow("终端接口隔离度/dB:", self._terminal_isolation)
        components.layout().addLayout(isolation_form)
        layout.addWidget(components)

        phase = _group("水凝物相态与双极化")
        form = QFormLayout(); form.setSpacing(5); phase.layout().addLayout(form)
        form.addRow("相态模式:", self._phase)
        self._physical_fields = {}
        for key, text, default in (("melting_layer_thickness_km", "融化层垂直厚度/km", .72), ("melting_layer_enhancement", "融化层衰减增强", 1.35), ("frozen_ice_fraction", "冻结层冰晶份额", .45), ("ice_layer_thickness_km", "冰晶层厚度/km", 2.0), ("ice_specific_atten_db_km_40ghz", "冰晶40GHz比衰减", .003), ("snow_layer_thickness_km", "降雪层厚度/km", 1.0), ("snow_specific_atten_db_km_40ghz", "雪40GHz比衰减", .02)):
            field = self._edit(str(default)); self._physical_fields[key] = field; form.addRow(text + ":", field)
        layout.addWidget(phase)

        model = _group("模型参数")
        form = QFormLayout(); model.layout().addLayout(form)
        self._rain_model = self._combo(RAIN_MODELS); self._rain_model.currentTextChanged.connect(self._rebuild_model_parameters)
        self._fusion_mode = self._combo(("分层物理融合", "事件门控融合", "验证误差加权", "单模型"))
        self._rate = self._edit("35"); self._probability = self._edit("0.32")
        form.addRow("雨衰时序:", self._rain_model); form.addRow("融合方式:", self._fusion_mode); form.addRow("湿态典型雨强/(mm/h):", self._rate); form.addRow("降雨占时率(0~1):", self._probability)
        self._model_form = QGridLayout(); model.layout().addLayout(self._model_form); self._model_param_fields = {}; self._rebuild_model_parameters()
        layout.addWidget(model)

        buttons = QHBoxLayout(); buttons.setSpacing(6)
        run = self._button("生成序列", primary=True); run.clicked.connect(self._run)
        save = self._button("导出数据"); save.clicked.connect(self._export_data)
        buttons.addWidget(run); buttons.addWidget(save)
        layout.addLayout(buttons)
        buttons = QHBoxLayout(); buttons.setSpacing(6)
        save_plot = self._button("保存图像"); save_plot.clicked.connect(self._save_plot)
        export_cfg = self._button("导出配置"); export_cfg.clicked.connect(self._export_config)
        load_cfg = self._button("导入配置"); load_cfg.clicked.connect(self._load_config)
        buttons.addWidget(save_plot); buttons.addWidget(export_cfg); buttons.addWidget(load_cfg)
        layout.addLayout(buttons)
        layout.addStretch()
        scroll.setWidget(panel)
        return scroll

    def _build_results(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(6, 0, 0, 0)
        output_box = _group("生成与结果")
        output_grid = QGridLayout(); output_box.layout().addLayout(output_grid)
        self._output_checks = {}
        output_labels = ("上行总衰减", "下行雨衰分量", "融化层/冰雪/雪", "XPD/差分衰减/差分相位", "event_id/雨型标签", "下行总衰减", "气体/云/闪烁分量", "2×2极化信道 hHH/hHV/hVH/hVV", "星地距离/仰角变化")
        for index, (key, default) in enumerate(OutputConfig().selected.items()):
            checkbox = QCheckBox(output_labels[index]); checkbox.setChecked(default); checkbox.setStyleSheet("font-size:9pt;color:#C9E6F2;")
            self._output_checks[key] = checkbox; output_grid.addWidget(checkbox, index % 5, index // 5)
        self._duration_s = self._edit("440"); self._internal_ms = self._edit("100"); self._output_ms = self._edit("100"); self._sequence_count = self._edit("1"); self._seed = self._edit("20260821")
        controls = QGridLayout(); output_box.layout().addLayout(controls)
        for row, label, field in ((0, "持续时间/s", self._duration_s), (0, "内部步长/ms", self._internal_ms), (0, "序列条数", self._sequence_count), (1, "输出时间粒度/ms", self._output_ms), (1, "随机种子", self._seed)):
            col = 0 if label in ("持续时间/s", "输出时间粒度/ms") else 2 if label in ("内部步长/ms", "随机种子") else 4
            controls.addWidget(QLabel(label), row, col); controls.addWidget(field, row, col + 1)
        layout.addWidget(output_box)
        top = QHBoxLayout()
        title = QLabel("雨衰序列结果")
        title.setStyleSheet("font-size:11pt;color:#DFF8FF;font-weight:500;")
        self._sequence_selector = self._combo(("序列 1",))
        self._sequence_selector.currentIndexChanged.connect(self._select_sequence)
        self._view = self._combo(("时间序列", "上行总衰减", "下行总衰减", "降雨率", "雨衰增量", "XPD（上行）", "衰减分量", "自相关 ACF", "超越概率 CCDF", "轨道几何"))
        self._view.currentTextChanged.connect(lambda _: self._plot())
        top.addWidget(title); top.addStretch(); top.addWidget(QLabel("序列:")); top.addWidget(self._sequence_selector); top.addWidget(QLabel("显示:")); top.addWidget(self._view)
        layout.addLayout(top)
        series = QHBoxLayout(); series.setSpacing(9)
        self._series_checks: dict[str, QCheckBox] = {}
        for key, text, checked in (
            ("uplink_total_db", "上行总衰减(50 GHz)", True),
            ("downlink_total_db", "下行总衰减(40 GHz)", True),
            ("rain_uplink_db", "雨衰", True),
            ("cloud_uplink_db", "云/气体", True),
            ("ice_snow_uplink_db", "融化层/冰雪", True),
            ("xpd_db", "XPD", False),
        ):
            check = QCheckBox(text); check.setChecked(checked); check.setStyleSheet("font-size:9pt;color:#C9E6F2;")
            check.toggled.connect(self._plot)
            self._series_checks[key] = check; series.addWidget(check)
        series.addStretch(); layout.addLayout(series)
        self._figure = Figure(figsize=(7.2, 5.2), dpi=96)
        self._figure.patch.set_facecolor("#07131F")
        self._canvas = FigureCanvas(self._figure)
        layout.addWidget(self._canvas, stretch=1)
        layout.addWidget(NavigationToolbar(self._canvas, panel))
        self._metrics = QLabel()
        self._metrics.setStyleSheet("font-size:9pt;color:#C9E6F2;background:#091B29;border:1px solid #1C4961;border-radius:4px;padding:5px;")
        self._metrics.setWordWrap(True)
        layout.addWidget(self._metrics)
        self._status = QLabel("就绪 — 设置参数后点击“生成序列”。")
        self._status.setStyleSheet("font-size:9pt;color:#648A9C;")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)
        return panel

    @staticmethod
    def _edit(text: str) -> QLineEdit:
        field = QLineEdit(text); field.setStyleSheet(_EDIT_STYLE); return field

    @staticmethod
    def _combo(items) -> QComboBox:
        combo = QComboBox(); combo.setStyleSheet(_COMBO_STYLE); combo.addItems(items); return combo

    @staticmethod
    def _button(text: str, primary: bool = False) -> QPushButton:
        button = QPushButton(text); button.setFixedHeight(29)
        if primary:
            button.setStyleSheet("QPushButton{background:#1D9E75;color:#FFF;border:none;border-radius:5px;font-size:10pt;font-weight:bold;}QPushButton:hover{background:#14705A;}")
        else:
            button.setStyleSheet("QPushButton{background:#091B29;color:#C9E6F2;border:1px solid #CCC;border-radius:5px;font-size:9pt;padding:0 7px;}QPushButton:hover{background:#0D3042;}")
        return button

    def _number(self, field: QLineEdit, label: str, integer: bool = False):
        try:
            return int(field.text()) if integer else float(field.text())
        except ValueError as exc:
            raise ValueError(f"{label} 不是有效数值。") from exc

    def _rebuild_model_parameters(self, *_):
        previous = {key: field.text() for key, field in getattr(self, "_model_param_fields", {}).items()}
        while self._model_form.count():
            item = self._model_form.takeAt(0)
            if item.widget() is not None: item.widget().deleteLater()
        self._model_param_fields = {}
        for index, (key, label, default) in enumerate(MODEL_PARAM_DEFS[self._rain_model.currentText()]):
            field = self._edit(previous.get(key, str(default)))
            self._model_param_fields[key] = field
            row, col = divmod(index, 2)
            self._model_form.addWidget(QLabel(label + ":"), row, col * 2)
            self._model_form.addWidget(field, row, col * 2 + 1)

    def _apply_station(self, name: str):
        record = self._station_store.get(name)
        if record is None:
            return
        self._station.setText(name)
        self._lat.setText(str(record["lat_deg"]))
        self._lon.setText(str(record["lon_deg"]))
        self._alt.setText(str(record["alt_km"]))

    def _save_station(self):
        try:
            name = self._station.text().strip()
            self._station_store.save(name, self._number(self._lat, "纬度"), self._number(self._lon, "经度"), self._number(self._alt, "海拔"))
            self._station_picker.blockSignals(True)
            self._station_picker.clear(); self._station_picker.addItems(self._station_store.names())
            self._station_picker.setCurrentText(name)
            self._station_picker.blockSignals(False)
            QMessageBox.information(self, "站址保存", f"已保存站址：{name}")
        except Exception as exc:
            QMessageBox.critical(self, "站址保存失败", str(exc))

    def _make_configs(self) -> tuple[SimulationConfig, ModelConfig, OutputConfig]:
        sim = SimulationConfig(
            orbit=self._orbit.currentText(), leo_altitude_km=self._number(self._leo_alt, "LEO高度"),
            station_name=self._station.text().strip() or "未命名站", station_lat_deg=self._number(self._lat, "纬度"),
            station_lon_deg=self._number(self._lon, "经度"), station_alt_km=self._number(self._alt, "海拔"),
            rain_height_mode=self._height_mode.currentText(), rain_height_km=self._number(self._rain_height, "雨高"),
            direction=self._direction.currentText(), uplink_ghz=self._number(self._uplink, "上行频率"),
            downlink_ghz=self._number(self._downlink, "下行频率"), polarization=self._polarization.currentText(),
            duration_s=self._number(self._duration_s, "时长"), internal_dt_ms=self._number(self._internal_ms, "内部步长"),
            sequence_count=self._number(self._sequence_count, "序列数量", True), elevation_min_deg=self._number(self._emin, "最小仰角"),
            elevation_max_deg=self._number(self._emax, "最大仰角"), geo_longitude_deg=self._number(self._geo_lon, "GEO经度"),
            rain_rate_mm_h=self._number(self._rate, "雨强"), rain_probability=self._number(self._probability, "降水概率"),
            seed=self._number(self._seed, "随机种子", True), hydrometeor_phase_mode=self._phase.currentText(),
            **{key: check.isChecked() for key, check in self._checks.items()},
        )
        for key, field in self._physical_fields.items():
            setattr(sim, key, self._number(field, key))
        if sim.orbit == "外部星历（固定仰角）":
            sim.elevation_max_deg = self._number(self._fixed_el, "固定仰角")
            sim.elevation_min_deg = max(1.0, sim.elevation_max_deg - 0.01)
        model_parameters = {key: self._number(field, key) for key, field in self._model_param_fields.items()}
        model = ModelConfig(
            rain_model=self._rain_model.currentText(), fusion_mode=self._fusion_mode.currentText(),
            fast_tau_s=float(model_parameters.get("fast_tau_s", 2.5)), slow_tau_s=float(model_parameters.get("slow_tau_s", 75.0)),
            event_mean_s=float(model_parameters.get("event_mean_s", 90.0)), event_gap_s=float(model_parameters.get("event_gap_s", 180.0)),
            wind_speed_m_s=float(model_parameters.get("sst_wind_speed_m_s", model_parameters.get("excell_wind_speed_m_s", 12.0))),
            cell_count=max(1, int(round(model_parameters.get("excell_cell_count", 5)))),
            marginal_shape=float(model_parameters.get("onera_slow_shape", model_parameters.get("gamma_shape", 2.2))),
            fusion_weight_dynamic=float(model_parameters.get("fusion_dynamic_weight", .65)),
            fusion_weight_event=float(model_parameters.get("fusion_event_weight", .35)),
            terminal_isolation_db=self._number(self._terminal_isolation, "终端接口隔离度"),
            parameters={key: float(value) for key, value in model_parameters.items()},
        )
        # ``ModelConfig`` 的固定字段（如 cloud_tau_s、terminal_isolation_db）
        # 和各模型的 ``parameters`` 字典均可由高级参数覆盖。
        for key, value in model_parameters.items():
            if key != "parameters" and hasattr(model, key):
                setattr(model, key, float(value))
        output = OutputConfig(output_dt_ms=self._number(self._output_ms, "输出步长"), selected={key: check.isChecked() for key, check in self._output_checks.items()})
        sim.validate(output.output_dt_ms)
        return sim, model, output

    def _run(self):
        try:
            self.setCursor(Qt.CursorShape.WaitCursor)
            sim, model, output = self._make_configs()
            self._results = [simulate_channel(sim, model, output, sequence_index=index) for index in range(sim.sequence_count)]
            self._result = self._results[0]
            self._sequence_selector.blockSignals(True)
            self._sequence_selector.clear()
            self._sequence_selector.addItems([f"序列 {index}" for index in range(1, len(self._results) + 1)])
            self._sequence_selector.blockSignals(False)
            self._plot()
            metrics = self._result.metrics
            self._status.setText(
                f"完成：{len(self._results)} 条序列，每条 {len(self._result.time_s):,} 点；"
                f"上行 P99 {metrics['uplink_p99_db']:.2f} dB，最大值 {metrics['uplink_max_db']:.2f} dB。"
            )
            self._metrics.setText(
                f"统计：上行最大 {metrics['uplink_max_db']:.2f} dB ｜ 下行最大 {metrics['downlink_max_db']:.2f} dB ｜ "
                f"动态范围 {metrics['dynamic_range_db']:.2f} dB ｜ t95 {metrics['t95_s']:.3g} s ｜ "
                f"t99 {metrics['t99_s']:.3g} s ｜ 雨时占比 {metrics['rain_time_fraction']:.1%} ｜ "
                f"湿态雨强中位数 {metrics['rain_rate_wet_median_mm_h']:.2f} mm/h"
            )
        except Exception as exc:
            QMessageBox.critical(self, "雨衰序列生成", str(exc))
        finally:
            self.unsetCursor()

    def _select_sequence(self, index: int):
        if 0 <= index < len(self._results):
            self._result = self._results[index]
            self._plot()

    def _plot(self):
        if self._result is None:
            return
        result, data, view = self._result, self._result.data, self._view.currentText()
        self._figure.clear(); axis = self._figure.add_subplot(111)
        axis.set_facecolor("#091B29")
        if view == "时间序列":
            index = self._indices(result.time_s.size)
            series = (
                ("uplink_total_db", "上行总衰减 50 GHz", "#12628B", "-"),
                ("downlink_total_db", "下行总衰减 40 GHz", "#C25D21", "--"),
                ("rain_uplink_db", "上行雨衰", "#2D8065", "-"),
                ("cloud_uplink_db", "上行云/气体", "#7555A8", "-"),
                ("ice_snow_uplink_db", "上行融化层/冰雪", "#7097A8", "-"),
                ("xpd_db", "XPD", "#9A7B19", "-"),
            )
            for key, label, color, linestyle in series:
                if not self._series_checks[key].isChecked():
                    continue
                values = data[key] if key != "cloud_uplink_db" else data["cloud_uplink_db"] + data["gas_uplink_db"]
                values = values[index]
                finite = np.isfinite(values)
                axis.plot(result.time_s[index][finite], values[finite], label=label, color=color, lw=1.35, ls=linestyle)
            axis.set(xlabel="时间 / s", ylabel="衰减或XPD / dB", title=f"{self._orbit.currentText()} · {self._rain_model.currentText()} · 时间序列")
            if axis.lines: axis.legend(loc="best", fontsize=8, ncol=2)
        elif view == "自相关 ACF":
            lag, values = acf_curve(data["uplink_total_db"], float(self._output_ms.text()) / 1000.0)
            axis.plot(lag, values, color="#1D9E75", lw=1.5)
            axis.set(xlabel="时延 (s)", ylabel="归一化自相关", title="上行总衰减自相关函数")
        elif view == "超越概率 CCDF":
            values, exceedance = ccdf_curve(data["uplink_total_db"])
            axis.semilogy(values, np.maximum(exceedance, 1e-8), color="#1D9E75", lw=1.5)
            axis.set(xlabel="上行总衰减 (dB)", ylabel="超越概率", title="上行总衰减 CCDF")
        elif view == "衰减分量":
            index = self._indices(result.time_s.size)
            for key, text, color in (("rain_uplink_db", "降雨", "#0055CC"), ("gas_uplink_db", "气体", "#BA7517"),
                                     ("cloud_uplink_db", "云雾", "#7B3FA0"), ("scint_uplink_db", "闪烁", "#CC2200"),
                                     ("ice_uplink_db", "冰晶", "#00838F"), ("snow_uplink_db", "降雪", "#37474F")):
                axis.plot(result.time_s[index], data[key][index], label=text, color=color, lw=1.1)
            axis.legend(fontsize=8, ncol=2)
            axis.set(xlabel="时间 (s)", ylabel="衰减 (dB)", title="上行衰减分量")
        elif view == "雨衰增量":
            index = self._indices(result.time_s.size)
            increments = np.diff(data["rain_uplink_db"], prepend=data["rain_uplink_db"][0])
            axis.plot(result.time_s[index], increments[index], color="#8F5B3A", lw=1.1)
            axis.axhline(0, color="#999999", ls=":", lw=.8)
            axis.set(xlabel="时间 (s)", ylabel="相邻样本雨衰增量 (dB)", title="上行雨衰快变增量")
        elif view == "轨道几何":
            index = self._indices(result.time_s.size)
            axis.plot(result.time_s[index], data["elevation_deg"][index], color="#1D9E75", lw=1.3, label="仰角")
            distance_axis = axis.twinx()
            distance_axis.plot(result.time_s[index], data["slant_range_km"][index], color="#12628B", lw=1.2, label="星地距离")
            axis.set(xlabel="时间 (s)", ylabel="仰角 (°)", title="星地几何变化")
            distance_axis.set_ylabel("星地距离 (km)")
            axis.legend(loc="best", fontsize=8); distance_axis.legend(loc="upper right", fontsize=8)
        else:
            key, ylabel, title = {
                "上行总衰减": ("uplink_total_db", "衰减 (dB)", "上行总衰减时序"),
                "下行总衰减": ("downlink_total_db", "衰减 (dB)", "下行总衰减时序"),
                "降雨率": ("rain_rate_mm_h", "雨强 (mm/h)", "降雨率时序"),
                "XPD（上行）": ("xpd_uplink_db", "XPD (dB)", "上行交叉极化鉴别度"),
            }[view]
            index = self._indices(result.time_s.size)
            values = data[key][index]
            finite = np.isfinite(values)
            axis.plot(result.time_s[index][finite], values[finite], color="#1D9E75", lw=1.2)
            axis.set(xlabel="时间 (s)", ylabel=ylabel, title=title)
        axis.grid(True, color="#E2E2E2", lw=.6)
        for spine in axis.spines.values(): spine.set_color("#1C4961")
        axis.tick_params(labelsize=9)
        self._figure.tight_layout(); self._canvas.draw()

    @staticmethod
    def _indices(size: int) -> np.ndarray:
        return np.arange(size) if size <= 12000 else np.linspace(0, size - 1, 12000, dtype=int)

    def _export_data(self):
        if self._result is None:
            QMessageBox.information(self, "雨衰序列生成", "请先生成序列。"); return
        path, selected = QFileDialog.getSaveFileName(self, "导出雨衰序列", "rain_sequence.mat", "MAT 文件 (*.mat);;HDF5 文件 (*.h5);;CSV 文件 (*.csv)")
        if not path: return
        fmt = "HDF5" if "h5" in selected.lower() else "CSV" if "csv" in selected.lower() else "MAT"
        suffix = {"MAT": ".mat", "HDF5": ".h5", "CSV": ".csv"}[fmt]
        if not path.lower().endswith(suffix): path += suffix
        try:
            selected_groups = {key: check.isChecked() for key, check in self._output_checks.items()}
            for index, result in enumerate(self._results or [self._result], 1):
                target = Path(path)
                if len(self._results) > 1:
                    target = target.with_name(f"{target.stem}_{index:03d}{target.suffix}")
                export_result(target, fmt, result, selected_groups)
            message = f"已导出 {len(self._results or [self._result])} 条序列：\n{path}"
            QMessageBox.information(self, "导出成功", message)
        except Exception as exc:
            QMessageBox.critical(self, "导出失败", str(exc))

    def _save_plot(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存图像", "rain_sequence.png", "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)")
        if path:
            self._figure.savefig(path, dpi=150, bbox_inches="tight")

    def _export_config(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出雨衰配置", "rain_sequence_config.json", "JSON (*.json)")
        if path:
            if not path.lower().endswith(".json"): path += ".json"
            sim, model, output = self._make_configs()
            Path(path).write_text(json.dumps({"simulation": asdict(sim), "model": asdict(model), "output": asdict(output)}, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "导入雨衰配置", "", "JSON (*.json)")
        if not path: return
        try:
            payload = json.loads(Path(path).read_text(encoding="utf-8"))
            values = payload.get("simulation", payload)
            model = payload.get("model", {})
            output = payload.get("output", {})
            mapping = {
                "orbit": self._orbit, "duration_s": self._duration_s, "internal_dt_ms": self._internal_ms, "output_dt_ms": self._output_ms,
                "sequence_count": self._sequence_count,
                "seed": self._seed, "leo_altitude_km": self._leo_alt, "geo_longitude_deg": self._geo_lon, "fixed_elevation_deg": self._fixed_el,
                "elevation_min_deg": self._emin, "elevation_max_deg": self._emax, "station_name": self._station, "station_lat_deg": self._lat,
                "station_lon_deg": self._lon, "station_alt_km": self._alt, "rain_height_mode": self._height_mode, "rain_height_km": self._rain_height,
                "uplink_ghz": self._uplink, "downlink_ghz": self._downlink, "polarization": self._polarization, "hydrometeor_phase_mode": self._phase,
                "rain_rate_mm_h": self._rate, "rain_probability": self._probability,
            }
            for key, widget in mapping.items():
                if key not in values: continue
                if isinstance(widget, QComboBox): widget.setCurrentText(str(values[key]))
                else: widget.setText(str(values[key]))
            for key, check in self._checks.items():
                if key in values: check.setChecked(bool(values[key]))
            self._rain_model.setCurrentText(str(model.get("rain_model", self._rain_model.currentText())))
            self._fusion_mode.setCurrentText(str(model.get("fusion_mode", self._fusion_mode.currentText())))
            if "terminal_isolation_db" in model: self._terminal_isolation.setText(str(model["terminal_isolation_db"]))
            self._rebuild_model_parameters()
            model_parameters = dict(model.get("parameters", {}))
            model_parameters.update({key: value for key, value in model.items() if key not in {"rain_model", "fusion_mode", "parameters"}})
            for key, field in self._model_param_fields.items():
                if key in model_parameters: field.setText(str(model_parameters[key]))
            for key, field in self._physical_fields.items():
                if key in values: field.setText(str(values[key]))
            for key, check in self._output_checks.items():
                if key in output.get("selected", {}): check.setChecked(bool(output["selected"][key]))
            if "output_dt_ms" in output: self._output_ms.setText(str(output["output_dt_ms"]))
        except (OSError, ValueError, TypeError) as exc:
            QMessageBox.critical(self, "导入配置失败", str(exc))


class RainSequenceDialog(ModuleDialog):
    """主界面的独立雨衰序列生成功能页。"""

    TITLE = "雨衰序列生成"
    ACCENT_COLOR = "#22C7E6"
    MIN_WIDTH = 1450
    MIN_HEIGHT = 820

    def build_content(self, layout: QVBoxLayout):
        layout.setContentsMargins(10, 8, 10, 10)
        layout.addWidget(RainSequenceWidget(self), stretch=1)
