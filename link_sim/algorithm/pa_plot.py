"""Shared spectrum presentation for PA characterization and simulation."""


def draw_pa_spectrum(figure, frequency_mhz, curves, title,
                     ylabel="功率 (dBm)", limits=(-80, 30), band_mhz=None):
    """curves contains (values, legend, color); frequencies share one axis."""
    dark = "#091B29"
    figure.clear()
    figure.set_layout_engine("constrained")
    figure.patch.set_facecolor(dark)
    ax = figure.subplots(1, 1)
    ax.set_facecolor(dark)
    if band_mhz is not None:
        ax.axvspan(-band_mhz, band_mhz, color="#C9A227", alpha=0.22)
        for bound in (-band_mhz, band_mhz):
            ax.axvline(bound, color="#E0C040", ls="--", lw=1)
    for values, label, color in curves:
        ax.plot(frequency_mhz, values, color=color, lw=1, label=label)
    ax.set_xlim(frequency_mhz[0], frequency_mhz[-1])
    if limits is not None:
        ax.set_ylim(*limits)
    ax.set_xlabel("频率 (MHz)", color="#D9ECF7", fontsize=10)
    ax.set_ylabel(ylabel, color="#D9ECF7", fontsize=10)
    ax.set_title(title, color="#EEEEEE", fontsize=10)
    ax.tick_params(colors="#BBBBBB", labelsize=9)
    ax.grid(True, color="#3C3C3C", lw=0.5)
    for spine in ax.spines.values():
        spine.set_color("#555555")
    if curves:
        legend = ax.legend(loc="upper left", facecolor=dark, edgecolor="#555555", fontsize=9)
        for text in legend.get_texts():
            text.set_color("#D9ECF7")
    return ax
