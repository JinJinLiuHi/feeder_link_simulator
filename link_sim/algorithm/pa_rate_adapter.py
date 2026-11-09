"""Compatibility entry point for direct discrete PA/DPD evaluation."""

from .pa_simulation_core import simulate


def simulate_at_rate(model, x, sample_rate):
    """Use saved sample delays directly; the recorded model rate is metadata."""
    return simulate(model, x, sample_rate)
