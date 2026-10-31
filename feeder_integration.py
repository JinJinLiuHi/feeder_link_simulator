"""Integration boundary for the independently maintained link domains."""

from __future__ import annotations

def create_calculation_pages(parent=None):
    from link_budget.ui.workspace import calculation_pages
    return calculation_pages(parent)


def create_simulation_pages(parent=None):
    from link_sim.ui.workspace import simulation_pages
    return simulation_pages(parent)
