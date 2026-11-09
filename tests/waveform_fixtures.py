"""Explicit loaded RRC models for integration tests of configured links."""


def rrc_model(sps=4, beta=.25, span=8):
    from link_sim.algorithm.device_models import DeviceModel, FILTER_DEFAULTS
    from link_sim.algorithm.burst_link.pulse_shaping import RRC
    taps = RRC(samples_per_symbol=sps, rolloff=beta, span_symbols=span).coefficients().numpy()
    return DeviceModel("filter", dict(FILTER_DEFAULTS, impl="RRC 成形", fs=700e6*sps,
                                      sps=sps, beta=beta, span=span), {"h": taps}, "测试 RRC").validated()


def configured_link(dual=False):
    from link_sim.algorithm.waveform_chain import default_link
    stages, edges, positions = default_link(dual)
    for stage in stages:
        if stage.kind == "frame_source":
            stage.settings["seed"] = 2026+stage.settings["lane"]
        if stage.kind == "rrc_tx":
            stage.model = rrc_model()
    return stages, edges, positions


def configure_workspace_rrc(workspace):
    from link_sim.algorithm.waveform_chain import model_to_dict
    for node in workspace.graph.all_nodes():
        if node.KIND == "frame_source":
            node.set_property("settings", dict(node.get_property("settings"), seed=2026), push_undo=False)
        if node.KIND == "rrc_tx":
            node.set_property("device_model", model_to_dict(rrc_model()), push_undo=False)
