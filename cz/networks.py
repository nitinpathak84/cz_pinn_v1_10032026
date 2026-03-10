from sympy import Symbol, Number

from physicsnemo.sym.hydra import instantiate_arch
from physicsnemo.sym.key import Key
from physicsnemo.sym.eq.pdes.basic import GradNormal
from physicsnemo.sym.eq.pdes.diffusion import DiffusionInterface
from physicsnemo.sym.node import Node

from cz.pdes.axisymmetric_diffusion import AxisymmetricDiffusion


def build_cz_model(cfg):
    aspect_sq = float(cfg.custom.nondim.aspect_sq)
    eps_r = float(cfg.custom.numerics.eps_r)

    T_seed_K = float(cfg.custom.dimensional.T_seed_K)
    T_hot_K = float(cfg.custom.dimensional.T_hot_K)
    dT = T_hot_K - T_seed_K

    # -------------------------------------------------------
    # PDE nodes for training
    # -------------------------------------------------------
    eq_cr = AxisymmetricDiffusion(T="theta_cr", aspect_sq=aspect_sq, eps_r=eps_r)
    eq_m = AxisymmetricDiffusion(T="theta_m", aspect_sq=aspect_sq, eps_r=eps_r)
    eq_cu = AxisymmetricDiffusion(T="theta_cu", aspect_sq=aspect_sq, eps_r=eps_r)
    eq_ins = AxisymmetricDiffusion(T="theta_ins", aspect_sq=aspect_sq, eps_r=eps_r)

    gn_cr = GradNormal(T="theta_cr", dim=2, time=False)
    gn_m = GradNormal(T="theta_m", dim=2, time=False)
    gn_cu = GradNormal(T="theta_cu", dim=2, time=False)
    gn_ins = GradNormal(T="theta_ins", dim=2, time=False)

    if_mc = DiffusionInterface(
        "theta_m",
        "theta_cu",
        float(cfg.custom.physics.k_m),
        float(cfg.custom.physics.k_cu),
        dim=2,
        time=False,
    )

    if_ci = DiffusionInterface(
        "theta_cu",
        "theta_ins",
        float(cfg.custom.physics.k_cu),
        float(cfg.custom.physics.k_ins),
        dim=2,
        time=False,
    )

    # -------------------------------------------------------
    # Shared trainable networks
    # -------------------------------------------------------
    input_keys = [Key("x"), Key("y")]

    net_cr = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("theta_cr")],
        cfg=cfg.arch.fully_connected,
    )

    net_m = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("theta_m")],
        cfg=cfg.arch.fully_connected,
    )

    net_cu = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("theta_cu")],
        cfg=cfg.arch.fully_connected,
    )

    net_ins = instantiate_arch(
        input_keys=input_keys,
        output_keys=[Key("theta_ins")],
        cfg=cfg.arch.fully_connected,
    )

    # -------------------------------------------------------
    # Training node list
    # -------------------------------------------------------
    train_nodes = (
        eq_cr.make_nodes()
        + eq_m.make_nodes()
        + eq_cu.make_nodes()
        + eq_ins.make_nodes()
        + gn_cr.make_nodes()
        + gn_m.make_nodes()
        + gn_cu.make_nodes()
        + gn_ins.make_nodes()
        + if_mc.make_nodes()
        + if_ci.make_nodes()
        + [net_cr.make_node(name="theta_cr_net")]
        + [net_m.make_node(name="theta_m_net")]
        + [net_cu.make_node(name="theta_cu_net")]
        + [net_ins.make_node(name="theta_ins_net")]
    )

    # -------------------------------------------------------
    # Region-specific inferencer node lists
    # Each inferencer writes:
    #   temperature_K
    #   region_id
    # so ParaView can append them into one common plot
    # -------------------------------------------------------
    theta_cr = Symbol("theta_cr")
    theta_m = Symbol("theta_m")
    theta_cu = Symbol("theta_cu")
    theta_ins = Symbol("theta_ins")

    inferencer_nodes = {
        "crystal": [
            net_cr.make_node(name="theta_cr_net_infer"),
            Node.from_sympy(T_seed_K + dT * theta_cr, "temperature_K"),
            Node.from_sympy(theta_cr, "theta"),
            Node.from_sympy(0.0 * theta_cr + Number(1.0), "region_id"),
        ],
        "melt": [
            net_m.make_node(name="theta_m_net_infer"),
            Node.from_sympy(T_seed_K + dT * theta_m, "temperature_K"),
            Node.from_sympy(theta_m, "theta"),
            Node.from_sympy(0.0 * theta_m + Number(2.0), "region_id"),
        ],
        "crucible": [
            net_cu.make_node(name="theta_cu_net_infer"),
            Node.from_sympy(T_seed_K + dT * theta_cu, "temperature_K"),
            Node.from_sympy(theta_cu, "theta"),
            Node.from_sympy(0.0 * theta_cu + Number(3.0), "region_id"),
        ],
        "insulation": [
            net_ins.make_node(name="theta_ins_net_infer"),
            Node.from_sympy(T_seed_K + dT * theta_ins, "temperature_K"),
            Node.from_sympy(theta_ins, "theta"),
            Node.from_sympy(0.0 * theta_ins + Number(4.0), "region_id"),
        ],
    }

    return {
        "train_nodes": train_nodes,
        "inferencer_nodes": inferencer_nodes,
    }
