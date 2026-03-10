from sympy import Eq, And, Or

from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.inferencer import PointwiseInferencer


def add_boundary_constraints(domain, nodes, geo, cfg):
    x = geo.x
    y = geo.y
    bs = cfg.batch_size
    bc = cfg.custom.boundary

    theta_seed = float(bc.theta_seed)
    theta_hot = float(bc.theta_hot)
    theta_melt = float(bc.theta_melt)

    # Axis symmetry
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.crystal,
            outvar={"normal_gradient_theta_cr": 0.0},
            batch_size=bs.axis_cr,
            criteria=Eq(x, 0.0),
        ),
        "axis_crystal",
    )

    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.melt,
            outvar={"normal_gradient_theta_m": 0.0},
            batch_size=bs.axis_m,
            criteria=Eq(x, 0.0),
        ),
        "axis_melt",
    )

    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.crucible,
            outvar={"normal_gradient_theta_cu": 0.0},
            batch_size=bs.axis_cu,
            criteria=Eq(x, 0.0),
        ),
        "axis_crucible",
    )

    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.insulation,
            outvar={"normal_gradient_theta_ins": 0.0},
            batch_size=bs.axis_ins,
            criteria=Eq(x, 0.0),
        ),
        "axis_insulation",
    )

    # Crystal top
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.crystal,
            outvar={"theta_cr": theta_seed},
            batch_size=bs.crystal_top,
            criteria=Eq(y, geo.y_crystal_top),
        ),
        "crystal_top",
    )

    # Heater bottom on insulation
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.insulation,
            outvar={"theta_ins": theta_hot},
            batch_size=bs.heater_bottom,
            criteria=Eq(y, -geo.t_crucible_bottom),
        ),
        "heater_bottom",
    )

    # Outer right wall adiabatic
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.insulation,
            outvar={"normal_gradient_theta_ins": 0.0},
            batch_size=bs.outer_right,
            criteria=Eq(x, geo.r_outer),
        ),
        "outer_right_adiabatic",
    )

    # Outer top wall adiabatic
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.insulation,
            outvar={"normal_gradient_theta_ins": 0.0},
            batch_size=bs.outer_top,
            criteria=Eq(y, geo.h_outer),
        ),
        "outer_top_adiabatic",
    )

    # Solid-liquid interface: crystal side
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.crystal,
            outvar={"theta_cr": theta_melt},
            batch_size=bs.sl_cr,
            criteria=And(Eq(y, geo.y_sl), x <= geo.r_crystal),
        ),
        "sl_interface_crystal_side",
    )

    # Solid-liquid interface: melt side
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.melt,
            outvar={"theta_m": theta_melt},
            batch_size=bs.sl_m,
            criteria=And(Eq(y, geo.y_sl), x <= geo.r_crystal),
        ),
        "sl_interface_melt_side",
    )

    # Melt free surface outside crystal footprint
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.melt,
            outvar={"normal_gradient_theta_m": 0.0},
            batch_size=bs.melt_free_surface,
            criteria=And(Eq(y, geo.y_sl), x > geo.r_crystal),
        ),
        "melt_free_surface",
    )


def add_interior_constraints(domain, nodes, geo, cfg):
    bs = cfg.batch_size

    domain.add_constraint(
        PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=geo.crystal,
            outvar={"axisym_diffusion_theta_cr": 0.0},
            batch_size=bs.interior_cr,
        ),
        "interior_crystal",
    )

    domain.add_constraint(
        PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=geo.melt,
            outvar={"axisym_diffusion_theta_m": 0.0},
            batch_size=bs.interior_m,
        ),
        "interior_melt",
    )

    domain.add_constraint(
        PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=geo.crucible,
            outvar={"axisym_diffusion_theta_cu": 0.0},
            batch_size=bs.interior_cu,
        ),
        "interior_crucible",
    )

    domain.add_constraint(
        PointwiseInteriorConstraint(
            nodes=nodes,
            geometry=geo.insulation,
            outvar={"axisym_diffusion_theta_ins": 0.0},
            batch_size=bs.interior_ins,
        ),
        "interior_insulation",
    )


def add_interface_constraints(domain, nodes, geo, cfg):
    x = geo.x
    y = geo.y
    bs = cfg.batch_size

    # Melt <-> Crucible
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.melt,
            outvar={
                "diffusion_interface_dirichlet_theta_m_theta_cu": 0.0,
                "diffusion_interface_neumann_theta_m_theta_cu": 0.0,
            },
            batch_size=bs.interface_mc,
            criteria=Or(Eq(y, 0.0), Eq(x, geo.r_melt)),
        ),
        "interface_melt_crucible",
    )

    # Crucible <-> Insulation
    domain.add_constraint(
        PointwiseBoundaryConstraint(
            nodes=nodes,
            geometry=geo.crucible,
            outvar={
                "diffusion_interface_dirichlet_theta_cu_theta_ins": 0.0,
                "diffusion_interface_neumann_theta_cu_theta_ins": 0.0,
            },
            batch_size=bs.interface_ci,
            criteria=Or(
                Eq(x, geo.r_crucible_outer),
                Eq(y, -geo.t_crucible_bottom),
                And(Eq(y, geo.h_melt), x >= geo.r_melt),
            ),
        ),
        "interface_crucible_insulation",
    )


def add_inferencers(domain, nodes, geo, cfg):
    inf = cfg.custom.inference

    n_cr = int(inf.n_crystal)
    n_m = int(inf.n_melt)
    n_cu = int(inf.n_crucible)
    n_ins = int(inf.n_insulation)
    bs = int(inf.batch_size)

    invar_cr = geo.crystal.sample_interior(nr_points=n_cr)
    invar_m = geo.melt.sample_interior(nr_points=n_m)
    invar_cu = geo.crucible.sample_interior(nr_points=n_cu)
    invar_ins = geo.insulation.sample_interior(nr_points=n_ins)

    domain.add_inferencer(
        PointwiseInferencer(
            nodes=nodes,
            invar=invar_cr,
            output_names=["theta_cr"],
            batch_size=bs,
        ),
        "crystal",
    )

    domain.add_inferencer(
        PointwiseInferencer(
            nodes=nodes,
            invar=invar_m,
            output_names=["theta_m"],
            batch_size=bs,
        ),
        "melt",
    )

    domain.add_inferencer(
        PointwiseInferencer(
            nodes=nodes,
            invar=invar_cu,
            output_names=["theta_cu"],
            batch_size=bs,
        ),
        "crucible",
    )

    domain.add_inferencer(
        PointwiseInferencer(
            nodes=nodes,
            invar=invar_ins,
            output_names=["theta_ins"],
            batch_size=bs,
        ),
        "insulation",
    )


def add_monitors(domain, nodes, geo, cfg):
    return
