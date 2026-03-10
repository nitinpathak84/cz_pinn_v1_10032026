from sympy import Eq, And, Or

from physicsnemo.sym.domain.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from physicsnemo.sym.domain.inferencer import PointwiseInferencer

from cz.postprocess import make_regular_grid


def add_boundary_constraints(domain, nodes, geo, cfg):
    x = geo.x
    y = geo.y
    bs = cfg.batch_size
    bc = cfg.custom.boundary

    theta_seed = float(bc.theta_seed)
    theta_hot = float(bc.theta_hot)
    theta_melt = float(bc.theta_melt)

    # ------------------------------------------------------------------
    # Axis symmetry: dT/dr = 0 on x = 0 for all regions touching axis
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Crystal top: seed/puller side fixed temperature
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Outer heater on insulation bottom
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Outer insulation right wall: adiabatic
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Outer insulation top wall: adiabatic
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Fixed solid-liquid interface temperature (reduced V1)
    # Crystal side of interface
    # ------------------------------------------------------------------
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

    # Melt side of interface: only under the crystal radius
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

    # ------------------------------------------------------------------
    # Melt free surface outside crystal footprint: adiabatic in V1
    # ------------------------------------------------------------------
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

    # ---------------------------------------------------------------
    # Melt <-> Crucible interface
    # On melt geometry boundary:
    #   y = 0 (bottom)
    #   x = r_melt (side wall)
    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    # Crucible <-> Insulation interface
    # On crucible geometry outer boundaries:
    #   x = r_crucible_outer
    #   y = -t_bottom
    #   y = h_melt, for x >= r_melt
    # ---------------------------------------------------------------
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

    grid_invar = make_regular_grid(
        xmin=0.0,
        xmax=geo.r_outer,
        ymin=-geo.t_crucible_bottom,
        ymax=geo.h_outer,
        nx=int(inf.nx),
        ny=int(inf.ny),
    )

    inferencer = PointwiseInferencer(
        nodes=nodes,
        invar=grid_invar,
        output_names=["theta_cr", "theta_m", "theta_cu", "theta_ins"],
        batch_size=int(inf.batch_size),
    )
    domain.add_inferencer(inferencer, "global_grid")


def add_monitors(domain, nodes, geo, cfg):
    # Keep V1 simple.
    # Add PointwiseMonitor objects later once basic training is stable.
    return