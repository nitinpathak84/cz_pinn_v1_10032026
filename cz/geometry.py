from dataclasses import dataclass

from sympy import Symbol
from physicsnemo.sym.geometry.primitives_2d import Rectangle


@dataclass
class CzGeometry:
    x: object
    y: object

    crystal: object
    melt: object
    crucible: object
    insulation: object
    outer_box: object

    r_crystal: float
    h_crystal: float

    r_melt: float
    h_melt: float

    t_crucible_wall: float
    t_crucible_bottom: float

    r_crucible_outer: float
    r_outer: float
    h_outer: float

    y_sl: float
    y_crystal_top: float


def build_cz_geometry(cfg) -> CzGeometry:
    g = cfg.custom.geometry

    x = Symbol("x")
    y = Symbol("y")

    r_crystal = float(g.crystal_radius)
    h_crystal = float(g.crystal_height)

    r_melt = float(g.melt_radius)
    h_melt = float(g.melt_height)

    t_wall = float(g.crucible_wall)
    t_bottom = float(g.crucible_bottom)

    r_outer = float(g.outer_radius)
    h_outer = float(g.outer_height)

    r_crucible_outer = r_melt + t_wall
    y_sl = h_melt
    y_crystal_top = h_melt + h_crystal

    # Region 1: crystal
    crystal = Rectangle((0.0, h_melt), (r_crystal, h_melt + h_crystal))

    # Region 2: melt
    melt = Rectangle((0.0, 0.0), (r_melt, h_melt))

    # Region 3: crucible = bottom + side wall
    crucible_bottom = Rectangle((0.0, -t_bottom), (r_crucible_outer, 0.0))
    crucible_side = Rectangle((r_melt, 0.0), (r_crucible_outer, h_melt))
    crucible = crucible_bottom + crucible_side

    # Outer box
    outer_box = Rectangle((0.0, -t_bottom), (r_outer, h_outer))

    # Region 4: insulation / support solid
    # Reduced-order V1 simplification:
    # insulation = everything in outer_box excluding crystal, melt, and crucible
    insulation = outer_box - crystal - melt - crucible

    return CzGeometry(
        x=x,
        y=y,
        crystal=crystal,
        melt=melt,
        crucible=crucible,
        insulation=insulation,
        outer_box=outer_box,
        r_crystal=r_crystal,
        h_crystal=h_crystal,
        r_melt=r_melt,
        h_melt=h_melt,
        t_crucible_wall=t_wall,
        t_crucible_bottom=t_bottom,
        r_crucible_outer=r_crucible_outer,
        r_outer=r_outer,
        h_outer=h_outer,
        y_sl=y_sl,
        y_crystal_top=y_crystal_top,
    )