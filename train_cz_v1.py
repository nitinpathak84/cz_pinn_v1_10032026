import physicsnemo.sym
from physicsnemo.sym.hydra import PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain

from cz.geometry import build_cz_geometry
from cz.networks import build_cz_nodes
from cz.constraints import (
    add_boundary_constraints,
    add_interior_constraints,
    add_interface_constraints,
    add_inferencers,
    add_monitors,
)


@physicsnemo.sym.main(config_path="conf", config_name="config")
def run(cfg: PhysicsNeMoConfig) -> None:
    geo = build_cz_geometry(cfg)
    nodes = build_cz_nodes(cfg)

    domain = Domain()

    add_boundary_constraints(domain, nodes, geo, cfg)
    add_interior_constraints(domain, nodes, geo, cfg)
    add_interface_constraints(domain, nodes, geo, cfg)
    add_inferencers(domain, nodes, geo, cfg)
    add_monitors(domain, nodes, geo, cfg)

    solver = Solver(cfg, domain)
    solver.solve()


if __name__ == "__main__":
    run()