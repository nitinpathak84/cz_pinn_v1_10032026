import physicsnemo.sym
from physicsnemo.sym.hydra import PhysicsNeMoConfig
from physicsnemo.sym.solver import Solver
from physicsnemo.sym.domain import Domain

from cz.geometry import build_cz_geometry
from cz.networks import build_cz_model
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
    model = build_cz_model(cfg)

    train_nodes = model["train_nodes"]
    inferencer_nodes = model["inferencer_nodes"]

    domain = Domain()

    add_boundary_constraints(domain, train_nodes, geo, cfg)
    add_interior_constraints(domain, train_nodes, geo, cfg)
    add_interface_constraints(domain, train_nodes, geo, cfg)
    add_inferencers(domain, inferencer_nodes, geo, cfg)
    add_monitors(domain, train_nodes, geo, cfg)

    solver = Solver(cfg, domain)
    solver.solve()


if __name__ == "__main__":
    run()
