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
