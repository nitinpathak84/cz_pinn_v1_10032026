"""
Custom steady 2D axisymmetric diffusion equation in normalized (r,z) coordinates.

Independent variables:
    x = r_hat
    y = z_hat

Unknown:
    T(x, y)

Equation:
    T_xx + (1 / (x + eps_r)) * T_x + aspect_sq * T_yy = 0

where:
    aspect_sq = (R_ref / Z_ref)^2
"""

from sympy import Symbol, Function, Number
from physicsnemo.sym.eq.pde import PDE


class AxisymmetricDiffusion(PDE):
    name = "AxisymmetricDiffusion"

    def __init__(
        self,
        T="theta",
        aspect_sq=1.0,
        eps_r=1.0e-4,
        source=0.0,
    ):
        self.T = T
        self.aspect_sq = float(aspect_sq)
        self.eps_r = float(eps_r)
        self.source = float(source)

        x = Symbol("x")
        y = Symbol("y")

        input_variables = {"x": x, "y": y}

        T_sym = Function(T)(*input_variables)

        aspect_sq = Number(self.aspect_sq)
        eps_r = Number(self.eps_r)
        source = Number(self.source)

        r_eff = x + eps_r

        self.equations = {}
        self.equations[f"axisym_diffusion_{self.T}"] = (
            T_sym.diff(x, 2)
            + (1.0 / r_eff) * T_sym.diff(x)
            + aspect_sq * T_sym.diff(y, 2)
            - source
        )