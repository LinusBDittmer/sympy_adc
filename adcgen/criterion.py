from sympy import Rational, latex, sympify, Mul, S, symbols, Add
from sympy.physics.secondquant import NO, F, Fd

from .sympy_objects import AntiSymmetricTensor, RotationTensor
from .indices import Indices
from .misc import cached_member, Inputerror, process_arguments, transform_to_tuple, validate_input
from .simplify import simplify
from .func import gen_term_orders, wicks
from .expr_container import Expr
from .operators import Operators
from .groundstate import GroundState

class Criterion:

    def __init__(self, name):
        self._name = name

    @process_arguments
    @cached_member
    def expr(self):
        """
        TO BE OVERRIDEN
        """
        return S.Zero

    def gradient(self):
        pass

    def hessian(self):
        pass

class CNC_Criterion(Criterion):

    def __init__(self, name, c_order, nc_order, h_c, h_nc):
        super().__init__(name)
        self._c_order = c_order
        self._nc_order = nc_order
        self._h_c = h_c
        self._h_nc = h_nc
        self._gs_c = GroundState(h_c)
        self._gs_nc = GroundState(h_nc)

class CNC_Overlap(CNC_Criterion):

    def __init__(self, c_order, nc_order, h_c, h_nc):
        super().__init__("CNC_Overlap", c_order, nc_order, h_c, h_nc)

    @process_arguments
    @cached_member
    def expr(self):
        bra_list = [self._gs_c.psi(o, "bra") for o in range(self._c_order)]
        ket_list = [self._gs_nc.psi(o, "ket") for o in range(self._nc_order)]

        braket = Sum(*bra_list) * Sum(*ket_list)
        
        braket_s = wicks(braket, keep_only_fully_contracted=True, simplify_kronecker_deltas=True)
        return braket_s

class CNC_Energy(CNC_Criterion):

    def __init__(self, c_order, nc_order, h_c, h_nc):
        super().__init__("CNC_Energy", c_order, nc_order, h_c, h_nc)

    @process_arguments
    @cached_member
    def expr(self):
        c_list = [self._gs_c.energy(o) for o in range(self._c_order)]
        nc_list = [self._gs_nc.energy(o) for o in range(self._nc_order)]

        e_c = Add(*c_list)
        e_nc = Add(*nc_list)
        return (e_nc - e_c) * (e_nc - e_c)


