from .misc import cached_property, cached_member
from .indices import Indices
from .rules import Rules
from .sympy_objects import AntiSymmetricTensor, NonSymmetricTensor
from .logger import log

from sympy import Rational, factorial, Mul, latex
from sympy.physics.secondquant import Fd, F


class Operators:
    """
    Constructs operators, like the zeroth and first order Hamiltonian or
    arbitrary N-particle operators.

    Parameters
    ----------
    variant : str, optional
        Defines the partitioning of the Hamiltonian.
        (default: the MP Hamiltonian)
    """
    def __init__(self, variant: str = "mp"):
        self._indices = Indices()
        self._variant = variant

    @cached_property
    def h0(self):
        """Constructs the zeroth order Hamiltonian."""
        if self._variant == 'mp':
            return self.mp_h0()
        elif self._variant == 're':
            return self.re_h0()
        elif self._variant == 'ordmp':
            return self.ordmp_h0()
        else:
            raise NotImplementedError(
                f"H0 not implemented for {self._variant}"
            )

    @cached_property
    def h1(self):
        """Constructs the first order Hamiltonian."""
        if self._variant == 'mp':
            return self.mp_h1()
        elif self._variant == 're':
            return self.re_h1()
        elif self._variant == 'ordmp':
            return self.ordmp_h1()
        else:
            raise NotImplementedError(
                f"H1 not implented for {self._variant}"
            )

    @cached_member
    def operator(self, n_create: int, n_annihilate: int, idx_list=None):
        """
        Constructs an arbitrary second quantized operator placing creation
        operators to the left of annihilation operators.

        Parameters
        ----------
        n_create : int
            The number of creation operators. Placed left of the annihilation
            operators.
        n_annihilate : int
            The number of annihilation operators. Placed right of the creation
            operators.
        """
        validate_input(opstring=opstring)
        if idx_list is None:
            idx = self._indices.get_generic_indices(n_g=n_create+n_annihilate)["general"]
            create = idx[:n_create]
            annihilate = idx[n_create:]
        else:
            create, annihilate = idx_list
        pref = Rational(1, factorial(n_create) * factorial(n_annihilate))
        d = AntiSymmetricTensor('d', create, annihilate)
        op = Mul(*[Fd(s) for s in create]) * \
            Mul(*[F(s) for s in reversed(annihilate)])
        log(f"Operator built: {pref * d * op}")
        return pref * d * op, None

    @staticmethod
    def mp_h0():
        """Constructs the zeroth order MP-Hamiltonian."""
        idx_cls = Indices()
        p, q = idx_cls.get_indices('pq')['general']
        f = AntiSymmetricTensor('f', (p,), (q,))
        pq = Fd(p) * F(q)
        h0 = f * pq
        log("H0 = ", latex(h0))
        return h0, None

    @staticmethod
    def mp_h1():
        """Constructs the first order MP-Hamiltonian."""
        idx_cls = Indices()
        p, q, r, s = idx_cls.get_indices('pqrs')['general']
        # get an occ index for 1 particle part of H1
        occ = idx_cls.get_generic_indices(n_o=1)['occ'][0]
        v1 = AntiSymmetricTensor('V', (p, occ), (q, occ))
        pq = Fd(p) * F(q)
        v2 = AntiSymmetricTensor('V', (p, q), (r, s))
        pqsr = Fd(p) * Fd(q) * F(s) * F(r)
        h1 = -v1 * pq + Rational(1, 4) * v2 * pqsr
        log("H1 = ", latex(h1))
        return h1, None

    @staticmethod
    def ordmp_h0():
        idx_cls = Indices()
        p, q, r, s = idx_cls.get_indices('pqrs')['general']
        u_pr = NonSymmetricTensor('U', (p, r))
        u_qs = NonSymmetricTensor('U', (q, s))
        f = AntiSymmetricTensor('f', (r,), (s,))
        pq = Fd(p) * F(q)
        h0 = u_pr * f * u_qs * pq
        rules = Rules(forbidden_tensor_blocks = {'U': ('ov', 'vo')})
        log("H0 = ", latex(h0))
        return h0, rules

    @staticmethod
    def ordmp_h1():
        idx_cls = Indices()
        p, q, r, s = idx_cls.get_indices('pqrs')['general']
        occ = idx_cls.get_generic_indices(n_o=1)['occ'][0]

        h = AntiSymmetricTensor('f', (p,), (q,))
        hr = AntiSymmetricTensor('f', (r,), (s,))
        v1 = AntiSymmetricTensor('V', (p, occ), (q, occ))
        v2 = AntiSymmetricTensor('V', (p, q), (r, s))
        
        u_pr = NonSymmetricTensor('U', (p, r))
        u_qs = NonSymmetricTensor('U', (q, s))

        pq = Fd(p) * F(q)
        pqsr = Fd(p) * Fd(q) * F(s) * F(r)

        h1 = (-u_pr * hr * u_qs + h - v1) * pq + Rational(1, 4) * v2 * pqsr
        
        rules = Rules(forbidden_tensor_blocks = {'U': ('ov', 'vo')})
        log("H1 = ", latex(h1))
        return h1, rules

    @staticmethod
    def re_h0():
        """Constructs the zeroth order RE-Hamiltonian."""
        idx_cls = Indices()
        p, q, r, s = idx_cls.get_indices('pqrs')['general']
        # get an occ index for 1 particle part of H0
        occ = idx_cls.get_generic_indices(n_o=1)['occ'][0]

        f = AntiSymmetricTensor('f', (p,), (q,))
        piqi = AntiSymmetricTensor('V', (p, occ), (q, occ))
        pqrs = AntiSymmetricTensor('V', (p, q), (r, s))
        op_pq = Fd(p) * F(q)
        op_pqsr = Fd(p) * Fd(q) * F(s) * F(r)

        h0 = f * op_pq - piqi * op_pq + Rational(1, 4) * pqrs * op_pqsr
        log("H0 = ", latex(h0))
        # construct the rules for forbidden blocks in H0
        # we are not in a real orbital basis!! -> More canonical blocks
        rules = Rules(forbidden_tensor_blocks={
            'f': ('ov', 'vo'),
            'V': ('ooov', 'oovv', 'ovvv', 'ovoo', 'vvoo', 'vvov')
        })
        return h0, rules

    @staticmethod
    def re_h1():
        """Constructs the first order RE-Hamiltonian."""
        idx_cls = Indices()
        p, q, r, s = idx_cls.get_indices('pqrs')['general']
        # get an occ index for 1 particle part of H0
        occ = idx_cls.get_generic_indices(n_o=1)['occ'][0]

        f = AntiSymmetricTensor('f', (p,), (q,))
        piqi = AntiSymmetricTensor('V', (p, occ), (q, occ))
        pqrs = AntiSymmetricTensor('V', (p, q), (r, s))
        op_pq = Fd(p) * F(q)
        op_pqsr = Fd(p) * Fd(q) * F(s) * F(r)

        h1 = f * op_pq - piqi * op_pq + Rational(1, 4) * pqrs * op_pqsr
        log("H1 = ", latex(h1))
        # construct the rules for forbidden blocks in H1
        rules = Rules(forbidden_tensor_blocks={
            'f': ['oo', 'vv'],
            'V': ['oooo', 'ovov', 'vvvv']
        })
        return h1, rules

    def __eq__(self, other):
        if isinstance(other, Operators):
            return self._variant == other._variant
        return False
