from .misc import cached_member
from .indices import Indices
from .simplify import simplify, simplify_unitary
from .derivative import implicit_derivative, premade_deriv_dicts
from .expr_container import Expr
from .func import evaluate_deltas, wicks, remove_disconnected_terms, remove_mean_field_terms
from .operators import Operators
from .groundstate import GroundState
from .spatial_orbitals import transform_to_spatial_orbitals
from .sympy_objects import AntiSymmetricTensor, NonSymmetricTensor
from .generate_code import generate_code

from sympy import Add

class Criterion:

    def __init__(self, name: str, real: bool = True, restricted: bool = True):
        self._name = name
        self._real = real
        self._restricted = restricted
        self._target_idx = None
        self._grad = dict()
        self._hess = dict()
        self._grad_idx = dict()
        self._hess_idx = dict()

    @cached_member
    def expr(self):
        """
        TO BE OVERRIDEN
        """
        return S.Zero

    def _preprocess(self, expr):
        if isinstance(expr, Expr):
            expr = expr.sympy
        expr = Expr(expr, self._real)
        expr.expand_intermediates()
        expr.use_symbolic_denominators()
        expr = simplify_unitary(expr, t_name='U')
        expr = Expr(evaluate_deltas(expr.sympy, target_idx=self._target_idx), real=self._real)
        
        if self._target_idx is None:
            expr = transform_to_spatial_orbitals(expr, '', '', restricted=self._restricted)
            expr = simplify(expr)
            return expr
        else:
            raise NotImplementedError('Only scalar quantities implemented so far.')
                
    def _postprocess(self, expr, target_idx):
        if not isinstance(expr, Expr):
            expr = Expr(expr, real=self._real)
        if self._target_idx is not None:
            target_idx += self._target_idx
        expr.expand()
        expr = Expr(evaluate_deltas(expr.sympy, target_idx=target_idx), real=self._real)
        expr = simplify(expr)
        expr.substitute_contracted()
        return expr

    def gradient(self):
        expr = self._preprocess(self.expr())
        
        deriv1, deriv2 = premade_deriv_dicts('orbital rotation')
        indices = Indices()

        if self._restricted:
            grad_oo_idx = indices.get_indices('ij', 'aa')['occ_a']
            grad_vv_idx = indices.get_indices('ab', 'aa')['virt_a']
            
            print("Occupied Gradient")
            grad_oo = implicit_derivative(expr, 1, grad_oo_idx, deriv1)
            print("Virtual Gradient")
            grad_vv = implicit_derivative(expr, 1, grad_vv_idx, deriv1)

            grad_oo = self._postprocess(grad_oo, grad_oo_idx)
            grad_vv = self._postprocess(grad_vv, grad_vv_idx)

            self._grad.update({"oo": grad_oo, "vv": grad_vv})
            self._grad_idx.update({"oo": grad_oo_idx, "vv": grad_vv_idx})
        else:
            grad_ooaa_idx = indices.get_indices('ij', 'aa')['occ_a']
            grad_oobb_idx = indices.get_indices('ij', 'bb')['occ_b']
            grad_vvaa_idx = indices.get_indices('ab', 'aa')['virt_a']
            grad_vvbb_idx = indices.get_indices('ab', 'bb')['virt_b']

            grad_ooaa = implicit_derivative(expr, 1, grad_ooaa_idx, deriv1)
            grad_oobb = implicit_derivative(expr, 1, grad_oobb_idx, deriv1)
            grad_vvaa = implicit_derivative(expr, 1, grad_vvaa_idx, deriv1)
            grad_vvbb = implicit_derivative(expr, 1, grad_vvbb_idx, deriv1)

            grad_ooaa = self._postprocess(grad_ooaa, grad_ooaa_idx)
            grad_oobb = self._postprocess(grad_oobb, grad_oobb_idx)
            grad_vvaa = self._postprocess(grad_vvaa, grad_vvaa_idx)
            grad_vvbb = self._postprocess(grad_vvbb, grad_vvbb_idx)

            self._grad.update({"oo_aa": grad_ooaa, "oo_bb": grad_oobb, 
                               "vv_aa": grad_vvaa, "vv_bb": grad_vvbb})
            self._grad_idx.update({"oo_aa": grad_ooaa_idx, "oo_bb": grad_oobb_idx,
                                   "vv_aa": grad_vvaa_idx, "vv_bb": grad_vvbb_idx})
        return self._grad

    def hessian(self):
        expr = self._preprocess(self.expr())

        deriv1, deriv2 = premade_deriv_dicts('orbital rotation')
        indices = Indices()
        if self._restricted:
            i, j, k, l = indices.get_indices('ijkl', 'aaaa')['occ_a']
            a, b, c, d = indices.get_indices('abcd', 'aaaa')['virt_a']
            
            hess_oooo_idx = ((i, j), (k, l))
            hess_vvvv_idx = ((a, b), (c, d))
            hess_oovv_idx = ((i, j), (a, b))
            
            print("Hessian Block 1")
            hess_oooo = implicit_derivative(expr, 2, hess_oooo_idx, deriv1, deriv2)
            print("Hessian Block 2")
            hess_oovv = implicit_derivative(expr, 2, hess_oovv_idx, deriv1, deriv2)
            print("Hessian Block 3")
            hess_vvvv = implicit_derivative(expr, 2, hess_vvvv_idx, deriv1, deriv2)

            hess_oooo_idx = hess_oooo_idx[0] + hess_oooo_idx[1]
            hess_oovv_idx = hess_oovv_idx[0] + hess_oovv_idx[1]
            hess_vvvv_idx = hess_vvvv_idx[0] + hess_vvvv_idx[1]

            hess_oooo = self._postprocess(hess_oooo, hess_oooo_idx)
            hess_oovv = self._postprocess(hess_oovv, hess_oovv_idx)
            hess_vvvv = self._postprocess(hess_vvvv, hess_vvvv_idx)

            self._hess.update({'oooo': hess_oooo, 'oovv': hess_oovv, 'vvvv': hess_vvvv})
            self._hess_idx.update({'oooo': list(hess_oooo_idx), 'oovv': list(hess_oovv_idx),
                                   'vvvv': list(hess_vvvv_idx)})
        else:
            ia, ja, ka, la = indices.get_indices('ijkl', 'aaaa')['occ_a']
            ib, jb, kb, lb = indices.get_indices('ijkl', 'bbbb')['occ_b']
            aa, ba, ca, da = indices.get_indices('abcd', 'aaaa')['virt_a']
            ab, bb, cb, db = indices.get_indices('abcd', 'bbbb')['virt_b']

            hess_ooooaaaa_idx = ((ia, ja), (ka, la))
            hess_ooooaabb_idx = ((ia, ja), (kb, lb))
            hess_oooobbaa_idx = ((ib, jb), (ka, la))
            hess_oooobbbb_idx = ((ib, jb), (kb, lb))

            hess_oovvaaaa_idx = ((ia, ja), (aa, ba))
            hess_oovvaabb_idx = ((ia, ja), (ab, bb))
            hess_oovvbbaa_idx = ((ib, jb), (aa, ba))
            hess_oovvbbbb_idx = ((ib, jb), (ab, bb))

            hess_vvvvaaaa_idx = ((aa, ba), (ca, da))
            hess_vvvvaabb_idx = ((aa, ba), (cb, db))
            hess_vvvvbbaa_idx = ((ab, bb), (ca, da))
            hess_vvvvbbbb_idx = ((ab, bb), (cb, db))

            
            hess_ooooaaaa = implicit_derivative(expr, 2, hess_ooooaaaa_idx, deriv1, deriv2)
            hess_ooooaabb = implicit_derivative(expr, 2, hess_ooooaabb_idx, deriv1, deriv2)
            hess_oooobbaa = implicit_derivative(expr, 2, hess_oooobbaa_idx, deriv1, deriv2)
            hess_oooobbbb = implicit_derivative(expr, 2, hess_oooobbbb_idx, deriv1, deriv2)
            
            hess_oovvaaaa = implicit_derivative(expr, 2, hess_oovvaaaa_idx, deriv1, deriv2)
            hess_oovvaabb = implicit_derivative(expr, 2, hess_oovvaabb_idx, deriv1, deriv2)
            hess_oovvbbaa = implicit_derivative(expr, 2, hess_oovvbbaa_idx, deriv1, deriv2)
            hess_oovvbbbb = implicit_derivative(expr, 2, hess_oovvbbbb_idx, deriv1, deriv2)

            hess_vvvvaaaa = implicit_derivative(expr, 2, hess_vvvvaaaa_idx, deriv1, deriv2)
            hess_vvvvaabb = implicit_derivative(expr, 2, hess_vvvvaabb_idx, deriv1, deriv2)
            hess_vvvvbbaa = implicit_derivative(expr, 2, hess_vvvvbbaa_idx, deriv1, deriv2)
            hess_vvvvbbbb = implicit_derivative(expr, 2, hess_vvvvbbbb_idx, deriv1, deriv2)

            hess_ooooaaaa_idx = hess_ooooaaaa_idx[0] + hess_ooooaaaa_idx[1]
            hess_ooooaabb_idx = hess_ooooaabb_idx[0] + hess_ooooaabb_idx[1]
            hess_oooobbaa_idx = hess_oooobbaa_idx[0] + hess_oooobbaa_idx[1]
            hess_oooobbbb_idx = hess_oooobbbb_idx[0] + hess_oooobbbb_idx[1]
            hess_oovvaaaa_idx = hess_oovvaaaa_idx[0] + hess_oovvaaaa_idx[1]
            hess_oovvaabb_idx = hess_oovvaabb_idx[0] + hess_oovvaabb_idx[1]
            hess_oovvbbaa_idx = hess_oovvbbaa_idx[0] + hess_oovvbbaa_idx[1]
            hess_oovvbbbb_idx = hess_oovvbbbb_idx[0] + hess_oovvbbbb_idx[1]
            hess_vvvvaaaa_idx = hess_vvvvaaaa_idx[0] + hess_vvvvaaaa_idx[1]
            hess_vvvvaabb_idx = hess_vvvvaabb_idx[0] + hess_vvvvaabb_idx[1]
            hess_vvvvbbaa_idx = hess_vvvvbbaa_idx[0] + hess_vvvvbbaa_idx[1]
            hess_vvvvbbbb_idx = hess_vvvvbbbb_idx[0] + hess_vvvvbbbb_idx[1]

            hess_ooooaaaa = self._postprocess(hess_ooooaaaa, hess_ooooaaaa_idx)
            hess_ooooaabb = self._postprocess(hess_ooooaabb, hess_ooooaabb_idx)
            hess_oooobbaa = self._postprocess(hess_oooobbaa, hess_oooobbaa_idx)
            hess_oooobbbb = self._postprocess(hess_oooobbbb, hess_oooobbbb_idx)
            
            hess_oovvaaaa = self._postprocess(hess_oovvaaaa, hess_oovvaaaa_idx)
            hess_oovvaabb = self._postprocess(hess_oovvaabb, hess_oovvaabb_idx)
            hess_oovvbbaa = self._postprocess(hess_oovvbbaa, hess_oovvbbaa_idx)
            hess_oovvbbbb = self._postprocess(hess_oovvbbbb, hess_oovvbbbb_idx)
           
            hess_vvvvaaaa = self._postprocess(hess_vvvvaaaa, hess_vvvvaaaa_idx)
            hess_vvvvaabb = self._postprocess(hess_vvvvaabb, hess_vvvvaabb_idx)
            hess_vvvvbbaa = self._postprocess(hess_vvvvbbaa, hess_vvvvbbaa_idx)
            hess_vvvvbbbb = self._postprocess(hess_vvvvbbbb, hess_vvvvbbbb_idx)
            
            self._hess.update({'oooo_aaaa': hess_ooooaaaa,
                               'oooo_aabb': hess_ooooaabb,
                               'oooo_bbaa': hess_oooobbaa,
                               'oooo_bbbb': hess_oooobbbb,
                               'oovv_aaaa': hess_oovvaaaa,
                               'oovv_aabb': hess_oovvaabb,
                               'oovv_bbaa': hess_oovvbbaa,
                               'oovv_bbbb': hess_oovvbbbb,
                               'vvvv_aaaa': hess_vvvvaaaa,
                               'vvvv_aabb': hess_vvvvaabb,
                               'vvvv_bbaa': hess_vvvvbbaa,
                               'vvvv_bbbb': hess_vvvvbbbb
                              })
            
            self._hess_idx.update({'oooo_aaaa': list(hess_ooooaaaa_idx),
                                   'oooo_aabb': list(hess_ooooaabb_idx),
                                   'oooo_bbaa': list(hess_oooobbaa_idx),
                                   'oooo_bbbb': list(hess_oooobbbb_idx),
                                   'oovv_aaaa': list(hess_oovvaaaa_idx),
                                   'oovv_aabb': list(hess_oovvaabb_idx),
                                   'oovv_bbaa': list(hess_oovvbbaa_idx),
                                   'oovv_bbbb': list(hess_oovvbbbb_idx),
                                   'vvvv_aaaa': list(hess_vvvvaaaa_idx),
                                   'vvvv_aabb': list(hess_vvvvaabb_idx),
                                   'vvvv_bbaa': list(hess_vvvvbbaa_idx),
                                   'vvvv_bbbb': list(hess_vvvvbbbb_idx)
                                  })
        return self._hess

    def _build_dummy_tensor(self, symbol: str, indices):
        occ_idx, vir_idx, g_idx = [], [], []
        for index in indices:
            if index.space[0] == 'o':
                occ_idx.append(index)
            elif index.space[0] == 'v':
                vir_idx.append(index)
            else:
                g_idx.append(index)
        if len(g_idx) == 0:
            t = AntiSymmetricTensor(symbol, tuple(vir_idx), tuple(occ_idx))
        else:
            g_idx.extend(occ_idx)
            g_idx.extend(vir_idx)
            t = NonSymmetricTensor(symbol, tuple(g_idx))
        return Expr(t, real=True)

    def dump_formulas(self, io=None, terms_per_line: int = 3, spin_as_overbar: bool=True) -> None:
        if io is None:
            import sys
            io = sys.stdout
        
        # Expression dump
        output_lines = []
        e = self.expr()
        e = self._postprocess(self._preprocess(e), self._target_idx)
        output_lines.append("Expression:\n")
        output_lines.append(e.print_latex(terms_per_line=terms_per_line, 
                                          spin_as_overbar=spin_as_overbar))

        # Gradient dump
        for space, formula in self._grad.items():
            s = space.replace("_", "/").replace("a", r"\alpha").replace("b", r"\beta")
            t = self._build_dummy_tensor('G', self._grad_idx[space])
            loc = f"Gradient block {s}:\n"
            loc += t.print_latex(spin_as_overbar=spin_as_overbar) + " = "
            loc += formula.print_latex(terms_per_line=terms_per_line,
                                       spin_as_overbar=spin_as_overbar)
            output_lines.append(loc)
        # Hessian dump
        for space, formula in self._hess.items():
            s = space.replace("_", "/").replace("a", r"\alpha").replace("b", r"\beta")
            t = self._build_dummy_tensor('H', self._hess_idx[space])
            loc = f"Hessian block {s}:\n"
            loc += t.print_latex(spin_as_overbar=spin_as_overbar) + " = "
            loc += formula.print_latex(terms_per_line=terms_per_line,
                   spin_as_overbar=spin_as_overbar)
            output_lines.append(loc)

        io.write("\n\n".join(output_lines) + "\n")

    def dump_code(self, io=None, itmd_start_idx: str = 0, comment: str = '#'):
        if io is None:
            import sys
            io = sys.stdout

        # Expression 
        itmd_index = 0
        codelines = []
        e = self.expr()
        e = self._postprocess(self._preprocess(e), self._target_idx)
        ti = self._target_idx
        if ti is None:
            ti = []
        code, ishift = generate_code(e, target_indices=ti, backend='einsum',
                                     optimize_contractions=False, itmd_start_idx=itmd_index, 
                                     strict_target_idx=False)
        loc = f"{comment} " + "=" * 80 + f"\n{comment} EXPRESSION\n" + code
        codelines.append(loc)

        # Gradient
        itmd_index = 0
        for space, formula in self._grad.items():
            code, ishift = generate_code(formula, target_indices=self._grad_idx[space],
                                         backend='einsum', optimize_contractions=False, 
                                         itmd_start_idx=itmd_index, strict_target_idx=False)
            loc = f"{comment} " + "=" * 80 + f"\n{comment} GRADIENT Block {space.lower()}\n" + code
            itmd_index += ishift
            codelines.append(loc)

        # Hessian
        itmd_index = 0
        for space, formula in self._hess.items():
            code, ishift = generate_code(formula, target_indices=self._hess_idx[space],
                                         backend='einsum', optimize_contractions=False, 
                                         itmd_start_idx=itmd_index, strict_target_idx=False)
            loc = f"{comment} " + "=" * 80 + f"\n{comment} HESSIAN Block {space.lower()}\n" + code
            itmd_index += ishift
            codelines.append(loc)

        io.write("\n\n".join(codelines) + "\n")


class MaxEnergy(Criterion):

    def __init__(self, order, operator, mp = None, restricted: bool = True):
        super().__init__("Max Energy", real=True, restricted=restricted)
        self._h = operator
        if mp is None:
            self._mp = GroundState(self._h)
        else:
            self._mp = mp
        self._order = order

    def expr(self):
        e = Expr(self._mp.energy(self._order), real=True)
        return e

class Variational(Criterion):

    def __init__(self, order, operator, mp = None, restricted: bool = True, 
                 remove_disconnected: bool = True, remove_mean_field: bool = True):
        super().__init__("Variational", real=True, restricted=restricted)
        self._h = operator
        if mp is None:
            self._mp = GroundState(self._h)
        else:
            self._mp = mp
        self._order = order
        self.rm_disc = remove_disconnected
        self.rm_mf = remove_mean_field

    def expr(self):
        hamiltonian = self._h.h0[0] + self._h.h1[0]
        ket = self._mp.psi(order=self._order-1, braket='ket')
        bra = self._mp.psi(order=self._order-1, braket='bra')
        numerator = wicks(bra * hamiltonian * ket, simplify_kronecker_deltas=True)
        if self.rm_disc:
            numerator = remove_disconnected_terms(Expr(numerator, real=True)).sympy
        if self.rm_mf:
            numerator = remove_mean_field_terms(Expr(numerator, real=True)).sympy
        denominator = wicks(bra * ket, simplify_kronecker_deltas=True)
        e = simplify(Expr(numerator)) / denominator
        print("Expr: " + Expr(e).print_latex())
        return e


class Projective(Criterion):

    def __init__(self, order, operator, mp = None, restricted: bool = True):
        super().__init__("Projective", real=True, restricted=restricted)
        self._h = operator
        if mp is None:
            self._mp = GroundState(self._h)
        else:
            self._mp = mp
        self._order = order

    def expr(self):
        hamiltonian = self._h.h0[0] + self._h.h1[0]
        ket = Add(*[self._mp.psi(order=i, braket='ket') for i in range(self._order)])
        pb = self._mp.psi(order=self._order-1, braket='bra')
        projective_bra = 1
        for i in pb.args:
            if not isinstance(i, AntiSymmetricTensor):
                projective_bra *= i
        pb_idx = tuple(set(Expr(projective_bra, real=True).idx))
        energy = Add(*[self._mp.energy(order=i) for i in range(self._order+1)])
        energy = simplify_unitary(Expr(energy), t_name='U', block_diagonal=True, evaluate_deltas=True)
        energy = simplify(energy).sympy
        e = projective_bra * hamiltonian * ket - energy * projective_bra * ket
        e2 = wicks(e, simplify_kronecker_deltas=False)
        e2 = evaluate_deltas(e2, target_idx=pb_idx)
        e2 = simplify(Expr(e2))
        print(f"Criterion Expr: {e2.substitute_contracted().print_latex(terms_per_line=2)}")
        return e2

class Norm(Criterion):

    def __init__(self, order, operator, mp = None, restricted: bool = True):
        super().__init__("Projective", real=True, restricted=restricted)
        self._h = operator
        if mp is None:
            self._mp = GroundState(self._h)
        else:
            self._mp = mp
        self._order = order

    def expr(self):
        # Each order correction is multiplied with itself
        kets = [self._mp.psi(order=i, braket='ket') for i in range(self._order+1)]
        bras = [self._mp.psi(order=i, braket='bra') for i in range(self._order+1)]
        wf = Add(*[b*k for b, k in zip(bras, kets)])
        e = Expr(wicks(wf, simplify_kronecker_deltas=True))
        e = simplify_unitary(e, t_name='U', block_diagonal=True, evaluate_deltas=True)
        e = simplify(e)
        return e
