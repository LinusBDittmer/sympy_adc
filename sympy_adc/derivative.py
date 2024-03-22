from . import expr_container as e
from .indices import minimize_tensor_indices, Index
from .simplify import simplify
from .intermediates import Intermediates
from sympy import Rational, diff, S
from sympy import Add, Mul, Pow

def derivative(expr: e.Expr, t_string: str):
    """Computes the derivative of an expression with respect to a tensor.
       The derivative is separated block whise, e.g, terms that contribute to
       the derivative w.r.t. the oooo ERI block are separated from terms that
       contribute to the ooov block.
       Assumptions of the input expression are NOT updated or modified.
       The derivative is NOT simplified."""

    if not isinstance(t_string, str):
        raise TypeError("Tensor name needs to be provided as str.")

    expr = expr.expand()
    if not isinstance(expr, e.Expr):  # ensure expr is in a container
        expr = e.Expr(expr)

    # create some Dummy Symbol. Replace the tensor with the Symbol and
    # compute the derivative with respect to the Symbol. Afterwards
    # resubstitute the Tensor for the Dummy Symbol.
    x = Index('x')

    derivative = {}
    for term in expr.terms:
        assumptions = term.assumptions
        objects = term.objects
        # - find all occurences of the desired tensor
        tensor_obj = []
        remaining_obj = e.Expr(1, **term.assumptions)
        for obj in objects:
            if obj.name == t_string:
                tensor_obj.append(obj)
            else:
                remaining_obj *= obj

        # - extract the names of target indices of the term
        target_names_by_space = {}
        for s in term.target:
            if (sp := s.space) not in target_names_by_space:
                target_names_by_space[sp] = set()
            target_names_by_space[sp].add(s.name)

        # 2) go through the tensor_obj list and compute the derivative
        #    for all found occurences one after another (product rule)
        for i, obj in enumerate(tensor_obj):
            # - extract the exponent of the tensor
            exponent = obj.exponent
            # - rebuild the term without the current occurence of the
            #   tensor obj
            deriv_contrib = remaining_obj.copy()
            for other_i, other_obj in enumerate(tensor_obj):
                if i != other_i:
                    deriv_contrib *= other_obj
            # - minimize the indices of the removed tensor
            _, perms = minimize_tensor_indices(obj.idx, target_names_by_space)
            # - apply the permutations to the remaining term
            deriv_contrib = deriv_contrib.permute(*perms)
            if deriv_contrib.sympy is S.Zero:
                raise RuntimeError(f"Mnimization permutations {perms} let "
                                   f"the remaining term {deriv_contrib} "
                                   "vanish.")
            # - Apply the permutations to the object. Might introduce
            #   a prefactor of -1 that we need to move to the deriv_contrib.
            #   Also the indices might be further minimized due to the
            #   symmetry of the tensor obj
            obj: e.Term = obj.permute(*perms).terms[0]
            if (factor := obj.prefactor) < 0:
                deriv_contrib *= factor
            # - Apply the symmetry of the removed tensor to the remaining
            #   term to ensure that the result has the correct symmetry.
            #   Also replace the removed tensor by a Dummy Variable x.
            #   This allows to compute the symbolic derivative with diff.
            tensor_sym = obj.symmetry()
            deriv_contrib *= Rational(1, len(tensor_sym) + 1)
            symmetrized_deriv_contrib = deriv_contrib.sympy * x**exponent
            for perms, factor in tensor_sym.items():
                symmetrized_deriv_contrib += (
                    deriv_contrib.copy().permute(*perms).sympy *
                    factor * x**exponent
                )
            # - compute the derivative with respect to x
            symmetrized_deriv_contrib = diff(symmetrized_deriv_contrib, x)
            # - replace x by the removed tensor (due to diff the exponent
            #   is lowered by 1)
            obj = obj.tensors
            assert len(obj) == 1
            obj = obj[0]
            symmetrized_deriv_contrib = (
                symmetrized_deriv_contrib.subs(x, obj)
            )
            # - sort the derivative according to the space of the minimal
            #   tensor indices
            #   -> sort the derivative block whise.
            space = obj.space
            if space not in derivative:
                derivative[space] = e.Expr(0, **assumptions)
            derivative[space] += symmetrized_deriv_contrib
    return derivative

def premade_deriv_dicts(t: str):
    if t not in {'orbital rotation'}:
        raise NotImplementedError('No premade derivative dict for {t} exist.')

    deriv_dict1: dict = {}
    deriv_dict2: dict = {}

    if t == 'orbital rotation':
        from .func import eri_unitary_deriv1, eri_unitary_deriv2
        
        def eri1_wrapper(idx, didx):
            return eri_unitary_deriv1(idx, didx, notation='c')

        def eri2_wrapper(idx, didx1, didx2):
            return eri_unitary_deriv2(idx, didx1+didx2, notation='c')

        deriv_dict1 = {
                'v': eri1_wrapper
                }
        deriv_dict2 = {
                'v': eri2_wrapper
                }
    return deriv_dict1, deriv_dict2

def implicit_derivative(expr: e.Expr, order: int, deriv_idx: [Index], deriv_dict1: dict, 
                        deriv_dict2: dict = None) -> e.Expr:
    if order == 0:
        return expr
    if order == 1:
        return e.Expr(_implicit_deriv1(expr.sympy, deriv_idx, deriv_dict1))
    if order == 2:
        return e.Expr(_implicit_deriv2(expr.sympy, deriv_idx[0], deriv_idx[1], deriv_dict1,
                      deriv_dict2))
    raise RuntimeError(f'Only first and second derivatives supported, not of order {order}')

def _implicit_deriv1(expr, deriv_idx: [Index], deriv_dict: dict) -> e.Expr:
    # Sum rule
    if isinstance(expr, Add):
        # f = a + b
        # f' = a' + b'
        d = Add(*[_implicit_deriv1(a, deriv_idx, deriv_dict) for a in expr.args])
        #print("After Addition")
        #print(expr)
        #print(d)
        #print()
        return d
    # Product rule
    if isinstance(expr, Mul):
        # f = a * b
        # f' = a' * b + a * b'
        new_terms = []
        for i, a in enumerate(expr.args):
            adiff = _implicit_deriv1(a, deriv_idx, deriv_dict)
            pterms = [b for j, b in enumerate(expr.args) if j != i]
            pterms.insert(i, adiff)
            new_terms.append(Mul(*pterms))
        d = Add(*new_terms)
        #print("After Multiplication")
        #print(expr)
        #print(d)
        #print()
        return d

    # Polynomials
    if isinstance(expr, Pow):
        # f = a^n
        # f' = n * a^(n-1) * a'
        new_expr = Pow(expr.base, expr.exp-1) * expr.exp
        d = new_expr * _implicit_deriv1(expr.base, deriv_idx, deriv_dict)
        #print("After Powering:")
        #print(expr)
        #print(d)
        #print()
        return d

    # Direct derivative
    if hasattr(expr, 'symbol'):
        if str(expr.symbol) in deriv_dict:
            #print("Direct derivative ")
            idx = None
            if hasattr(expr, 'indices'):
                idx = expr.indices
            elif hasattr(expr, 'idx'):
                idx = expr.idx
            elif hasattr(expr, 'upper') and hasattr(expr, 'lower'):
                idx = expr.upper + expr.lower
            d = deriv_dict[str(expr.symbol)](idx, deriv_idx)
            #print(expr)
            #print(d)
            #print()
            return d
    return 0

def _implicit_deriv2(expr, deriv_idx1: [Index], deriv_idx2: [Index], deriv_dict1: dict,
                     deriv_dict2: dict) -> e.Expr:
    
    # Sum rule
    if isinstance(expr, Add):
        # f = a + b
        # f'' = a'' + b''
        return Add(*[_implicit_deriv2(a, deriv_idx1, deriv_idx2, deriv_dict1, deriv_dict2) 
                     for a in expr.args])

    # Product rule
    if isinstance(expr, Mul):
        # f = a * b
        # f'' = a'' * b + a' * b' + a' * b' + a * b''
        #     = a'' * b + 2 * a' * b' + a * b''
        new_terms = []
        for i, a in enumerate(expr.args):
            for j, b in enumerate(expr.args):
                pterms = [c for k, c in enumerate(expr.args) if k not in {i, j}]
                if i == j:
                    diffterm = _implicit_deriv2(a, deriv_idx1, deriv_idx2, deriv_dict1,
                                                deriv_dict2)
                    pterms.insert(i, diffterm)
                else:
                    adiff = _implicit_deriv1(a, deriv_idx1, deriv_dict1)
                    bdiff = _implicit_deriv1(b, deriv_idx2, deriv_dict1)
                    pterms.insert(i, adiff)
                    pterms.insert(j, bdiff)
                new_terms.append(Mul(*pterms))
        return Add(*new_terms)

    # Polynomials
    if isinstance(expr, Pow):
        # f = a^n
        # f'' = n * (n-1) * a^(n-2) * (a')^2 + n * a^(n-1) * a''
        # ONLY FOR n != 1 (and 0 technically but that is not a Pow object)
        # For n = 1 the first term vanishes
        new_term = expr.exp * Pow(expr.base, expr.exp-1)
        new_term *= _implicit_deriv2(expr.base, deriv_idx1, deriv_idx2, deriv_dict1, deriv_dict2)
        if expr.exp != 1:
            nt2 = expr.exp * (expr.exp-1) * Pow(expr.base, expr.exp-2)
            nt2 *= _implicit_deriv1(expr.base, deriv_idx1, deriv_dict1)
            nt2 *= _implicit_deriv1(expr.base, deriv_idx2, deriv_dict1)
            new_term += nt2
        return new_term

    # Direct derivative
    if hasattr(expr, 'symbol'):
        s = str(expr.symbol)
        if s in deriv_dict2:
            idx = None
            if hasattr(expr, 'indices'):
                idx = expr.indices
            elif hasattr(expr, 'idx'):
                idx = expr.idx
            elif hasattr(expr, 'upper') and hasattr(expr, 'lower'):
                idx = expr.upper + expr.lower
            return deriv_dict2[s](idx, deriv_idx1, deriv_idx2)
    return 0


