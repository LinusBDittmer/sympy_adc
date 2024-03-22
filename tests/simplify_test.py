from adcgen.simplify import simplify_unitary
from adcgen.sympy_objects import NonSymmetricTensor, AntiSymmetricTensor, \
    KroneckerDelta
from sympy_adc.expr_container import Expr
from sympy_adc.indices import get_symbols
from sympy_adc.func import evaluate_deltas

from sympy import S


class TestSimplify:
    def test_simplify_unitary(self):
        i, j, k = get_symbols('ijk')
        a, b, c = get_symbols('abc')
        p, q, r = get_symbols('pqr')

        # trivial positive: non-symmetric
        expr = Expr(
            NonSymmetricTensor('U', (i, j)) * NonSymmetricTensor('U', (k, j))
        )
        res = KroneckerDelta(i, k)
        assert simplify_unitary(expr, t_name='U', block_diagonal=True).sympy == res
        # trivial positive: anti-symmetric
        expr = Expr(AntiSymmetricTensor('A', (i,), (j,))
                    * AntiSymmetricTensor('A', (k,), (j,)))
        assert simplify_unitary(expr, t_name='A', block_diagonal=True).sympy - res is S.Zero

        # with remainder:
        expr = Expr(
            NonSymmetricTensor('U', (i, j)) * NonSymmetricTensor('U', (k, j))
            * NonSymmetricTensor('U', (a, b))
        )
        res = NonSymmetricTensor('U', (a, b)) * KroneckerDelta(i, k)
        assert simplify_unitary(expr, t_name='U', block_diagonal=True).sympy - res is S.Zero

        # U_ij U_ik U_ab U_ac = delta_jk * delta_bc
        expr = Expr(
            NonSymmetricTensor('U', (i, j)) * NonSymmetricTensor('U', (i, k))
            * NonSymmetricTensor('U', (a, b)) * NonSymmetricTensor('U', (a, c))
        )
        res = KroneckerDelta(j, k) * KroneckerDelta(b, c)
        assert simplify_unitary(expr, t_name='U', block_diagonal=True).sympy - res is S.Zero

        # switch index positions
        expr = Expr(
            NonSymmetricTensor('U', (j, i)) * NonSymmetricTensor('U', (k, i))
            * NonSymmetricTensor('U', (a, b)) * NonSymmetricTensor('U', (a, c))
        )
        assert simplify_unitary(expr, t_name='U', block_diagonal=True).sympy - res is S.Zero

        # index occurs at 3 objects
        expr = Expr(
            NonSymmetricTensor('U', (i, j)) * NonSymmetricTensor('U', (i, k))
            * NonSymmetricTensor('V', (i, j, k))
        )
        assert (simplify_unitary(expr, t_name='U', block_diagonal=True) - expr).sympy is S.Zero

        # exponent > 1 and multiple occurences of index
        expr = Expr(
            NonSymmetricTensor('U', (i, j)) * NonSymmetricTensor('U', (i, j))
            * NonSymmetricTensor('U', (i, k))
        )
        res = NonSymmetricTensor('U', (i, k))
        assert simplify_unitary(expr, t_name='U', block_diagonal=True).sympy - res is S.Zero

        # Simplify over left index and occupied space
        expr = Expr(
            NonSymmetricTensor('U', (i, p)) * NonSymmetricTensor('U', (i, q))
            * NonSymmetricTensor('A', (p, q))
        )
        expr = simplify_unitary(expr, t_name='U', block_diagonal=True)
        expr = evaluate_deltas(expr.sympy)
        assert isinstance(expr, NonSymmetricTensor)
        assert expr.indices[0] == expr.indices[1] and expr.indices[0].space[0] == 'o'

        # Simplify over left index and virtual space
        expr = Expr(
            NonSymmetricTensor('U', (a, p)) * NonSymmetricTensor('U', (a, q))
            * NonSymmetricTensor('A', (p, q))
        )
        expr = simplify_unitary(expr, t_name='U', block_diagonal=True)
        expr = evaluate_deltas(expr.sympy)
        assert isinstance(expr, NonSymmetricTensor)
        assert expr.indices[0] == expr.indices[1] and expr.indices[0].space[0] == 'v'

        # Simplify over right index and occupied space
        expr = Expr(
            NonSymmetricTensor('U', (p, i)) * NonSymmetricTensor('U', (q, i))
            * NonSymmetricTensor('A', (p, q))
        )
        expr = simplify_unitary(expr, t_name='U', block_diagonal=True)
        expr = evaluate_deltas(expr.sympy)
        assert isinstance(expr, NonSymmetricTensor)
        assert expr.indices[0] == expr.indices[1] and expr.indices[0].space[0] == 'o'

        # Simplify over right index and virtual space
        expr = Expr(
            NonSymmetricTensor('U', (p, a)) * NonSymmetricTensor('U', (q, a))
            * NonSymmetricTensor('A', (p, q))
        )
        expr = simplify_unitary(expr, t_name='U', block_diagonal=True)
        expr = evaluate_deltas(expr.sympy)
        assert isinstance(expr, NonSymmetricTensor)
        assert expr.indices[0] == expr.indices[1] and expr.indices[0].space[0] == 'v'
