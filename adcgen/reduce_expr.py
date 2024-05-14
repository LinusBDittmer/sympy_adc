from . import expr_container as e
from .eri_orbenergy import EriOrbenergy
from .misc import Inputerror
from collections import defaultdict
from sympy import S
import time


def reduce_expr(expr):
    """Function that reduces the number of terms in an expression as much as
       possible by expanding all available intermediates and simplifying the
       resulting expression as much as possible by canceling orbital energy
       fractions."""
    from itertools import chain

    if not isinstance(expr, e.Expr):
        raise Inputerror(f"Expr to reduce needs to be an instance of {e.Expr}")
    if not expr.real:
        raise NotImplementedError("Intermediates only implemented for a real "
                                  "orbital basis.")
    expr = expr.expand()

    # check if we have anything to do
    if expr.sympy.is_number:
        return expr

    print('\n', '#'*80, '\n', ' '*25, "REDUCING EXPRESSION\n", '#'*80, '\n',
          sep='')

    # 1) Insert the definitions of all defined intermediates in the expr
    #    and reduce the number of terms by factoring the ERI in each term.
    start = time.perf_counter()
    print("Expanding intermediates... ")
    expanded_expr: list[e.Expr] = []
    for term_i, term in enumerate(expr.terms):
        print('#'*80)
        print(f"Expanding term {term_i+1} of {len(expr)}: {term}... ", end='',
              flush=True)
        term = term.expand_intermediates().expand()
        print(f"into {len(term)} terms.\nCollecting terms.... ")
        term = factor_eri_parts(term)
        print('-'*80)
        for j, equal_eri in enumerate(term):
            # minimize the contracted indices
            # each term in eri should hold exactly the same indices
            # -> build substitutions once and apply to the whole expr
            sub = equal_eri.terms[0].substitute_contracted(only_build_sub=True)
            sub_equal_eri = equal_eri.subs(sub)
            # ensure that we are not creating a complete mess
            if sub_equal_eri.sympy is S.Zero and equal_eri.sympy is not S.Zero:
                raise ValueError(f"Invalid substitutions {sub} for "
                                 f"{equal_eri}")
            term[j] = sub_equal_eri
            print(f"\n{j+1}: {EriOrbenergy(sub_equal_eri.terms[0]).eri}")
        print('-'*80)
        print(f"Found {len(term)} different ERI Structures")
        expanded_expr.extend(term)
    del expr
    # 2) Now try to factor the whole expression
    #    Only necessary to consider the first term of each of the expressions
    #    in the list (they all have same ERI)
    #    -> build new term list and try to factor ERI + Denominator
    #    -> simplify the orbital energy fraction in the resulting terms
    print("\nExpanding and ERI factoring took "
          f"{time.perf_counter() - start:.2f}s\n")
    print('#'*80, "\n", '#'*80, sep='')
    start = time.perf_counter()
    print("\nSumming up all terms:")
    print('#'*80)
    unique_terms = [unique_expr.terms[0] for unique_expr in expanded_expr]
    print("Factoring ERI...")
    unique_compatible_eri = find_compatible_eri_parts(unique_terms)
    n = 1
    n_eri_denom = 0
    factored: e.Expr = 0
    # - factor eri again
    for i, compatible_eri_subs in unique_compatible_eri.items():
        temp = expanded_expr[i]
        eri = EriOrbenergy(expanded_expr[i].terms[0]).eri
        print("\n", '#'*80, sep='')
        print(f"ERI {n} of {len(unique_compatible_eri)}: {eri}")
        n += 1
        for other_i, sub in compatible_eri_subs.items():
            temp += expanded_expr[other_i].subs(sub)

        # collected all terms with equal ERI -> factor denominators
        eri_sym = eri.symmetry(only_contracted=True)
        print("\nFactoring Denominators...")
        for j, term in enumerate(factor_denom(temp, eri_sym=eri_sym)):
            term = term.factor()
            if len(term) != 1:
                raise RuntimeError("Expected the sub expression to have "
                                   "identical Denoms and ERI, which should "
                                   "allow factorization to a single term:\n"
                                   f"{term}")
            # symmetrize the numerator and cancel the orbital energy fraction
            term = EriOrbenergy(term)
            print('-'*80)
            print(f"ERI/Denom {j}: {term}\n")
            print("Permuting numerator... ", flush=True, end='')
            term = term.permute_num(eri_sym=eri_sym)
            print(f"Done:\n{term}")
            print("\nCancel orbital energy fraction...")
            term = term.cancel_orb_energy_frac()
            print("Done.")

            if not all(EriOrbenergy(t).num.sympy.is_number
                       for t in term.terms):
                print("\nNUMERATOR NOT CANCELLED COMPLETELY:")
                for t in term.terms:
                    print(EriOrbenergy(t))

            factored += term
        n_eri_denom += j + 1
    del expanded_expr  # not up to date anymore
    print('#'*80)
    print("\nFactorizing and cancelling the orbital energy fractions in "
          f"{n_eri_denom} terms took {time.perf_counter() - start:.2f}s.\n"
          f"Expression consists now of {len(factored)} terms.")
    print('#'*80)

    # 3) Since we modified some denominators by canceling the orbital energy
    #    fractions, try to factor eri and denominator again
    print("\nFactoring again...", flush=True, end='')
    result = 0
    for term in chain.from_iterable(factor_denom(sub_expr) for sub_expr in
                                    factor_eri_parts(factored)):
        # factor the resulting term again, because we can have something like
        # 2/(4*a + 4*b) * X - 1/(2 * (a + b)) * X
        result += term.factor()
    print(f"Done. {len(result)} terms remaining.\n")
    print('#'*80)
    return result


def factor_eri_parts(expr: e.Expr) -> list[e.Expr]:
    """Factors the eri's of an expression."""

    if len(expr) == 1:  # trivial case
        return [expr]

    terms = expr.terms
    ret: list[e.Expr] = []
    for i, compatible_eri_subs in find_compatible_eri_parts(terms).items():
        temp = e.Expr(terms[i].sympy, **expr.assumptions)
        for other_i, sub in compatible_eri_subs.items():
            temp += terms[other_i].subs(sub)
        ret.append(temp)
    return ret


def find_compatible_eri_parts(term_list: list[e.Term]) -> dict[int, dict]:
    """Determines the necessary substitutions to make the ERI parts of terms
       identical to each other - so they can be factored easily.
       Does not modify any of the terms, but returns a dict that connects
       the indices of the terms with a substitution list."""
    from .simplify import find_compatible_terms

    if len(term_list) == 1:  # trivial: only a single eri
        return {0: {}}

    # dont use EriOrbenergy class, but rather only do whats necessary to
    # extract the eri part of the terms
    eri_parts: list[e.Term] = []
    for term in term_list:
        assumptions = term.assumptions
        assumptions['target_idx'] = term.target
        eris = e.Expr(1, **assumptions)
        for o in term.objects:
            if not o.sympy.is_number and not o.contains_only_orb_energies:
                eris *= o
        eri_parts.append(eris.terms[0])

    return find_compatible_terms(eri_parts)


def factor_denom(expr: e.Expr, eri_sym: dict = None) -> list[e.Expr]:
    """Factor the orbital energy denominators of an expr."""

    if len(expr) == 1:  # trivial case: single term
        return [expr]

    terms: tuple[e.Term] = expr.terms
    compatible_denoms = find_compatible_denom(terms, eri_sym=eri_sym)
    ret: list[e.Expr] = []
    for i, compatible_denom_perms in compatible_denoms.items():
        temp = e.Expr(terms[i].sympy, **expr.assumptions)
        for other_i, perms in compatible_denom_perms.items():
            temp += terms[other_i].permute(*perms)
        ret.append(temp)
    return ret


def find_compatible_denom(terms: list[e.Term],
                          eri_sym: dict = None) -> dict[int, dict]:
    if len(terms) == 1:  # trivial case: single term
        return {0: {}}

    terms: list[EriOrbenergy] = [
        EriOrbenergy(term).canonicalize_sign(only_denom=True)
        for term in terms
    ]

    # split the terms according to length and and number of denominator
    # brackets
    filtered_terms = defaultdict(list)
    for term_i, term in enumerate(terms):
        filtered_terms[term.denom_description()].append(term_i)

    ret = {}
    matched = set()
    permutations = {}
    for term_idx_list in filtered_terms.values():
        # check which denominators are already equal
        identical_denom = {}
        for i, term_i in enumerate(term_idx_list):
            if term_i in matched:
                continue
            term: EriOrbenergy = terms[term_i]
            identical_denom[term_i] = []
            for other_i in range(i+1, len(term_idx_list)):
                other_term_i = term_idx_list[other_i]
                if other_term_i in matched:
                    continue
                if term.denom.sympy == terms[other_term_i].denom.sympy:
                    identical_denom[term_i].append(other_term_i)
                    matched.add(other_term_i)

        if len(identical_denom) == 1:  # all denoms are equal
            term_i, matches = identical_denom.popitem()
            ret[term_i] = {other_term_i: tuple() for other_term_i in matches}
            continue

        # try to match more denominators by applying index permutations that
        # satisfy:  P_pq ERI = +- ERI  AND  P_pq Denom != +- Denom
        identical_denom = list(identical_denom.items())
        for i, (term_i, matches) in enumerate(identical_denom):
            if term_i in matched:
                continue
            ret[term_i] = {}
            for other_term_i in matches:  # add all identical denominators
                ret[term_i][other_term_i] = []

            denom = terms[term_i].denom.sympy
            for other_i in range(i+1, len(identical_denom)):
                other_term_i, other_matches = identical_denom[other_i]
                if other_term_i in matched:
                    continue

                other_term: EriOrbenergy = terms[other_term_i]
                other_denom: e.Expr = other_term.denom

                # find all valid permutations
                if other_term_i not in permutations:
                    permutations[other_term_i] = tuple(
                        perms for perms, factor in
                        other_term.denom_eri_sym(eri_sym=eri_sym,
                                                 only_contracted=True).items()
                        if factor is None
                    )
                for perms in permutations[other_term_i]:
                    # found a permutation!
                    if denom == other_denom.copy().permute(*perms).sympy:
                        ret[term_i][other_term_i] = perms
                        for match in other_matches:
                            ret[term_i][match] = perms
                        matched.add(other_term_i)
                        break
    return ret
