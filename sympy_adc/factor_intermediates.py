from . import expr_container as e
from .misc import Inputerror, cached_property
from .eri_orbenergy import EriOrbenergy
from .indices import (order_substitutions, get_symbols, index_space,
                      minimize_tensor_indices)
from .sympy_objects import AntiSymmetricTensor
from .symmetry import LazyTermMap
from sympy import S, Mul, Rational
from collections import Counter, defaultdict


def factor_intermediates(expr, types_or_names: str | list[str] = None,
                         max_order: int = None) -> e.Expr:
    from .intermediates import Intermediates
    from time import perf_counter

    if not isinstance(expr, e.Expr):
        raise Inputerror("The expression to factor needs to be provided "
                         f"as {e.Expr} instance.")

    if expr.sympy.is_number:  # nothing to factor
        return expr

    # get all intermediates that are about to be factored in the expr
    itmd = Intermediates()
    if types_or_names is not None:
        if isinstance(types_or_names, str):
            itmd_to_factor = getattr(itmd, types_or_names)
        else:  # list / tuple / set of strings
            itmd_to_factor = {}
            for t_or_n in types_or_names:
                if not isinstance(t_or_n, str):
                    raise TypeError("Intermediate types/names to factor have "
                                    "to be provided as str or list of strings."
                                    f"Got {t_or_n} of type {type(t_or_n)}.")
                itmd_to_factor |= getattr(itmd, t_or_n)
    else:
        itmd_to_factor: dict = itmd.available

    print('\n\n', '#'*80, '\n', " "*25, "INTERMEDIATE FACTORIZATION\n", '#'*80,
          "\n", sep='')
    print(f"Trying to factor intermediates in expr of length {len(expr)}\n")
    for i, term in enumerate(expr.terms):
        print(f"{i+1}:  {EriOrbenergy(term)}\n")
    print('#'*80)
    # try to factor all requested intermediates
    factored = []
    for name, itmd_cls in itmd_to_factor.items():
        print("\n", ' '*25, f"Factoring {name}\n\n", '#'*80, sep='')
        start = perf_counter()
        expr = itmd_cls.factor_itmd(expr, factored, max_order)
        factored.append(name)
        print('\n', '-'*80, sep='')
        print(f"Done in {perf_counter()-start:.2f}s. {len(expr)} terms remain")
        print('-'*80, '\n')
        for i, term in enumerate(expr.terms):
            print(f"{i+1: >{len(str(len(expr)+1))}}:  {EriOrbenergy(term)}\n")
        print('#'*80)
    print("\n\n", '#'*80, "\n", " "*25,
          "INTERMEDIATE FACTORIZATION FINISHED\n", '#'*80, sep='')
    # make the result pretty by minimizing contracted indices:
    # some contracted indices might be hidden inside some intermediates.
    # -> ensure that the remaining ones are the lowest available
    expr = expr.substitute_contracted()
    print(f"\n{len(expr)} terms in the final result:")
    width = len(str(len(expr)+1))
    for i, term in enumerate(expr.terms):
        print(f"{i+1: >{width}}: {EriOrbenergy(term)}")
    return expr


def _factor_long_intermediate(expr: e.Expr, itmd: list[EriOrbenergy],
                              itmd_data: tuple, itmd_term_map: LazyTermMap,
                              itmd_cls) -> e.Expr:
    """Function for factoring a long intermediate, i.e., a intermediate that
       consists of more than one term."""

    if expr.sympy.is_number:
        return expr

    # does any itmd term has a denominator?
    itmd_has_denom = any(term_data.denom_bracket_lengths is not None
                         for term_data in itmd_data)
    itmd_length = len(itmd)
    # get the default symbols of the intermediate
    itmd_default_symbols = tuple(get_symbols(itmd_cls.default_idx))

    terms: list[EriOrbenergy] = list(expr.terms)

    # create a dict where all the found assignments are collected
    # {itmd_indices: {remainder: [(pref, [term_i,])]}}
    found_intermediates = defaultdict(dict)
    for term_i, term in enumerate(terms):
        term = EriOrbenergy(term).canonicalize_sign()
        # prescan: check that the term holds the correct tensors and
        #          denominator brackets
        term_data = FactorizationTermData(term)
        # description of all objects in the eri part, exponent implicitly
        # included
        obj_descr = term_data.eri_obj_descriptions
        if itmd_has_denom:
            bracket_lengths = term_data.denom_bracket_lengths

        # compare to all of the itmd terms -> only try to map on a subset of
        # intermediate terms later
        possible_matches = []
        for itmd_i, itmd_term_data in enumerate(itmd_data):
            # do all tensors in the eri part occur at least as often as
            # in the intermediate
            if any(obj_descr[descr] < n for descr, n in
                   itmd_term_data.eri_obj_descriptions.items()):
                continue
            # itmd_term has a denominator?
            itmd_bracket_lengths = itmd_term_data.denom_bracket_lengths
            if itmd_bracket_lengths is not None:
                if bracket_lengths is None:  # term has no denom -> cant match
                    continue
                else:  # term also has a denominator
                    # ensure that bracket of the correct length are available
                    if any(bracket_lengths[length] < n for length, n in
                           itmd_bracket_lengths.items()):
                        continue
            possible_matches.append(itmd_i)
        if not possible_matches:  # did not find any possible matches
            continue

        # extract the target idx names of the term
        target_idx_by_space = {}
        for s in term.eri.target:
            if (sp := index_space(s.name)) not in target_idx_by_space:
                target_idx_by_space[sp] = set()
            target_idx_by_space[sp].add(s.name)

        # go through all possible matches
        for itmd_i in possible_matches:
            # - compare and obtain data (sub_dict, obj indices, factor)
            #   that makes the itmd_term equal to the defined sub part
            #   of the term.
            variants = _compare_terms(term, itmd[itmd_i], term_data=term_data,
                                      itmd_term_data=itmd_data[itmd_i])
            if variants is None:  # was not possible to map the terms
                continue

            # The term_map allows to spread a term assignement to multiple
            # terms taking the symmetry of the remainder into account, e.g.,
            # for the t2_2 amplitudes:
            #    t2_2 <- (1-P_ij)(1-P_ab) X
            #  - Depending on the symmetry of the remainder these 4 terms
            #    might occur as 4, 2 or 1 terms in the expression to factor:
            #     Rem * (1-P_ij)(1-P_ab) X -> 4 * Rem * X
            #    (if Rem has ij and ab antisymmetry)
            #  - If a term with such a remainder is matched with one of the 4
            #    4 terms he will automatically also be matched with the other
            #    3 terms using the term_map for the intermediate.
            # NOTE: it is not possible to exploit this to reduce the workload
            #       by exploiting the fact that the current term has already
            #       been matched to a itmd_term through the term map, because
            #       more complicated permutations do not provide a back and
            #       forth relation ship between terms:
            #         P_ij P_ik A(ijk) -> B(kij)
            #         P_ij P_ik B(kij) -> C(jki)
            #       comparing the current term to A can also provide a match
            #       with B through the term_map.
            #       comparing the current term to B however can provide a match
            #       with C!
            #       Therefore, the comparison with B can not be skipped, even
            #       if remainder and itmd_indices are identical to a previously
            #       found variant that matched to A and B!
            # What can be done: for matching term 1 -> itmd_term A
            # due to the symmetry of tensors one probably obtains multiple
            # variants for the same itmd_indices and remainder that only
            # differ in contracted indices.
            # -> for each itmd_indices only consider one variant for each
            #    remainder
            found_remainders = {}  # {itmd_indices: [remainder]}
            for variant_data in variants:  # go through all valid variants
                # - extract the remainder of the term (objects, excluding
                #   prefactors that will remain if the current variant is
                #   used to factor the itmd)

                remainder = _get_remainder(term, variant_data['eri_i'],
                                           variant_data['denom_i'])

                # - obtain the indices of the intermediate
                itmd_indices = tuple(variant_data['sub'].get(s, s) for s in
                                     itmd_default_symbols)

                # - minimize the indices of the intermediate to ensure that
                #   the same indices are used in each term of the long itmd
                #   (use the lowest non target indices)
                itmd_indices, minimization_perms = minimize_tensor_indices(
                    itmd_indices, target_idx_by_space
                )

                # - apply the substitutions to the remainder
                remainder = remainder.permute(*minimization_perms)
                # if this ever triggers probably switch to a continue
                assert remainder.sympy is not S.Zero

                # - Further minimize the tensor indices taking the tensor
                #   symmetry of the itmd into account by building a tensor
                #   using the minimized tensor indices
                #   -> returns the tensor with completely minimized indices
                #      and possibly a factor of -1
                tensor_obj = itmd_cls.tensor(indices=itmd_indices).terms[0]
                if len(tensor_obj) > 2:
                    raise ValueError("Expected the term to be at most of "
                                     f"length 2. Got: {tensor_obj}.")
                for obj in tensor_obj.objects:
                    if 'tensor' in (o_type := obj.type):
                        itmd_indices = obj.idx
                    elif o_type == 'prefactor':
                        variant_data['factor'] *= obj.sympy
                    else:
                        raise TypeError("Only expected tensor and prefactor."
                                        f"Found {obj} in {tensor_obj}")

                # ensure that there are no indices in the numerator or the
                # denominator of the remainder, that do not occur in the eri
                # part or the itmd indices. This avoids factoring an
                # itmd using invalid contracted itmd indices that are only
                # partially removed from the term, e.g., m - a contracted itmd
                # index occurs in both, denominator and eri part, but is only
                # removed in the eri part, because the itmd has no denominator.
                _validate_indices(remainder, itmd_indices)

                # check if we already found another variant that gives the
                # same itmd_indices and remainder (an identical result that
                # only might differ in contracted itmd_indices)
                if itmd_indices not in found_remainders:
                    found_remainders[itmd_indices] = []
                if any(_compare_remainder(remainder, found_rem, itmd_indices)
                       is not None
                       for found_rem in found_remainders[itmd_indices]):
                    continue  # go to the next variant

                # - check if the current itmd_term can be mapped onto other
                #   itmd terms
                matching_itmd_terms = _map_on_other_terms(
                    itmd_i, remainder, itmd_term_map, itmd_indices,
                    itmd_default_symbols
                )

                # - calculate the final prefactor of the remainder if the
                #   current variant is applied for factorization
                #   keep the term normalized if spreading to multiple terms!
                #   (factor is +-1)
                prefactor = (term.pref * variant_data['factor'] *
                             Rational(1, len(matching_itmd_terms)) /
                             itmd[itmd_i].pref)

                # - compute the factor that the term should have if we want
                #   to factor the current variant with a prefactor of 1
                #   (required for factoring mixed prefactors)
                unit_factorization_pref = (
                    itmd[itmd_i].pref * variant_data['factor']
                    * len(matching_itmd_terms)
                )

                # - assign the found match to an intermediate
                #   (try to build complete intermediates)
                assigned = False
                matching_remainder = None
                for rem, itmd_list in found_intermediates[itmd_indices].items():  # noqa E501
                    # check if the remainders are compatible
                    factor = _compare_remainder(remainder=remainder,
                                                ref_remainder=rem,
                                                itmd_indices=itmd_indices)
                    if factor is None:  # not compatible
                        continue

                    matching_remainder = rem

                    # add the factor to the prefactor: might be another -1
                    prefactor *= factor
                    # iterator over all previously initialized intermediates
                    # and try to add to one of them
                    for pref, term_list, unit_factors in itmd_list:
                        # check if:
                        #  - the current term is already assigned to
                        #    some itmd term (cant assign the same term
                        #    multiple times)
                        #  - the itmd terms we want to map the term on
                        #    are still available
                        #  - the prefactors are identical
                        if any(term_list[itmd_j] is not None for itmd_j in
                               matching_itmd_terms) or \
                                term_i in term_list or pref != prefactor:
                            continue
                        # itmd_indices, remainder and prefactor are equal
                        # -> can add
                        for itmd_j in matching_itmd_terms:
                            term_list[itmd_j] = term_i
                        # add the factor which the term should have to factor
                        # the current intermediate variant with a prefactor
                        # of 1
                        unit_factors[term_i] = unit_factorization_pref
                        assigned = True
                        break
                    # either was possible to add to an intermediate or not
                    # for the given itmd indices and remainder
                    break
                if not assigned:  # build a new itmd
                    if matching_remainder is None:  # new remainder
                        found_intermediates[itmd_indices][remainder] = []
                    else:  # remainder is equal to another already found one
                        remainder = matching_remainder
                    found_intermediates[itmd_indices][remainder].append(
                        (prefactor,
                         [term_i if itmd_j in matching_itmd_terms else None
                          for itmd_j in range(itmd_length)],
                         {term_i: unit_factorization_pref})
                    )
    print("\nDONE COLLECTING INTERMEDIATES:")
    print(found_intermediates)

    # iterate through the found intermediates and try to factor the itmd
    # -> first only the onces that are complete
    factored: e.Expr = 0
    factored_successfully = False  # whether we factored the itmd at least once
    factored_terms = set()  # keep track which terms have already been factored
    for itmd_indices, remainders in found_intermediates.items():
        for rem, itmd_list in remainders.items():
            # new nested list of itmd that were not complete or
            # already used to factor another itmd
            remaining_itmds = []
            for pref, term_list, unit_factors in itmd_list:
                # only look for complete intermediates with all terms available
                # -> all itmd_terms found and all terms not already factored
                if any(term_i is None or term_i in factored_terms
                       for term_i in term_list):
                    remaining_itmds.append((pref, term_list, unit_factors))
                    continue
                # Found complete intermediate -> factor
                print(f"\nFactoring {itmd_cls.name} in terms:")
                for term_i in term_list:
                    print(EriOrbenergy(terms[term_i]))

                new_term = _build_factored_term(rem, pref, itmd_cls,
                                                itmd_indices)
                print(f"result:\n{EriOrbenergy(new_term)}")

                factored += new_term
                factored_terms.update(term_list)
                factored_successfully = True
            # add a list of all intermediates that were not all terms
            # were available to factor the intermediate
            if not remaining_itmds:
                remaining_itmds = None
            found_intermediates[itmd_indices][rem] = remaining_itmds

    # go again through the remaining itmd variants and try to build more
    # complete variants by allowing mixed prefactors, i.e.,
    # add a term that belongs to a variant with prefactor 1
    # to the nearly complete variant with prefactor 2. To compensate
    # for this, additional terms are added to the result.
    factored, factored_mixed_pref_successfully = _factor_mixed_prefactors(
        factored, terms, found_intermediates, factored_terms, itmd_cls
    )
    factored_successfully |= factored_mixed_pref_successfully

    # TODO:
    # go again through the remaining itmds and see if we can factor another
    # intermediate by filling up some terms, e.g. if we found 5 out of 6 terms
    # it still makes sense to factor the itmd

    # Currently everything is just added to the result
    for term_i, term in enumerate(terms):
        if term_i not in factored_terms:
            factored_terms.add(term_i)
            factored += term

    # if we factored the itmd successfully it might be necessary to adjust
    # sym_tensors or antisym_tensors of the returned expression
    if factored_successfully:
        tensor = itmd_cls.tensor(return_sympy=True)
        if isinstance(tensor, AntiSymmetricTensor):
            name = tensor.symbol.name
            if tensor.bra_ket_sym is S.One and \
                    name not in (sym_tensors := factored.sym_tensors):
                factored.set_sym_tensors(sym_tensors + (name,))
            elif tensor.bra_ket_sym is S.NegativeOne and \
                    name not in (antisym_t := factored.antisym_tensors):
                factored.set_antisym_tensors(antisym_t + (name,))
    return factored


def _factor_short_intermediate(expr: e.Expr, itmd: EriOrbenergy,
                               itmd_data, itmd_cls) -> e.Expr:
    """Function for factoring a short intermediate, i.e., an intermediate that
       consists of a single term."""

    if expr.sympy.is_number:
        return expr

    # get the default symbols of the intermediate
    itmd_default_symbols = tuple(get_symbols(itmd_cls.default_idx))

    terms = expr.terms

    factored: e.Expr = 0  # factored expression that is returned
    factored_sucessfully = False  # bool to indicate whether we factored
    for term in terms:
        term = EriOrbenergy(term).canonicalize_sign()
        data = FactorizationTermData(term)
        # check if the current term and the itmd are compatible:
        #  - check if all necessary objects occur in the eri part
        obj_descr = data.eri_obj_descriptions
        if any(obj_descr[descr] < n for descr, n in
               itmd_data.eri_obj_descriptions.items()):
            factored += term.expr
            continue
        # - check if brackets of the correct length occur in the denominator
        if itmd_data.denom_bracket_lengths is not None:  # itmd has a denom
            bracket_lengths = data.denom_bracket_lengths
            if bracket_lengths is None:  # term has no denom
                factored += term.expr
                continue
            else:  # term also has a denom
                if any(bracket_lengths[length] < n for length, n in
                       itmd_data.denom_bracket_lengths.items()):
                    factored += term.expr
                    continue
        # ok, the term seems to be a possible match -> try to factor

        # compare the term and the itmd term
        variants = _compare_terms(term, itmd, data, itmd_data)

        if variants is None:
            factored += term.expr
            continue

        # choose the variant with the lowest overlap to other variants
        #  - find all unique obj indices (eri and denom)
        #  - and determine all itmd_indices
        unique_obj_i = {}
        for var_idx, var in enumerate(variants):
            key = (tuple(sorted(set(var['eri_i']))),
                   tuple(sorted(set(var['denom_i']))))
            if key not in unique_obj_i:
                unique_obj_i[key] = []
            unique_obj_i[key].append(var_idx)

        if len(unique_obj_i) == 1:  # always the same objects in each variant
            _, rel_variant_indices = unique_obj_i.popitem()
            min_overlap = []
        else:
            # multiple different objects -> try to find the one with the
            # lowest overlap to the other variants (so that we can possibly
            # factor the itmd more than once)
            unique_obj_i = list(unique_obj_i.items())
            overlaps = []
            for i, (key, _) in enumerate(unique_obj_i):
                eri_i, denom_i = set(key[0]), set(key[1])
                # determine the intersection of the objects
                overlaps.append(sorted(
                    [len(eri_i & set(other_key[0])) +
                     len(denom_i & set(other_key[1]))
                     for other_i, (other_key, _) in enumerate(unique_obj_i)
                     if i != other_i]
                ))
            # get the idx of the unique_obj_i with minimal intersections,
            # get the variant_data of the first element in the variant_idx_list
            min_overlap = min(overlaps)
            # collect all variant indices that have this overlap
            rel_variant_indices = []
            for overlap, (_, var_idx_list) in zip(overlaps, unique_obj_i):
                if overlap == min_overlap:
                    rel_variant_indices.extend(var_idx_list)
        # choose the variant with the minimal itmd_indices
        variant_data = min(
            [variants[var_idx] for var_idx in rel_variant_indices],
            key=lambda var: [var['sub'].get(s, s).name for s in
                             itmd_default_symbols]
        )

        # now start with factoring
        # - extract the remainder that survives the factorization (excluding
        #   the prefactor)
        remainder: e.Expr = _get_remainder(term, variant_data['eri_i'],
                                           variant_data['denom_i'])
        # - find the itmd indices:
        #   for short itmds it is not necessary to minimize the itmd indices
        #   just use whatever is found
        itmd_indices = tuple(variant_data['sub'].get(s, s) for s in
                             get_symbols(itmd_cls.default_idx))

        # ensure that there are no indices in the numerator or the
        # denominator of the remainder, that do not also occur in the eri
        # part or the itmd indices. This avoids factoring an
        # itmd using invalid contracted itmd indices that are only
        # partially removed from the term, e.g., m, a contracted itmd
        # index occurs in both, denominator and eri part, but is only
        # removed in the eri part, because the itmd has no denominator.
        _validate_indices(remainder, itmd_indices)

        # - determine the prefactor of the factored term
        pref = term.pref * variant_data['factor'] / itmd.pref
        # - check if it is possible to factor the itmd another time:
        #   should be possible if there is a 0 in the min_overlap list:
        #   -> Currently factoring a variant that has 0 overlap with another
        #      variant
        #   -> It should be possible to factor the intermediate in the
        #      remainder again!
        if 0 in min_overlap:
            # factor again and ensure that the factored result has the
            # the current assumptions
            remainder = e.Expr(
                _factor_short_intermediate(remainder, itmd, itmd_data,
                                           itmd_cls).sympy,
                **remainder.assumptions
            )
        # - build the new term including the itmd
        factored_term = _build_factored_term(remainder, pref, itmd_cls,
                                             itmd_indices)

        factored_sucessfully = True
        print(f"\nFactoring {itmd_cls.name} in:\n{term}\n"
              f"result:\n{EriOrbenergy(factored_term)}")
        factored += factored_term
    # if we factored the itmd sucessfully it might be necessary to add
    # the itmd tensor to the sym or antisym tensors
    if factored_sucessfully:
        tensor = itmd_cls.tensor(return_sympy=True)
        if isinstance(tensor, AntiSymmetricTensor):
            name = tensor.symbol.name
            if tensor.bra_ket_sym is S.One and \
                    name not in (sym_tensors := factored.sym_tensors):
                factored.set_sym_tensors(sym_tensors + (name,))
            elif tensor.bra_ket_sym is S.NegativeOne and \
                    name not in (antisym_t := factored.antisym_tensors):
                factored.set_antisym_tensors(antisym_t + (name,))
    return factored


def _factor_mixed_prefactors(factored: e.Expr, terms: list[e.Term],
                             found_intermediates: dict, factored_terms: set,
                             itmd_cls):
    """Tries to factor an intermediate by filling missing terms in a
       variant with terms that belong to other variants with different
       prefactors. Additional terms are added to the result to
       compensate for the adjustment of the prefactors."""

    factored_successfully = False
    for itmd_indices, remainders in found_intermediates.items():
        for rem, itmd_list in remainders.items():
            # possibly we don't have any itmd left -> None
            if itmd_list is None:
                continue
            n_variants = len(itmd_list)
            remaining_itmds = []
            # because of the term map a term can correspond to
            # multiple intermediate terms!
            # -> start with a given incomplete variant and try
            #    to fill the missing parts with other variants
            # -> term_i needs to be unique, but each term_i can
            #    occur multiple times
            for i, (pref, term_list, unit_factors) in enumerate(itmd_list):
                # update the term_list, because some terms might have already
                # been used to factor complete intermediates
                # also update the unit_factors
                term_list = [None if term_i is None or term_i in factored_terms
                             else term_i for term_i in term_list]
                unit_factors = {term_i: f for term_i, f in unit_factors.items()
                                if term_i in term_list}
                prefactors = [None if term_i is None else pref
                              for term_i in term_list]
                n_missing_terms = term_list.count(None)
                # skip empty intermediates, where no terms are available
                if n_missing_terms == len(term_list):
                    continue

                # loop over the upper triangle and try to fill the itmd
                for j in range(i, n_variants):
                    other_pref, other_term_list, other_unit_factors = itmd_list[j]
                    available = {}
                    for itmd_i, term_i in enumerate(other_term_list):
                        # check if the other term is still available
                        # and if we don't already use the term in the
                        # variant we want to complete
                        if term_i is None or term_i in term_list \
                                or term_i in factored_terms:
                            continue
                        if term_i not in available:
                            available[term_i] = []
                        available[term_i].append(itmd_i)

                    for term_i, itmd_i_list in available.items():
                        # ensure that all the positions are vacant
                        if any(term_list[itmd_i] is not None for itmd_i
                               in itmd_i_list):
                            continue
                        assert len(itmd_i_list) <= n_missing_terms
                        for itmd_i in itmd_i_list:  # fill the missing pos
                            term_list[itmd_i] = term_i
                            prefactors[itmd_i] = other_pref
                            n_missing_terms -= 1

                        unit_factors[term_i] = other_unit_factors[term_i]

                        # check if we are done and completed the itmd
                        if n_missing_terms == 0:
                            break
                    if n_missing_terms == 0:
                        break

                # could not fill up the intermediate with other prefactors
                # -> keep the filled up state so we may try to
                #    factor it as incomplete intermediate
                if n_missing_terms:
                    remaining_itmds.append(
                        (prefactors, term_list, unit_factors)
                    )
                    continue
                # found a complete intermediate with mixed prefactors!!

                # check how many terms share a common prefactor
                most_common_pref, count = sorted(
                    Counter(prefactors).items(), key=lambda x: x[1]
                )[-1]

                # if not enough terms share a common prefactor
                # -> don't factor! will not give a shorter expression
                if count < 0.6 * len(term_list):
                    continue

                terms_to_add = {}
                for p, term_i in zip(prefactors, term_list):
                    if p == most_common_pref or term_i in terms_to_add:
                        continue
                    terms_to_add[term_i] = p

                # for all terms that don't have the most common prefactor:
                # determine extension that needs to be added to the term
                # to factor the intermediate
                print("\nAdding terms:")
                for term_i, p in terms_to_add.items():
                    desired_pref = most_common_pref * unit_factors[term_i]
                    term = EriOrbenergy(terms[term_i]).canonicalize_sign()
                    extension_pref = term.pref - desired_pref
                    term = extension_pref * term.num * term.eri / term.denom
                    print(EriOrbenergy(term))
                    factored += term

                print(f"\nFactoring {itmd_cls.name} with mixed prefs in:")
                for term_i in term_list:
                    print(EriOrbenergy(terms[term_i]))

                # build the factored intermediate and add to the term
                new_term = _build_factored_term(rem, most_common_pref,
                                                itmd_cls, itmd_indices)
                print(f"result:\n{EriOrbenergy(new_term)}")
                factored += new_term

                factored_terms.update(term_list)
                factored_successfully = True
            # add a list of all filled up variants back to
            # found_intermediates so we can try to factor incomplete
            # variants later
            if not remaining_itmds:
                remaining_itmds = None
            found_intermediates[itmd_indices][rem] = remaining_itmds
    return factored, factored_successfully


def _build_factored_term(remainder: e.Expr, pref, itmd_cls,
                         itmd_indices) -> e.Expr:
    tensor = itmd_cls.tensor(indices=itmd_indices, return_sympy=True)
    # resolve the Zero placeholder for residuals
    if tensor.symbol.name == "Zero":
        return e.Expr(0, **remainder.assumptions)
    return remainder * pref * tensor


def _get_remainder(term: EriOrbenergy, obj_i: list[int],
                   denom_i: list[int]) -> e.Expr:
    """Returns the remainding part of the provided term that survives the
       factorization of the itmd, excluding the prefactor!
       Note that the returned remainder can still hold a prefactor of -1,
       because sympy is not maintaining the canonical sign in the denominator.
       """
    eri: e.Expr = term.cancel_eri_objects(obj_i)
    denom: e.Expr = term.cancel_denom_brackets(denom_i)
    rem = term.num * eri / denom
    # explicitly set the target indices, because the remainder not necessarily
    # has to contain all of them.
    if rem.provided_target_idx is None:  # no target indices set
        rem.set_target_idx(term.eri.target)
    return rem


def _validate_indices(remainder: e.Expr, itmd_indices: tuple):
    """Ensure that the variant generates a valid remainder by checking that
       that all indices that occur in the numerator or denominator of the
       remainder also occur in the ERI part of the remainder.
       Say we want to factor t2sq = t_ik^ac t_jk^bc in some term.
       Because the has no denominator, the denominator of the original term
       where we try to factor the intermediate is ignored. However, one needs
       to be careful which indices are chosen as contracted intermediate
       indices k and c. They are not allowed to occur anywhere else in the
       term, i.e., also not in the denominator or numerator.
       """
    remainder = EriOrbenergy(remainder)
    required_frac_idx = set(remainder.num.idx) | set(remainder.denom.idx)
    missing_idx = (
        required_frac_idx - (set(remainder.eri.idx) | set(itmd_indices))
    )
    # maybe swith to return True/False and continue in the calling function
    # if False is Returned?
    # -> raise error for now and have a look at the cases
    if missing_idx:
        raise NotImplementedError(
            "All indices that occur in the term have to be present in the "
            "ERI part or the itmd_indices, i.e., no indices are allowed to "
            "only occur in the denominator or numerator of the remainder. "
            "This avoids only partially removing contracted itmd_indices from "
            f"a term.\n{remainder}"
        )


def _map_on_other_terms(itmd_i: int, remainder: e.Expr,
                        itmd_term_map, itmd_indices: tuple,
                        itmd_default_idx: tuple[str]):
    """Checks on which other itmd_terms the current itmd_term can be mapped if
       the symmetry of the remainder is taken into account. A set of all
       terms, the current term contributes to is returned."""
    from .symmetry import Permutation, PermutationProduct

    # find the itmd indices that are no target indices of the overall term
    # -> those are available for permutations
    target_indices = remainder.terms[0].target
    idx_to_permute = {s for s in itmd_indices if s not in target_indices}
    # copy the remainder and set the previously determined
    # indices as target indices
    rem: e.Expr = remainder.copy()
    rem.set_target_idx(idx_to_permute)
    # create a substitution dict to map the minimal indices to the
    # default indices of the intermediate
    minimal_to_default = {o: n for o, n in zip(itmd_indices, itmd_default_idx)}
    # iterate over the subset of remainder symmetry that only involves
    # non-target intermediate indices
    matching_itmd_terms: set[int] = {itmd_i}
    for perms, perm_factor in rem.terms[0].symmetry(only_target=True).items():
        # translate the permutations to the default indices
        perms = PermutationProduct(
            Permutation(minimal_to_default[p], minimal_to_default[q])
            for p, q in perms
        )
        # look up the translated symmetry in the term map
        term_map: dict = itmd_term_map[(perms, perm_factor)]
        if itmd_i in term_map:
            matching_itmd_terms.add(term_map[itmd_i])
    return matching_itmd_terms


def _compare_eri_parts(term: EriOrbenergy, itmd_term: EriOrbenergy,
                       term_data=None, itmd_term_data=None) -> list:
    """Compare the eri parts of two terms and return the substitutions
           that are necessary to transform the itmd_eri."""
    from itertools import product

    # the eri part of the term to factor has to be at least as long as the
    # eri part of the itmd (prefactors are separated!)
    if len(itmd_term.eri) > len(term.eri):
        return None

    objects = term.eri.objects
    itmd_objects = itmd_term.eri.objects

    # generate term_data if not provided
    if term_data is None:
        term_data = FactorizationTermData(term)
    # generate itmd_data if not provided
    if itmd_term_data is None:
        itmd_term_data = FactorizationTermData(itmd_term)

    relevant_itmd_data = zip(enumerate(itmd_term_data.eri_pattern),
                             itmd_term_data.eri_obj_indices,
                             itmd_term_data.eri_obj_symmetry)

    # compare all objects in the eri parts
    variants = []
    for (itmd_i, (itmd_descr, itmd_coupl)), itmd_indices, itmd_obj_sym in \
            relevant_itmd_data:
        itmd_obj_exponent = itmd_objects[itmd_i].exponent

        relevant_data = zip(enumerate(term_data.eri_pattern),
                            term_data.eri_obj_indices)
        # list to collect all obj that can match the itmd_obj
        # with their corresponding sub variants
        itmd_obj_matches = []
        for (i, (descr, coupl)), indices in relevant_data:
            # tensors have same name and space?
            # is the coupling of the itmd_obj a subset of the obj coupling?
            if descr != itmd_descr or any(coupl[c] < n for c, n in
                                          itmd_coupl.items()):
                continue
            # collect the obj index n-times to indicate how often the
            # object has to be cancelled (possibly multiple times depending
            # on the exponent of the itmd_obj)
            to_cancel = [i for _ in range(itmd_obj_exponent)]
            # create all possibilites to map the indices onto each other
            # by taking the symmetry of the itmd_obj into account
            # store them as tuple: (obj_indices, sub, factor)
            itmd_obj_matches.append((to_cancel,
                                     dict(zip(itmd_indices, indices)),
                                     1))
            for perms, factor in itmd_obj_sym.items():
                perm_itmd_indices = itmd_indices
                for p, q in perms:
                    sub = {p: q, q: p}
                    perm_itmd_indices = [sub.get(s, s) for s in
                                         perm_itmd_indices]
                itmd_obj_matches.append((to_cancel,
                                         dict(zip(perm_itmd_indices, indices)),
                                         factor))
        # was not possible to map the itmd_obj onto any obj in the term
        # -> terms can not match
        if not itmd_obj_matches:
            return None

        if not variants:  # initialize variants
            variants.extend(itmd_obj_matches)
        else:  # try to add the mapping of the current itmd_obj
            extended_variants = []
            for (i_list, sub, factor), (new_i_list, new_sub, new_factor) in \
                    product(variants, itmd_obj_matches):
                # was the obj already mapped onto another itmd_obj?
                # do we have a contradiction in the sub_dicts?
                #  -> a index in the itmd can only be mapped onto 1 index
                #     in the term simultaneously
                if new_i_list[0] not in i_list and all(
                        o not in sub or sub[o] is n
                        for o, n in new_sub.items()):
                    extended_variants.append((i_list + new_i_list,
                                              sub | new_sub,  # OR combine dict
                                              factor * new_factor))
            if not extended_variants:  # no valid combinations -> cant match
                return None
            variants = extended_variants
    # validate the found variants to map the terms onto each other
    valid = []
    for i_list, sub_dict, factor in variants:
        i_set = set(i_list)
        # did we find a match for all itmd_objects?
        if len(i_set) != len(itmd_objects):
            continue
        # extract the objects of the term
        relevant_obj = Mul(*(objects[i].sympy for i in i_set))
        # apply the substitutions to the itmd_term, remove the prefactor
        # (the substitutions might introduce a factor of -1 that we don't need)
        # and check if the substituted itmd_term is identical to the subset
        # of objects
        sub_list = order_substitutions(sub_dict)
        sub_itmd_eri = itmd_term.eri.subs(sub_list)

        if sub_itmd_eri.sympy is S.Zero:  # invalid substitution list
            continue
        pref = sub_itmd_eri.terms[0].prefactor  # +-1

        if relevant_obj - sub_itmd_eri.sympy * pref is S.Zero:
            valid.append((i_list, sub_dict, sub_list, factor))
    return valid if valid else None


def _compare_terms(term: EriOrbenergy, itmd_term: EriOrbenergy,
                   term_data=None, itmd_term_data=None) -> None | list:
    """Compare two terms and return a substitution dict that makes the
        itmd_term equal to the term. Also the indices of the objects in the
        eri part and the denominator that match the intermediate's objects
        are returned."""

    eri_variants = _compare_eri_parts(term, itmd_term, term_data,
                                      itmd_term_data)

    if eri_variants is None:
        return None

    # itmd_term has no denominator -> stop here
    if itmd_term.denom.sympy.is_number:
        return [{'eri_i': eri_i, 'denom_i': [],
                 'sub': sub_dict, 'sub_list': sub_list, 'factor': factor}
                for eri_i, sub_dict, sub_list, factor in eri_variants]

    # term and itmd_term should have a denominator at this point
    # -> extract the brackets
    brackets = term.denom_brackets
    itmd_brackets = itmd_term.denom_brackets
    # extract the lengths of all brakets
    bracket_lengths = [len(bk) for bk in brackets]
    # prescan the brackets according to their length to avoid unnecessary
    # substitutions
    compatible_brackets = {}
    for itmd_denom_i, itmd_bk in enumerate(itmd_brackets):
        itmd_bk_length = len(itmd_bk)
        matching_brackets = [denom_i for denom_i, bk_length
                             in enumerate(bracket_lengths)
                             if bk_length == itmd_bk_length]
        if not matching_brackets:  # could not find a match for a itmd bracket
            return None
        compatible_brackets[itmd_denom_i] = matching_brackets

    # check which of the found substitutions are also valid for the denominator
    variants = []
    for eri_i, sub_dict, sub_list, factor in eri_variants:
        # can only map each bracket onto 1 itmd bracket
        # otherwise something should be wrong
        denom_matches = []
        for itmd_denom_i, denom_idx_list in compatible_brackets.items():
            itmd_bk = itmd_brackets[itmd_denom_i]
            # extract base and exponent of the bracket
            if isinstance(itmd_bk, e.Expr):
                itmd_bk_exponent = 1
                itmd_bk = itmd_bk.sympy
            else:  # polynom  -> Pow object
                itmd_bk_exponent = itmd_bk.exponent
                itmd_bk = itmd_bk.extract_pow

            # apply the substitutions to the base of the bracket
            sub_itmd_bk = itmd_bk.subs(sub_list)
            if sub_itmd_bk is S.Zero:  # invalid substitution list
                continue

            # try to find a match in the subset of brackets of equal length
            for denom_i in denom_idx_list:
                if denom_i in denom_matches:  # denom bk is already assigned
                    continue
                bk = brackets[denom_i]
                # extract the base of the bracket
                bk = bk.sympy if isinstance(bk, e.Expr) else bk.extract_pow
                if sub_itmd_bk - bk is S.Zero:  # brackets are equal?
                    denom_matches.extend(denom_i for _ in
                                         range(itmd_bk_exponent))
                    break
            # did not run into the break:
            # -> could not find a match for the itmd_bracket
            # -> directly skip to next eri_variant
            else:
                break
        # did we find a match for all itmd brackets?
        if len(set(denom_matches)) == len(itmd_brackets):
            variants.append({'eri_i': eri_i, 'denom_i': denom_matches,
                             'sub': sub_dict, 'sub_list': sub_list,
                             'factor': factor})
    return variants if variants else None


def _compare_remainder(remainder: e.Expr, ref_remainder: e.Expr,
                       itmd_indices: tuple) -> int | None:
    """Try to map remainder onto ref_remainder. Return None if it is not
       possible. If the two remainders can be mapped, the required factor (+-1)
       is returned to indicate whether the sign of remainder needs to be
       changed to achieve equality."""
    from .reduce_expr import factor_eri_parts, factor_denom

    # if we have a number as remainder, it should be +-1
    if remainder.sympy is S.Zero or ref_remainder.sympy is S.Zero:
        raise ValueError("It should not be possible for a remainder to "
                         "be equal to 0.")

    # in addition to the target indices, the itmd_indices have to be fixed too.
    # -> set both indices sets as target indices of the expressions
    fixed_indices = remainder.terms[0].target
    assert fixed_indices == ref_remainder.terms[0].target
    fixed_indices += itmd_indices

    # create a copy of the expressions to keep the assumptions of the original
    # expressions valid (assumptions should hold the correct target indices)
    remainder, ref_remainder = remainder.copy(), ref_remainder.copy()
    remainder.set_target_idx(fixed_indices)
    ref_remainder.set_target_idx(fixed_indices)

    # TODO: we have a different situation in this function, because not all
    #       contracted indices have to occur in the eri part of the remainder:
    #         eri indices: jkln. Additionally we have m in the denominator.
    #       the function will only map n->m but not m->n, because it does
    #       not occur in the eri part. This might mess up the denominator
    #       or numerator of the term completely!
    #       -> can neither use find_compatible_terms nor compare_terms!!
    # I think in a usual run this should only occur if previously some
    # intermediate was not found correctly, because for t-amplitudes all
    # removed indices either only occur in the eri part or occur in eri and
    # denom. But if we did not find some t-amplitude and have some denominator
    # left, this problem might occur if a denom idx is a contracted index
    # in the eri part of the itmd.
    # -> but then we can not factor the itmd anyway, because the contracted
    #    idx in the eri part and the denom have to be identical
    # -> need to be solved at another point

    difference = remainder - ref_remainder
    if len(difference) == 1:  # already identical -> 0 or added to 1 term
        return 1 if difference.sympy is S.Zero else -1
    # check if the eri parts of both remainders can be mapped onto each other
    factored = factor_eri_parts(difference)
    if len(factored) > 1:  # eri parts not compatible
        return None

    # check if the denominators are compatible too.
    factored = factor_denom(factored[0])
    if len(factored) > 1:  # denominators are not compatible
        return None
    return 1 if factored[0].sympy is S.Zero else -1


class FactorizationTermData:
    """Class that extracts some data needed for the intermediate factorization.
       """

    def __init__(self, term: EriOrbenergy):
        self._term = term

    @cached_property
    def eri_pattern(self) -> tuple:
        """Returns the pattern of the eri part of the term. In contrast to the
           pattern used in simplify, the pattern is determined for each object
           as tuple that consists of the object description and the
           coupling of the object."""
        coupling = self._term.eri.coupling(include_exponent=False,
                                           include_target_idx=False)
        return tuple(
            (obj.description(include_exponent=False, include_target_idx=False),
             Counter(coupling.get(i, [])))
            for i, obj in enumerate(self._term.eri.objects)
        )

    @cached_property
    def eri_obj_indices(self) -> tuple:
        """Indices hold by each of the objects in the eri part."""
        return tuple(obj.idx for obj in self._term.eri.objects)

    @cached_property
    def eri_obj_symmetry(self) -> tuple:
        """Symmetry of all objects in the eri part."""
        return tuple(obj.symmetry() for obj in self._term.eri.objects)

    @cached_property
    def eri_obj_descriptions(self) -> Counter:
        """Count how often each description occurs in the eri part.
           Exponent of the objects is included implicitly by incrementing
           the description counter."""
        return Counter(
            obj.description(include_exponent=False, include_target_idx=False)
            for obj in self._term.eri.objects for _ in range(obj.exponent)
        )

    @cached_property
    def denom_bracket_lengths(self) -> None | Counter:
        """Determine the length of all brackets in the orbital energy
           denominator and count how often each length occurs in the
           denominator."""
        if self._term.denom.is_number:
            return None
        else:
            return Counter(len(bk) for bk in self._term.denom_brackets)
