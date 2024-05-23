from .expr_container import Expr, Term
from .misc import Inputerror
from .rules import Rules
from .simplify import simplify
from .indices import Index, get_symbols, split_idx_string, Indices
from .sympy_objects import (
    KroneckerDelta, NonSymmetricTensor, AntiSymmetricTensor, SymmetricTensor
)
from .logger import log
from sympy.physics.secondquant import (
    F, Fd, FermionicOperator, NO
)
from sympy import S, Add, Mul, Pow, sqrt

from itertools import product

from math import factorial

def eri_unitary_deriv1(eri_idx: [Index], deriv_idx: [Index], notation: str = 'c'):
    """Calculates the first derivative of an ERI with respect to a unitary
    transformation at the current ERI.

    :param eri_idx: The indices of the ERI
    :type eri_idx: list
    :param deriv_idx: The indices to which the derivative is made
    :type deriv_idx: list
    :param notation: Physicist/Chemist
    :type notation: str
    :return: The derivative
    :rtype: Add
    """
    if len(eri_idx) != 4:
        raise Inputerror("4 Indices are required for the ERI, received "
                f"{len(eri_idx)}")
    if len(deriv_idx) != 2:
        raise Inputerror("2 Indices are required for the derivative, "
                f"received {len(deriv_idx)}")
    elif notation not in ['p', 'physicist', 'c', 'chemist']:
        raise Inputerror("Only Physicist (p) and Chemist (c) notation are "
                "valid")
   
    if len(notation) > 1:
        notation = notation[0]
    
    indices = Indices()

    from .sympy_objects import KroneckerDelta as KD
    def ud(mat0, mat1, deriv0, deriv1):
        return -(KD(mat0, deriv0) * KD(mat1, deriv1) - KD(mat0, deriv1) * KD(mat1, deriv0))

    aux_symbols = indices.get_generic_copies(eri_idx)
    
    k0, k1 = deriv_idx
    eri0, eri1, eri2, eri3 = eri_idx

    if notation == 'p':
        a0, a1, b0, b1 = aux_symbols
        v = NonSymmetricTensor('g', (a0, a1, b0, b1))
        expr =  ud(eri0, a0, k0, k1) * KD(eri1, a1) * KD(eri2, b0) * KD(eri3, b1)
        expr += KD(eri0, a0) * ud(eri1, a1, k0, k1) * KD(eri2, b0) * KD(eri3, b1)
        expr += KD(eri0, a0) * KD(eri1, a1) * ud(eri2, b0, k0, k1) * KD(eri3, b1)
        expr += KD(eri0, a0) * KD(eri1, a1) * KD(eri2, b0) * ud(eri3, b1, k0, k1)
        return -v * expr

    a0, b0, a1, b1 = aux_symbols
    v = SymmetricTensor('v', (a0, b0), (a1, b1))

    expr =  ud(eri0, a0, k0, k1) * KD(eri1, b0) * KD(eri2, a1) * KD(eri3, b1)
    expr += KD(eri0, a0) * ud(eri1, b0, k0, k1) * KD(eri2, a1) * KD(eri3, b1)
    expr += KD(eri0, a0) * KD(eri1, b0) * ud(eri2, a1, k0, k1) * KD(eri3, b1)
    expr += KD(eri0, a0) * KD(eri1, b0) * KD(eri2, a1) * ud(eri3, b1, k0, k1)
    return -v * expr

def eri_unitary_deriv2(eri_idx: [Index], deriv_idx: [Index], notation: str = 'c'):
    """Calculates the second derivative of an ERI with respect to a unitary
    transformation at the current ERI.

    :param eri_idx: The indices of the ERI
    :type eri_idx: list
    :param deriv_idx: The indices to which the derivative is made
    :type deriv_idx: list
    :param notation: Physicist/Chemist
    :type notation: str
    :return: The derivative
    :rtype: Add
    """
    if len(eri_idx) != 4:
        raise Inputerror("4 Indices are required for the ERI, received "
                f"{len(eri_idx)}")
    if len(deriv_idx) != 4:
        raise Inputerror("4 Indices are required for the derivative, "
                f"received {len(deriv_idx)}")
    elif notation not in ['p', 'physicist', 'c', 'chemist']:
        raise Inputerror("Only Physicist (p) and Chemist (c) notation are "
                "valid")
   
    if len(notation) > 1:
        notation = notation[0]

    indices = Indices()

    from .sympy_objects import KroneckerDelta as KD
    def ud(mat0, mat1, deriv0, deriv1):
        return -(KD(mat0, deriv0) * KD(mat1, deriv1) - KD(mat0, deriv1) * KD(mat1, deriv0))

    def ud2(mat0, mat1, deriv0, deriv1, deriv2, deriv3):
        cont_idx = indices.get_generic_copies((mat0,))[0]
        m1_a = ud(mat0, cont_idx, deriv0, deriv1)
        m2_a = ud(cont_idx, mat1, deriv2, deriv3)
        m1_b = ud(mat0, cont_idx, deriv2, deriv3)
        m2_b = ud(cont_idx, mat1, deriv0, deriv1)
        return 0.5 * (m1_a * m2_a + m1_b * m2_b)

    aux_symbols = indices.get_generic_copies(eri_idx)
    k0, k1, k2, k3 = deriv_idx
    eri0, eri1, eri2, eri3 = eri_idx

    if notation == 'p':
        a0, a1, b0, b1 = aux_symbols
        v = NonSymmetricTensor('g', (a0, a1, b0, b1))
        # Building the prefactor to d U_pq / d k_ab
        expr0 =  ud(eri1, a1, k2, k3) * KD(eri2, b0) * KD(eri3, b1)
        expr0 += KD(eri1, a1) * ud(eri2, b0, k2, k3) * KD(eri3, b1)
        expr0 += KD(eri1, a1) * KD(eri2, b0) * ud(eri3, b1, k2, k3)
        # Building the prefactor to the second term
        expr1 =  ud(eri0, a0, k2, k3) * KD(eri2, b0) * KD(eri3, b1)
        expr1 += KD(eri0, a0) * ud(eri2, b0, k2, k3) * KD(eri3, b1)
        expr1 += KD(eri0, a0) * KD(eri2, b0) * ud(eri3, b1, k2, k3)
        # Building the prefactor to the third term
        expr2 =  ud(eri0, a0, k2, k3) * KD(eri1, a1) * KD(eri3, b1)
        expr2 += KD(eri0, a0) * ud(eri1, a1, k2, k3) * KD(eri3, b1)
        expr2 += KD(eri0, a0) * KD(eri1, a1) * ud(eri3, b1, k2, k3)
        # Building the prefactor to the fourth term
        expr3 =  ud(eri0, a0, k2, k3) * KD(eri1, a1) * KD(eri2, b0)
        expr3 += KD(eri0, a0) * ud(eri1, a1, k2, k3) * KD(eri2, b0)
        expr3 += KD(eri0, a0) * KD(eri1, a1) * ud(eri2, b0, k2, k3)

        # Building the complete prefactor
        expr = (ud(eri0, a0, k0, k1) * expr0 + ud2(eri0, a0, k0, k1, k2, k3) * KD(eri1, a1) * KD(eri2, b0) * KD(eri3, b1)
                + ud(eri1, a1, k0, k1) * expr1 + KD(eri0, a0) * ud2(eri1, a1, k0, k1, k2, k3) * KD(eri2, b0) * KD(eri3, b1)
                + ud(eri2, b0, k0, k1) * expr2 + KD(eri0, a0) * KD(eri1, a1) * ud2(eri2, b0, k0, k1, k2, k3) * KD(eri3, b1)
                + ud(eri3, b1, k0, k1) * expr3 + KD(eri0, a0) * KD(eri1, a1) * KD(eri2, b0) * ud2(eri3, b1, k0, k1, k2, k3))

        # Returning the complete expression
        return v * expr
    
    a0, b0, a1, b1 = aux_symbols
    
    #log(f"ERI symbols: {eri0}, {eri1}, {eri2}, {eri3}")
    #log(f"Aux symbols: {a0}, {b0}, {a1}, {b1}")

    v = SymmetricTensor('v', (a0, b0), (a1, b1))

    # Building the prefactor to d U_pq / d k_ab
    expr0 =  ud(eri1, b0, k2, k3) * KD(eri2, a1) * KD(eri3, b1)
    expr0 += KD(eri1, b0) * ud(eri2, a1, k2, k3) * KD(eri3, b1)
    expr0 += KD(eri1, b0) * KD(eri2, a1) * ud(eri3, b1, k2, k3)
    # Building the prefactor to the second term
    expr1 =  ud(eri0, a0, k2, k3) * KD(eri2, a1) * KD(eri3, b1)
    expr1 += KD(eri0, a0) * ud(eri2, a1, k2, k3) * KD(eri3, b1)
    expr1 += KD(eri0, a0) * KD(eri2, a1) * ud(eri3, b1, k2, k3)
    # Building the prefactor to the third term
    expr2 =  ud(eri0, a0, k2, k3) * KD(eri1, b0) * KD(eri3, b1)
    expr2 += KD(eri0, a0) * ud(eri1, b0, k2, k3) * KD(eri3, b1)
    expr2 += KD(eri0, a0) * KD(eri1, b0) * ud(eri3, b1, k2, k3)
    # Building the prefactor to the fourth term
    expr3 =  ud(eri0, a0, k2, k3) * KD(eri1, b0) * KD(eri2, a1)
    expr3 += KD(eri0, a0) * ud(eri1, b0, k2, k3) * KD(eri2, a1)
    expr3 += KD(eri0, a0) * KD(eri1, b0) * ud(eri2, a1, k2, k3)

    # Building the complete prefactor
    expr = (ud(eri0, a0, k0, k1) * expr0 + ud2(eri0, a0, k0, k1, k2, k3) * KD(eri1, b0) * KD(eri2, a1) * KD(eri3, b1)
            + ud(eri1, b0, k0, k1) * expr1 + KD(eri0, a0) * ud2(eri1, b0, k0, k1, k2, k3) * KD(eri2, a1) * KD(eri3, b1)
            + ud(eri2, a1, k0, k1) * expr2 + KD(eri0, a0) * KD(eri1, b0) * ud2(eri2, a1, k0, k1, k2, k3) * KD(eri3, b1)
            + ud(eri3, b1, k0, k1) * expr3 + KD(eri0, a0) * KD(eri1, b0) * KD(eri2, a1) * ud2(eri3, b1, k0, k1, k2, k3))

    # Returning the complete expression
    return v * expr

def commutator(e1: Expr, e2: Expr) -> Expr:
    tidx1 = e1._target_idx
    tidx2 = e2._target_idx
    if tidx1 is None or tidx2 is None:
        raise NotImplementedError('Target indices must be defined')
    if len(tidx1) != 2 or len(tidx2) != 2:
        raise NotImplementedError('Commutators are only implemented for matrices with 2 target indices. Found {len(tidx1)} / {len(tidx2)}')
    spaces = set([i.space for i in tidx1+tidx2])
    spins = set([i.spin for i in tidx1+tidx2])
    if len(spaces) > 1 or len(spins) > 1:
        raise NotImplementedError('Commutators are only available for qudratic matrices with equal spin and spatial parts.')
    idx = Indices()
    g1, g2, cont_idx = idx.get_generic_copies((tidx1[0], tidx2[1], tidx1[1]))
    e1_c1 = e1.copy().subs(tidx1[1], cont_idx).subs(tidx1[0], g1)
    e2_c1 = e2.copy().subs(tidx2[0], cont_idx).subs(tidx2[1], g2)
    e1_c2 = e1.copy().subs(tidx1[0], cont_idx).subs(tidx1[1], g2)
    e2_c2 = e2.copy().subs(tidx2[1], cont_idx).subs(tidx2[0], g1)
    for e in (e1_c1, e2_c1, e1_c2, e2_c2):
        e.set_target_idx((g1, g2))
    return e1_c1 * e2_c1 - e1_c2 * e2_c2


def is_term_connected(term: Term) -> bool:
    if len(term.objects) == 1:
        return True
    index_list = [obj.idx for obj in term.objects if len(obj.idx) > 0]
    # Create a dictionary to represent the graph
    graph = {}
    
    # Build the graph based on shared strings
    for sublist in index_list:
        for i, index in enumerate(sublist):
            if index not in graph:
                graph[index] = set()
            graph[index].update(sublist[:i])
            graph[index].update(sublist[i+1:])

    # Perform depth-first search (DFS) to check connectivity
    visited = set()
    stack = [next(iter(graph.keys()))]  # Start from an arbitrary string
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(graph.get(node, set()) - visited)
    
    # If all strings are visited, the graph is connected
    return 2*len(visited) == sum(len(sublist) for sublist in index_list)

def remove_disconnected_terms(expr: Expr):
    e_new = 0
    for term in expr.terms:
        if is_term_connected(term):
            e_new += term
    return Expr(e_new)

def is_term_mean_field(term: Term) -> bool:
    for obj in term.objects:
        if isinstance(obj.sympy, AntiSymmetricTensor):
            if str(obj.sympy.symbol) == 'V':
                red_idx = set(obj.upper)
                red_idx.update(obj.lower)
                if len(red_idx) < 4:
                    return True
    return False

def remove_mean_field_terms(expr: Expr):
    e_new = 0
    for term in expr.terms:
        if not is_term_mean_field(term):
            e_new += term
    return Expr(e_new)

def similarity_transform(operator: Expr, sim_op: Expr, max_order: int = 10):
    res = operator
    prev_expr = operator
    for i in range(1, max_order+1):
        new_comm = commutator(prev_expr, sim_op)
        if new_comm is S.Zero:
            break
        res += Rational(1, factorial(i)) * new_comm
        prev_expr = new_comm
    return prev_expr

def gen_term_orders(order, term_length, min_order):
    """Generates all combinations that contribute to the n'th order
       contribution of a term x*x*x*..., where x is expanded in a perturbation
       expansion.

       :param order: The desired order
       :type order: int
       :param term_length: The number of objects in the term to expand in
            perturbation theory.
       :type term_length: int
       :param min_order: The minimum order that should be considered
       :type min_order: int
       :return: All possible combinations of a given order
       :rtype: list
       """

    if not all(isinstance(n, int) and n >= 0
               for n in [order, term_length, min_order]):
        raise Inputerror("Order, term_length and min_order need to be "
                         "non-negative integers.")

    orders = (o for o in range(min_order, order + 1))
    combinations = product(orders, repeat=term_length)
    return [comb for comb in combinations if sum(comb) == order]


def import_from_sympy_latex(expr_string: str) -> Expr:
    """Function for importing an expression from a sympy latex string:
       latex(expression) -> string.
       The returned expression does not contain any assumptions."""

    def import_indices(indices: str):
        # split at the end of each index with a spin label
        # -> n1n2n3_{spin}
        idx = []
        for sub_part in indices.split("}"):
            if not sub_part:  # skip empty string
                continue
            if "_{\\" in sub_part:  # the last index has a spin label
                names, spin = sub_part.split("_{\\")
                if spin not in ["alpha", "beta"]:
                    raise RuntimeError(f"Found invalid spin on Index: {spin}. "
                                       f"Input: {indices}")
                names = split_idx_string(names)
                idx.extend(get_symbols(names[:-1]))
                idx.extend(get_symbols(names[-1], spin[0]))
            else:  # no index has a spin label
                idx.extend(get_symbols(sub_part))
        return idx

    def import_tensor(tensor: str):
        # split the tensor in base and exponent
        stack = []
        separator = None
        for i, c in enumerate(tensor):
            if c == "{":
                stack.append(c)
            elif c == "}":
                assert stack.pop() == "{"
            elif not stack and c == "^":
                separator = i
                break
        if separator is None:
            exponent = 1
        else:
            exponent = tensor[separator+1:]
            exponent = int(exponent.lstrip("{").rstrip("}"))
            tensor = tensor[:separator]
        # done with processing the exponent
        # -> deal with the tensor. remove 1 layer of curly brackets and
        #    afterwards split the tensor string into its components
        if tensor[0] == "{":
            tensor = tensor[1:]
        if tensor[-1] == "}":
            tensor = tensor[:-1]
        stack.clear()
        components = []
        temp = []
        for i, c in enumerate(tensor):
            if c == "{":
                stack.append(c)
            elif c == "}":
                assert stack.pop() == "{"
            elif not stack and c in ["^", "_"]:
                components.append("".join(temp))
                temp.clear()
                continue
            temp.append(c)
        if temp:
            components.append("".join(temp))
        name, indices = components[0], components[1:]
        # remove 1 layer of brackets from all indices
        for i, idx in enumerate(indices):
            if idx[0] == "{":
                idx = idx[1:]
            if idx[-1] == "}":
                idx = idx[:-1]
            indices[i] = idx

        if name == "a":  # create / annihilate
            if len(indices) == 2 and indices[0] == "\\dagger":
                base = Fd(*import_indices(indices[1]))
            elif len(indices) == 1:
                base = F(*import_indices(indices[0]))
            else:
                raise RuntimeError("Unknown second quantized operator: ",
                                   tensor)
        elif len(indices) == 2:  # antisymtensor
            upper = import_indices(indices[0])
            lower = import_indices(indices[1])
            if name == "v":
                base = SymmetricTensor(name, upper, lower)
            else:
                base = AntiSymmetricTensor(name, upper, lower)
        elif len(indices) == 1:  # nonsymtensor
            base = NonSymmetricTensor(name, import_indices(indices[0]))
        else:
            raise RuntimeError(f"Unknown tensor object: {tensor}")
        return Pow(base, exponent)

    def import_obj(obj_str: str):
        # import an individial object
        if obj_str.isnumeric():  # prefactor
            return int(obj_str)
        elif obj_str.startswith("\\sqrt{"):  # sqrt{x} prefactor
            return sqrt(int(obj_str[:-1].replace("\\sqrt{", "", 1)))
        elif obj_str.startswith("\\delta_"):  # KroneckerDelta
            idx = obj_str[:-1].replace("\\delta_{", "", 1).split()
            idx = import_indices("".join(idx))
            if len(idx) != 2:
                raise RuntimeError(f"Invalid indices for delta: {idx}.")
            return KroneckerDelta(*idx)
        elif obj_str.startswith("\\left("):  # braket
            # need to take care of exponent of the braket!
            base, exponent = obj_str.rsplit('\\right)', 1)
            if exponent:  # exponent != "" -> ^{x} -> exponent != 1
                exponent = int(exponent[:-1].lstrip('^{'))
            else:
                exponent = 1
            obj_str = base.replace("\\left(", "", 1)
            return Pow(import_from_sympy_latex(obj_str).sympy, exponent)
        elif obj_str.startswith("\\left\\{"):  # NO
            no, unexpected_stuff = obj_str.rsplit("\\right\\}", 1)
            if unexpected_stuff:
                raise NotImplementedError(f"Unexpected NO object: {obj_str}.")
            obj_str = no.replace("\\left\\{", "", 1)
            return NO(import_from_sympy_latex(obj_str).sympy)
        else:  # tensor or creation/annihilation operator
            return import_tensor(obj_str)

    def split_terms(expr_string: str) -> list[str]:
        stack: list[str] = []
        terms: list[str] = []

        term_start_idx = 0
        for i, char in enumerate(expr_string):
            if char in ['{', '(']:
                stack.append(char)
            elif char == '}':
                assert stack.pop() == '{'
            elif char == ')':
                assert stack.pop() == '('
            elif char in ['+', '-'] and not stack and i != term_start_idx:
                terms.append(expr_string[term_start_idx:i])
                term_start_idx = i
        terms.append(expr_string[term_start_idx:])  # append last term
        return terms

    def import_term(term_string: str) -> list[str]:
        from sympy import Mul

        stack: list[str] = []
        objects: list[str] = []

        obj_start_idx = 0
        for i, char in enumerate(term_string):
            if char in ['{', '(']:
                stack.append(char)
            elif char == '}':
                assert stack.pop() == '{'
            elif char == ')':
                assert stack.pop() == '('
            # in case we have a denom of the form:
            # 2a+2b+4c and not 2 * (a+b+2c)
            elif char in ['+', '-'] and not stack:
                return import_from_sympy_latex(term_string).sympy
            elif char == " " and not stack and i != obj_start_idx:
                objects.append(term_string[obj_start_idx:i])
                obj_start_idx = i + 1
        objects.append(term_string[obj_start_idx:])  # last object
        return Mul(*(import_obj(o) for o in objects))

    expr_string = expr_string.strip()
    if not expr_string:
        return Expr(0)

    terms = split_terms(expr_string)
    if terms[0][0] not in ['+', '-']:
        terms[0] = '+ ' + terms[0]

    sympy_expr = 0
    for term in terms:
        sign = term[0]  # extract the sign of the term
        if sign not in ['+', '-']:
            raise ValueError(f"Found invalid sign {sign} in term {term}")
        term = term[1:].strip()

        sympy_term = -1 if sign == '-' else +1

        if term.startswith("\\frac"):  # fraction
            # remove frac layout and split: \\frac{...}{...}
            num, denom = term[:-1].replace("\\frac{", "", 1).split("}{")
        else:  # no denominator
            num, denom = term, None

        sympy_term *= import_term(num)
        if denom is not None:
            sympy_term /= import_term(denom)
        sympy_expr += sympy_term
    return Expr(sympy_expr)


def evaluate_deltas(expr, target_idx=None):
    """Slightly modified version of the evaluate_deltas function from sympy
       that takes the target indices of the expr as additional input.
       Neccessary if the einstein sum convention is not sufficient
       to determine the target indices in all terms of the expression."""

    if isinstance(expr, Add):
        return expr.func(*[evaluate_deltas(arg, target_idx)
                           for arg in expr.args])
    elif isinstance(expr, Mul):
        if target_idx is None:
            # for determining the target indices it is sufficient to use
            # atoms, which lists every index only once per object, i.e.,
            # (f_ii).atoms(Index) -> i.
            # We are only interested in indices on deltas
            # -> it is sufficient to know that an index occurs on another
            #    object. (twice on the same delta is not possible)
            deltas = []
            indices = {}
            for obj in expr.args:
                for s in obj.atoms(Index):
                    if s in indices:
                        indices[s] += 1
                    else:
                        indices[s] = 0
                if isinstance(obj, KroneckerDelta):
                    deltas.append(obj)
            # extract the target indices and use them in next recursion
            # so they only need to be determined once
            target_idx = [s for s, n in indices.items() if not n]
        else:
            # find all occurrences of kronecker delta
            deltas = [d for d in expr.args if isinstance(d, KroneckerDelta)]
            target_idx = get_symbols(target_idx)

        index_occurences = {}
        for obj in expr.args:
            for s in obj.atoms(Index):
                if s in index_occurences:
                    index_occurences[s] += 1
                else:
                    index_occurences[s] = 0
        for d in deltas:
            # determine the killable and preferred index
            # in the case we have delta_{i p_alpha} we want to keep i_alpha
            # -> a new index is required. But for now just don't evaluate
            #    these deltas
            idx = d.preferred_and_killable
            if idx is None:  # delta_{i p_alpha}
                continue
            preferred, killable = idx
            # try to remove killable

            if killable not in target_idx:
                expr = expr.subs(killable, preferred)
                if len(deltas) > 1:
                    return evaluate_deltas(expr, target_idx)
                continue
            # try to remove preferred.
            # But only if no information is lost if doing so
            # -> killable has to be of length 1
            elif preferred not in target_idx \
                    and d.indices_contain_equal_information:
                expr = expr.subs(preferred, killable)
                if len(deltas) > 1:
                    return evaluate_deltas(expr, target_idx)
        return expr
    else:
        return expr


def wicks(expr, rules: Rules = None, simplify_kronecker_deltas: bool = False):
    """Evaluates Wicks theorem on the provided expression only returning
       fully contracted contributions.
       The resulting Kronecker deltas are evaluated automatically if
       simplify_kronecker_deltas is set.
       If some rules are provided, they are applied to the
       resulting expression before returning.
       Adapted from 'sympy.physics.secondquant'."""

    # normal ordered operator string has to evaluate to zero
    # and a single second quantized operator can not be contracted
    if isinstance(expr, (NO, FermionicOperator)):
        return S.Zero

    # break up any NO-objects, and evaluate commutators
    expr = expr.doit(wicks=True).expand()

    if isinstance(expr, Add):
        return Add(*[wicks(term, rules=rules,
                           simplify_kronecker_deltas=simplify_kronecker_deltas)
                     for term in expr.args])
    elif isinstance(expr, Mul):
        # we don't want to mess around with commuting part of Mul
        # so we factorize it out before starting recursion
        c_part = []
        op_string = []
        for factor in expr.args:
            if factor.is_commutative:
                c_part.append(factor)
            else:
                op_string.append(factor)

        if (n := len(op_string)) == 0:  # no operators
            result = expr
        elif n == 1:  # a single operator
            return S.Zero
        else:  # at least 2 operators
            result = _contract_operator_string(op_string)
            result = (Mul(*c_part) * result).expand()
            if simplify_kronecker_deltas:
                result = evaluate_deltas(result)
    else:  # neither add, Mul, NO or Operator -> maybe a number or a tensor
        return expr

    # apply rules to the result
    if rules is None:
        return result
    elif not isinstance(rules, Rules):
        raise TypeError(f"Rules needs to be of type {Rules}")

    return rules.apply(Expr(result)).sympy


def _contract_operator_string(op_string: list) -> Add:
    """Contracts the provided operator string only returning fully contracted
       contributions.
       Adapted from 'sympy.physics.secondquant'."""
    result = []
    for i in range(1, len(op_string)):
        c = _contraction(op_string[0], op_string[i])
        if c is S.Zero:
            continue
        if not i % 2:  # introduce -1 for swapping operators
            c *= S.NegativeOne

        if len(op_string) - 2 > 0:  # at least one operator left
            # remove the contracted operators from the string and recurse
            remaining = op_string[1:i] + op_string[i+1:]
            result.append(c * _contract_operator_string(remaining))
        else:  # no operators left
            result.append(c)
    return Add(*result)


def _contraction(p, q):
    """Evaluates the contraction of two sqcond quantized fermionic operators.
       Adapted from 'sympy.physics.secondquant'.
    """
    if not isinstance(p, FermionicOperator) or \
            not isinstance(q, FermionicOperator):
        raise NotImplementedError("Contraction only implemented for "
                                  "FermionicOperators.")
    if p.state.spin or q.state.spin:
        raise NotImplementedError("Contraction not implemented for indices "
                                  "with spin.")

    if isinstance(p, F) and isinstance(q, Fd):
        if p.state.assumptions0.get("below_fermi") or \
                q.state.assumptions0.get("below_fermi"):
            return S.Zero
        elif p.state.assumptions0.get("above_fermi") or \
                q.state.assumptions0.get("above_fermi"):
            return KroneckerDelta(p.state, q.state)
        else:
            return (KroneckerDelta(p.state, q.state) *
                    KroneckerDelta(q.state, Index('a', above_fermi=True)))
    elif isinstance(p, Fd) and isinstance(q, F):
        if q.state.assumptions0.get("above_fermi") or \
                p.state.assumptions0.get("above_fermi"):
            return S.Zero
        elif q.state.assumptions0.get("below_fermi") or \
                p.state.assumptions0.get("below_fermi"):
            return KroneckerDelta(p.state, q.state)
        else:
            return (KroneckerDelta(p.state, q.state) *
                    KroneckerDelta(q.state, Index('i', below_fermi=True)))
    else:  # vanish if 2xAnnihilator or 2xCreator
        return S.Zero
