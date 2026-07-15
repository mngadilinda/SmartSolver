"""Linear algebra step-by-step helpers for SmartSolver."""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import sympy as sp
from sympy import Matrix, symbols

from .math_step_tracker import parse_math


TrackStep = Callable[[sp.Expr, str, str], None]


def parse_variable_names(variables: Sequence[str]) -> List[sp.Symbol]:
    return [symbols(v.strip()) for v in variables if v and str(v).strip()]


def equation_to_row(expr: sp.Expr, syms: Sequence[sp.Symbol]) -> List[sp.Expr]:
    expanded = sp.expand(expr)
    coeffs = [sp.simplify(expanded.coeff(sym)) for sym in syms]
    constant = sp.simplify(-expanded.subs({sym: 0 for sym in syms}))
    return coeffs + [constant]


def build_augmented_matrix(
    equations: Sequence[str],
    variables: Sequence[str],
) -> Tuple[Matrix, List[sp.Symbol]]:
    syms = parse_variable_names(variables)
    if not syms:
        raise ValueError('At least one variable is required.')
    if not equations:
        raise ValueError('At least one equation is required.')

    rows: List[List[sp.Expr]] = []
    for equation in equations:
        text = (equation or '').strip()
        if not text:
            continue
        if '=' not in text:
            raise ValueError(f'Equation must contain =: {text}')
        left, right = text.split('=', 1)
        expr = sp.simplify(parse_math(left) - parse_math(right))
        rows.append(equation_to_row(expr, syms))

    if not rows:
        raise ValueError('At least one equation is required.')

    return Matrix(rows).applyfunc(sp.simplify), syms


def parse_matrix_input(matrix_input) -> Matrix:
    if isinstance(matrix_input, Matrix):
        return matrix_input.applyfunc(sp.simplify)
    if isinstance(matrix_input, (list, tuple)):
        return Matrix(matrix_input).applyfunc(sp.simplify)
    text = str(matrix_input).strip()
    if not text:
        raise ValueError('Matrix input is required.')
    parsed = parse_math(text)
    if isinstance(parsed, Matrix):
        return parsed.applyfunc(sp.simplify)
    raise ValueError('Matrix input must be a SymPy Matrix or nested list.')


def _matrix_step_expr(matrix: Matrix) -> Matrix:
    return matrix.applyfunc(sp.simplify)


def gaussian_elimination_steps(
    augmented: Matrix,
    track_step: TrackStep,
) -> Matrix:
    """Row-reduce an augmented matrix, recording elementary row operations."""
    m = augmented.copy().applyfunc(sp.simplify)
    n_rows, n_cols = m.shape
    n_vars = n_cols - 1

    track_step(_matrix_step_expr(m), 'Form the augmented matrix [A | b]', 'Augmented matrix')

    pivot_row = 0
    for col in range(n_vars):
        swap_row = None
        for row in range(pivot_row, n_rows):
            if m[row, col] != 0:
                swap_row = row
                break
        if swap_row is None:
            continue

        if swap_row != pivot_row:
            m.row_swap(pivot_row, swap_row)
            track_step(
                _matrix_step_expr(m),
                f'Swap row {pivot_row + 1} and row {swap_row + 1} to get a pivot in column {col + 1}',
                'Row swap',
            )

        pivot_val = sp.simplify(m[pivot_row, col])
        if pivot_val != 1:
            m[pivot_row, :] = (m[pivot_row, :] / pivot_val).applyfunc(sp.simplify)
            track_step(
                _matrix_step_expr(m),
                f'Scale row {pivot_row + 1} so the pivot in column {col + 1} is 1',
                'Scale row',
            )

        for row in range(n_rows):
            if row == pivot_row:
                continue
            factor = sp.simplify(m[row, col])
            if factor == 0:
                continue
            m[row, :] = (m[row, :] - factor * m[pivot_row, :]).applyfunc(sp.simplify)
            track_step(
                _matrix_step_expr(m),
                f'Row {row + 1} <- row {row + 1} - ({factor}) * row {pivot_row + 1}',
                'Row elimination',
            )

        pivot_row += 1
        if pivot_row >= n_rows:
            break

    track_step(_matrix_step_expr(m), 'Reduced row echelon form reached', 'Reduced row echelon form')
    return m.applyfunc(sp.simplify)


def extract_solution_from_rref(
    rref: Matrix,
    syms: Sequence[sp.Symbol],
) -> Tuple[List[sp.Expr], bool]:
    """Read variable values from an augmented matrix in RREF."""
    n_rows, n_cols = rref.shape
    n_vars = len(syms)
    values = [None] * n_vars
    consistent = True

    for row in range(n_rows):
        lhs = sp.simplify(sum(rref[row, col] * syms[col] for col in range(n_vars)))
        rhs = sp.simplify(rref[row, n_cols - 1])
        row_expr = sp.simplify(lhs - rhs)

        if row_expr == 0:
            continue
        if all(sp.simplify(rref[row, col]) == 0 for col in range(n_vars)):
            if rhs != 0:
                consistent = False
            continue

        pivot_col = next(
            (col for col in range(n_vars) if sp.simplify(rref[row, col]) != 0),
            None,
        )
        if pivot_col is None:
            continue
        values[pivot_col] = sp.simplify(rhs)

    if not consistent:
        return [], False

    if any(value is None for value in values):
        return [], True

    return values, True


def solve_linear_system_with_steps(
    equations: Sequence[str],
    variables: Sequence[str],
    track_step: TrackStep,
) -> Tuple[List[sp.Expr], Matrix, List[sp.Symbol], bool]:
    augmented, syms = build_augmented_matrix(equations, variables)
    rref = gaussian_elimination_steps(augmented, track_step)
    solution, consistent = extract_solution_from_rref(rref, syms)
    return solution, rref, syms, consistent


def matrix_rref_with_steps(
    matrix_input,
    track_step: TrackStep,
) -> Matrix:
    matrix = parse_matrix_input(matrix_input)
    if matrix.cols == 0:
        raise ValueError('Matrix must have at least one column.')

    m = matrix.copy().applyfunc(sp.simplify)
    track_step(_matrix_step_expr(m), 'Start with the input matrix', 'Setup')

    n_rows, n_cols = m.shape
    pivot_row = 0
    for col in range(n_cols):
        swap_row = None
        for row in range(pivot_row, n_rows):
            if m[row, col] != 0:
                swap_row = row
                break
        if swap_row is None:
            continue

        if swap_row != pivot_row:
            m.row_swap(pivot_row, swap_row)
            track_step(
                _matrix_step_expr(m),
                f'Swap row {pivot_row + 1} and row {swap_row + 1}',
                'Row swap',
            )

        pivot_val = sp.simplify(m[pivot_row, col])
        if pivot_val != 1:
            m[pivot_row, :] = (m[pivot_row, :] / pivot_val).applyfunc(sp.simplify)
            track_step(
                _matrix_step_expr(m),
                f'Scale row {pivot_row + 1} to make the pivot 1',
                'Scale row',
            )

        for row in range(n_rows):
            if row == pivot_row:
                continue
            factor = sp.simplify(m[row, col])
            if factor == 0:
                continue
            m[row, :] = (m[row, :] - factor * m[pivot_row, :]).applyfunc(sp.simplify)
            track_step(
                _matrix_step_expr(m),
                f'Row {row + 1} <- row {row + 1} - ({factor}) * row {pivot_row + 1}',
                'Row elimination',
            )

        pivot_row += 1
        if pivot_row >= n_rows:
            break

    track_step(_matrix_step_expr(m), 'Reduced row echelon form reached', 'Reduced row echelon form')
    return m.applyfunc(sp.simplify)
