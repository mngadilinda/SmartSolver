"""Multivariable calculus step-by-step helpers for SmartSolver."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import sympy as sp
from sympy import symbols

from .math_step_tracker import parse_math


TrackStep = Callable[[sp.Expr, str, str], None]
Bounds = Union[Tuple[sp.Expr, sp.Expr], List, None]


def parse_variable_names(variables: Sequence[str]) -> List[sp.Symbol]:
    return [symbols(v.strip()) for v in variables if v and str(v).strip()]


def infer_variables(expression: sp.Expr, explicit: Optional[Sequence[str]] = None) -> List[sp.Symbol]:
    if explicit:
        return parse_variable_names(explicit)
    free = sorted(expression.free_symbols, key=lambda sym: sym.name)
    if not free:
        raise ValueError('Could not infer variables from the expression.')
    return list(free)


def partial_derivative_with_steps(
    expression: str,
    wrt: str,
    variables: Optional[Sequence[str]],
    order: int,
    track_step: TrackStep,
    identify_rule: Callable[[sp.Expr, sp.Symbol], str],
) -> Tuple[sp.Expr, List[sp.Symbol]]:
    expr = parse_math(expression)
    all_vars = infer_variables(expr, variables)
    wrt_sym = symbols(wrt.strip())
    if wrt_sym not in all_vars:
        all_vars = all_vars + [wrt_sym]

    held = [sym for sym in all_vars if sym != wrt_sym]
    held_text = ', '.join(str(sym) for sym in held) if held else 'none'
    track_step(
        expr,
        f'Original function: f({", ".join(str(v) for v in all_vars)}) = {expression}',
        'Setup',
    )
    track_step(
        expr,
        f'Find the partial derivative with respect to {wrt_sym}, treating {held_text} as constant',
        'Partial derivative setup',
    )

    current = expr
    for index in range(order):
        current = sp.diff(current, wrt_sym)
        rule = identify_rule(current, wrt_sym) if index == 0 else 'Repeated differentiation'
        track_step(
            current,
            f'Apply differentiation (pass {index + 1}): d/d{wrt_sym} -> {current}',
            rule if index == 0 else 'Partial derivative',
        )

    track_step(current, f'Partial derivative result: {current}', 'Final result')
    return sp.simplify(current), all_vars


def gradient_with_steps(
    expression: str,
    variables: Sequence[str],
    track_step: TrackStep,
    identify_rule: Callable[[sp.Expr, sp.Symbol], str],
) -> Tuple[sp.Matrix, List[sp.Symbol]]:
    expr = parse_math(expression)
    syms = parse_variable_names(variables) or infer_variables(expr)
    track_step(
        expr,
        f'Original function: f({", ".join(str(v) for v in syms)}) = {expression}',
        'Setup',
    )
    track_step(
        expr,
        f'Gradient is the vector of partial derivatives: nabla f = ({", ".join(f"d/d{v}" for v in syms)})',
        'Gradient setup',
    )

    components: List[sp.Expr] = []
    for sym in syms:
        held = [other for other in syms if other != sym]
        held_text = ', '.join(str(other) for other in held)
        partial = sp.simplify(sp.diff(expr, sym))
        track_step(
            partial,
            f'Partial with respect to {sym}, holding {held_text} constant: {partial}',
            identify_rule(partial, sym),
        )
        components.append(partial)

    gradient = sp.Matrix(components)
    track_step(gradient, f'Gradient vector: {gradient}', 'Final result')
    return gradient, syms


def _parse_bound(value) -> Tuple[sp.Expr, sp.Expr]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return parse_math(str(value[0])), parse_math(str(value[1]))
    raise ValueError(f'Each bound must be a pair (lower, upper); got {value!r}')


def multiple_integral_with_steps(
    expression: str,
    variables: Sequence[str],
    bounds: Optional[Dict[str, Bounds]],
    track_step: TrackStep,
) -> Tuple[sp.Expr, List[sp.Symbol]]:
    expr = parse_math(expression)
    syms = parse_variable_names(variables)
    if not syms:
        raise ValueError('At least one integration variable is required.')

    track_step(
        expr,
        f'Integrand: {expression} over variables ({", ".join(str(v) for v in syms)})',
        'Setup',
    )

    current = expr
    bounds = bounds or {}
    for sym in syms:
        var_name = str(sym)
        if var_name in bounds:
            lower, upper = _parse_bound(bounds[var_name])
            current = sp.integrate(current, (sym, lower, upper))
            track_step(
                current,
                f'Integrate with respect to {sym} from {lower} to {upper}',
                'Definite integration',
            )
        else:
            current = sp.integrate(current, sym)
            track_step(
                current,
                f'Integrate with respect to {sym}',
                'Indefinite integration',
            )

    result = sp.simplify(current)
    track_step(result, f'Multiple integral result: {result}', 'Final result')
    return result, syms
