import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import sympy as sp
from sympy import Eq, symbols
from sympy.parsing.sympy_parser import (
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

PARSE_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application,)

TRIG_FUNCTIONS = (sp.sin, sp.cos, sp.tan, sp.sec, sp.csc, sp.cot)
INVERSE_TRIG_FUNCTIONS = (sp.asin, sp.acos, sp.atan, sp.acot, sp.asec, sp.acsc)
LIATE_LABELS = (
    'Logarithmic',
    'Inverse trig',
    'Algebraic',
    'Trigonometric',
    'Exponential',
)


def contains_trig(expr: sp.Expr) -> bool:
    return any(expr.has(func) for func in TRIG_FUNCTIONS)


def contains_log_or_exp(expr: sp.Expr, var: sp.Symbol) -> bool:
    if expr.has(sp.log):
        return True
    if expr.has(sp.exp):
        return True
    for atom in expr.atoms(sp.Pow):
        if atom.has(var) and atom.base.is_number and atom.base != 1:
            return True
    return False


def liate_rank(expr: sp.Expr, var: sp.Symbol) -> int:
    """Lower rank = higher priority to choose as u in integration by parts (LIATE)."""
    if expr.has(sp.log):
        return 0
    if any(expr.has(func) for func in INVERSE_TRIG_FUNCTIONS):
        return 1
    if expr.is_polynomial(var) or (expr.is_Mul and all(a.is_polynomial(var) for a in expr.args)):
        return 2
    if contains_trig(expr):
        return 3
    if expr.has(sp.exp) or (expr.is_Pow and expr.base == sp.E):
        return 4
    if expr.is_polynomial(var):
        return 2
    return 2


@dataclass
class Step:
    """Represents a single step in the solution."""

    description: str
    expression: str
    latex: str
    rule_applied: str
    substeps: List['Step'] = field(default_factory=list)


def step_to_dict(step: Step) -> Dict[str, Any]:
    return {
        'description': step.description,
        'expression': step.expression,
        'latex': step.latex,
        'rule_applied': step.rule_applied,
        'substeps': [step_to_dict(s) for s in step.substeps],
    }


class StepTracker:
    """Tracks all steps during computation."""

    def __init__(self):
        self.steps: List[Step] = []
        self.current_expression = None
        self.original_expression = None

    def reset(self):
        self.steps = []
        self.current_expression = None
        self.original_expression = None

    def add_step(self, description: str, expr: sp.Expr, rule: str = ''):
        step = Step(
            description=description,
            expression=str(expr),
            latex=sp.latex(expr),
            rule_applied=rule,
        )
        self.steps.append(step)
        self.current_expression = expr
        return step

    def get_solution(self) -> List[Step]:
        return self.steps


def normalize_math_input(text: str) -> str:
    """Normalize student-style math notation for SymPy parsing."""
    if not text:
        return ''
    text = text.strip()
    text = text.replace('^', '**')
    text = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', text)
    return text


def parse_math(text: str) -> sp.Expr:
    return parse_expr(normalize_math_input(text), transformations=PARSE_TRANSFORMATIONS)


def solution_to_latex(solutions) -> str:
    if not solutions:
        return 'No solution found'
    if isinstance(solutions, list):
        if len(solutions) == 1:
            return sp.latex(solutions[0])
        return sp.latex(solutions)
    return sp.latex(solutions)


class SmartSolver:
    """Main solver class with step tracking."""

    def __init__(self, show_steps: bool = True):
        self.step_tracker = StepTracker()
        self.show_steps = show_steps

    def _begin_problem(self):
        self.step_tracker.reset()

    def _track_step(self, expr, description, rule=''):
        if self.show_steps:
            self.step_tracker.add_step(description, expr, rule)
        return expr

    def _parse_equation(self, equation: str) -> sp.Expr:
        equation = equation.strip()
        if '=' not in equation:
            equation = f'{equation} = 0'
        left, right = equation.split('=', 1)
        return parse_math(left) - parse_math(right)

    # ============ ALGEBRA SOLVING ============

    def solve_equation(self, equation: str, variable: str = 'x') -> Dict:
        """
        Solve an equation with step-by-step solution.

        Example:
        >>> solver = SmartSolver()
        >>> result = solver.solve_equation("x^2 + 5x + 6 = 0", "x")
        """
        self._begin_problem()
        expr = self._parse_equation(equation)
        var = symbols(variable)

        self._track_step(expr, f'Original equation: {equation}', 'Setup')

        if expr.is_polynomial(var):
            solution = self._solve_polynomial(expr, var)
        elif contains_log_or_exp(expr, var):
            solution = self._solve_log_exp(expr, var)
        elif contains_trig(expr):
            solution = self._solve_trig(expr, var)
        else:
            solution = self._solve_general(expr, var)

        return {
            'steps': self.step_tracker.get_solution(),
            'solutions': solution,
            'solution_latex': solution_to_latex(solution),
        }

    def _solve_polynomial(self, expr: sp.Expr, var: sp.Symbol) -> List:
        degree = sp.degree(expr, var)

        if degree == 0:
            return []
        if degree == 1:
            a = expr.coeff(var, 1)
            b = expr.coeff(var, 0)
            self._track_step(
                expr,
                f'Linear equation: {a}{var} + {b} = 0',
                'Identify linear equation',
            )
            isolated = sp.simplify(-b / a)
            self._track_step(
                isolated,
                f'Divide both sides by {a}: {var} = {-b}/{a}',
                'Isolate variable',
            )
            return [isolated]

        if degree == 2:
            a = expr.coeff(var, 2)
            b = expr.coeff(var, 1)
            c = expr.coeff(var, 0)

            self._track_step(
                expr,
                f'Quadratic equation: {a}{var}^2 + {b}{var} + {c} = 0',
                'Identify quadratic equation',
            )

            discriminant = sp.simplify(b**2 - 4 * a * c)
            self._track_step(
                discriminant,
                f'Calculate discriminant: Δ = {b}^2 - 4({a})({c}) = {discriminant}',
                'Discriminant',
            )

            sqrt_d = sp.sqrt(discriminant)
            solutions = [
                sp.simplify((-b - sqrt_d) / (2 * a)),
                sp.simplify((-b + sqrt_d) / (2 * a)),
            ]

            if discriminant < 0:
                self._track_step(
                    solutions[0],
                    f'{var} has complex roots using the quadratic formula',
                    'Quadratic formula',
                )
            else:
                self._track_step(
                    solutions[0],
                    f'{var}_1 = (-{b} - sqrt({discriminant})) / (2*{a}) = {solutions[0]}',
                    'Quadratic formula',
                )
                self._track_step(
                    solutions[1],
                    f'{var}_2 = (-{b} + sqrt({discriminant})) / (2*{a}) = {solutions[1]}',
                    'Quadratic formula',
                )

            return solutions

        solutions = sp.solve(expr, var)
        if solutions:
            self._track_step(
                Eq(var, solutions[0]),
                f'Solved using SymPy: {var} = {solutions[0]}',
                'Polynomial solving',
            )
        return solutions

    def _solve_trig(self, expr: sp.Expr, var: sp.Symbol) -> List:
        """Solve trigonometric equations with explicit identity steps."""
        working = expr

        if working.has(sp.tan):
            rewritten = sp.trigsimp(sp.rewrite(working, sp.sin))
            if rewritten != working:
                self._track_step(
                    rewritten,
                    'Rewrite tangent using the quotient identity: tan(θ) = sin(θ)/cos(θ)',
                    'Quotient identity',
                )
                working = rewritten

        if working.has(sp.sec):
            rewritten = sp.trigsimp(sp.rewrite(working, sp.sin))
            if rewritten != working:
                self._track_step(
                    rewritten,
                    'Rewrite secant using 1/cos(θ)',
                    'Reciprocal identity',
                )
                working = rewritten

        pyth_sub = self._apply_pythagorean_identity(working, var)
        if pyth_sub is not None and pyth_sub != working:
            self._track_step(
                pyth_sub,
                'Apply the Pythagorean identity: sin²(θ) + cos²(θ) = 1',
                'Pythagorean identity',
            )
            working = sp.trigsimp(pyth_sub)

        double_angle = self._apply_double_angle_identity(working, var)
        if double_angle is not None and double_angle != working:
            self._track_step(
                double_angle,
                'Apply a double-angle identity to simplify the expression',
                'Double-angle identity',
            )
            working = sp.trigsimp(double_angle)

        quotient_step = self._try_quotient_reduction(working, var)
        if quotient_step is not None:
            reduced, description = quotient_step
            self._track_step(reduced, description, 'Quotient identity')
            working = sp.trigsimp(reduced)

        simplified = sp.trigsimp(working)
        if simplified != working:
            self._track_step(simplified, 'Simplify using standard trigonometric identities', 'Trig simplify')
            working = simplified

        try:
            solutions = sp.solve(working, var)
        except Exception:
            solutions = []

        if solutions:
            self._track_step(
                Eq(var, solutions[0]),
                f'Apply inverse trigonometric functions: {var} = {solutions[0]}',
                'Inverse trig',
            )
            if len(solutions) > 1:
                for sol in solutions[1:]:
                    self._track_step(
                        Eq(var, sol),
                        f'Additional solution from periodicity: {var} = {sol}',
                        'General solution',
                    )
        return solutions

    def _apply_pythagorean_identity(self, expr: sp.Expr, var: sp.Symbol) -> Optional[sp.Expr]:
        sin_var = sp.sin(var)
        cos_var = sp.cos(var)
        replacement = 1 - sin_var**2
        if expr.has(cos_var**2):
            return expr.replace(cos_var**2, replacement)
        replacement = 1 - cos_var**2
        if expr.has(sin_var**2):
            return expr.replace(sin_var**2, replacement)
        return None

    def _apply_double_angle_identity(self, expr: sp.Expr, var: sp.Symbol) -> Optional[sp.Expr]:
        sin_2var = sp.sin(2 * var)
        if expr.has(sin_2var):
            return expr.replace(sin_2var, 2 * sp.sin(var) * sp.cos(var))
        cos_2var = sp.cos(2 * var)
        if expr.has(cos_2var):
            return expr.replace(cos_2var, sp.cos(var) ** 2 - sp.sin(var) ** 2)
        return None

    def _try_quotient_reduction(
        self, expr: sp.Expr, var: sp.Symbol
    ) -> Optional[tuple[sp.Expr, str]]:
        """Reduce sin/cos mixtures to tan when both appear in a sum."""
        sin_var = sp.sin(var)
        cos_var = sp.cos(var)
        if not (expr.has(sin_var) and expr.has(cos_var)) or not expr.is_Add:
            return None

        sin_term = None
        cos_coeff = None
        for term in expr.args:
            if term.has(sin_var) and not term.has(cos_var):
                sin_term = term
            elif term.has(cos_var) and not term.has(sin_var):
                cos_coeff = term.coeff(cos_var)

        if sin_term is None or cos_coeff is None:
            return None

        tan_value = sp.simplify(-cos_coeff / sin_term.coeff(sin_var))
        tan_eq = sp.tan(var) - tan_value
        description = (
            f'Divide both sides by cos({var}) (cos({var}) ≠ 0): '
            f'sin({var})/cos({var}) = {tan_value} → tan({var}) = {tan_value}'
        )
        return tan_eq, description

    def _solve_log_exp(self, expr: sp.Expr, var: sp.Symbol) -> List:
        """Solve logarithmic and exponential equations with explicit log rules."""
        working = expr

        combined = sp.logcombine(working, force=True)
        if combined != working:
            self._track_step(
                combined,
                'Combine logarithms: log(a) + log(b) = log(ab) and n·log(a) = log(aⁿ)',
                'Log product/power rule',
            )
            working = combined

        expanded = sp.expand_log(working, force=True)
        if expanded != working:
            self._track_step(
                expanded,
                'Expand logarithms using log(a/b) = log(a) − log(b) or log(aⁿ) = n·log(a)',
                'Log expansion',
            )
            working = expanded

        exp_solution = self._try_solve_exponential(working, var)
        if exp_solution is not None:
            return exp_solution

        log_solution = self._try_solve_logarithm(working, var)
        if log_solution is not None:
            return log_solution

        try:
            solutions = sp.solve(working, var)
            if solutions:
                self._track_step(
                    Eq(var, solutions[0]),
                    f'Solve using inverse logarithm/exponential: {var} = {solutions[0]}',
                    'Log/exp solving',
                )
            return solutions
        except Exception:
            return []

    def _try_solve_exponential(self, expr: sp.Expr, var: sp.Symbol) -> Optional[List]:
        """Handle a**x = b style equations."""
        if not expr.has(var):
            return None

        for atom in expr.atoms(sp.Pow):
            if atom.has(var) and atom.base.is_number and atom.base > 0 and expr.is_Add:
                base = atom.base
                other = sp.simplify(expr - atom)
                if not other.has(var):
                    rhs_val = sp.simplify(-other)
                    self._track_step(
                        atom,
                        f'Isolate exponential term: {base}^({var}) = {rhs_val}',
                        'Isolate exponential',
                    )
                    if rhs_val <= 0:
                        return []
                    log_step = sp.log(rhs_val) / sp.log(base)
                    self._track_step(
                        log_step,
                        f'Take log base {base} of both sides: {var} = log({rhs_val})/log({base})',
                        'Log both sides',
                    )
                    solved = sp.simplify(log_step)
                    self._track_step(
                        solved,
                        f'Simplify: {var} = {solved}',
                        'Exponential rule',
                    )
                    return [solved]
        return None

    def _try_solve_logarithm(self, expr: sp.Expr, var: sp.Symbol) -> Optional[List]:
        """Handle log(f(x)) = c  →  f(x) = e^c."""
        if not expr.has(sp.log):
            return None

        log_atoms = list(expr.atoms(sp.log))
        if len(log_atoms) == 1 and expr.is_Add:
            log_term = log_atoms[0]
            other = sp.simplify(expr - log_term)
            if not other.has(sp.log) and not other.has(var):
                rhs = sp.simplify(-other)
                self._track_step(
                    log_term,
                    f'Isolate logarithm: log(...) = {rhs}',
                    'Isolate logarithm',
                )
                inner = log_term.args[0]
                exponentiated = sp.Eq(inner, sp.exp(rhs))
                self._track_step(
                    exponentiated,
                    f'Convert to exponential form: if log(f({var})) = {rhs}, then f({var}) = e^({rhs})',
                    'Definition of natural log',
                )
                try:
                    solutions = sp.solve(exponentiated.lhs - exponentiated.rhs, var)
                    if solutions:
                        self._track_step(
                            Eq(var, solutions[0]),
                            f'Solve the resulting equation: {var} = {solutions[0]}',
                            'Algebraic solve',
                        )
                    return solutions
                except Exception:
                    return None
        return None

    def partial_fractions(self, expression: str, variable: str = 'x') -> Dict:
        """Decompose a rational expression with partial-fraction steps."""
        self._begin_problem()
        expr = parse_math(expression)
        var = symbols(variable)

        self._track_step(expr, f'Rational expression: {expression}', 'Setup')

        together = sp.together(expr)
        if together != expr:
            self._track_step(together, 'Write as a single rational fraction', 'Common denominator')

        numer, denom = sp.fraction(sp.together(expr))
        self._track_step(
            sp.Mul(numer, 1 / denom, evaluate=False),
            f'Numerator: {numer}, denominator: {denom}',
            'Identify rational form',
        )

        if sp.degree(numer, var) >= sp.degree(denom, var):
            poly, remainder = sp.div(numer, denom, var)
            if poly != 0:
                self._track_step(
                    poly + remainder / denom,
                    'Polynomial division: separate polynomial part before partial fractions',
                    'Polynomial division',
                )
                expr = remainder / denom

        factored_denom = sp.factor(sp.fraction(sp.together(expr))[1])
        self._track_step(
            factored_denom,
            f'Factor the denominator: {factored_denom}',
            'Factor denominator',
        )

        decomposed = sp.apart(sp.together(expr), var)
        self._track_step(
            decomposed,
            'Decompose into partial fractions with undetermined coefficients',
            'Partial fraction decomposition',
        )

        final_form = sp.together(decomposed)
        if final_form != decomposed:
            self._track_step(final_form, 'Combine into a single expression over common factors', 'Simplify')

        return {
            'steps': self.step_tracker.get_solution(),
            'decomposition': decomposed,
            'decomposition_latex': sp.latex(decomposed),
        }

    def _solve_general(self, expr: sp.Expr, var: sp.Symbol) -> List:
        try:
            solutions = sp.solve(expr, var)
            if solutions:
                self._track_step(
                    Eq(var, solutions[0]),
                    f'Solved: {var} = {solutions[0]}',
                    'General solving',
                )
            return solutions
        except Exception:
            return []

    # ============ DIFFERENTIATION ============

    def differentiate(self, expression: str, variable: str = 'x', order: int = 1) -> Dict:
        self._begin_problem()
        expr = parse_math(expression)
        var = symbols(variable)

        self._track_step(expr, f'Original function: f({variable}) = {expression}', 'Setup')

        result = expr
        for i in range(order):
            result = self._differentiate_with_steps(result, var, step_number=i + 1)

        return {
            'steps': self.step_tracker.get_solution(),
            'derivative': result,
            'derivative_latex': sp.latex(result),
        }

    def _differentiate_with_steps(self, expr: sp.Expr, var: sp.Symbol, step_number: int) -> sp.Expr:
        if expr.is_Add:
            self._track_step(expr, f'Step {step_number}: Apply sum rule to each term', 'Sum rule')
            derivatives = []
            for term in expr.args:
                term_derivative = sp.diff(term, var)
                rule_used = self._identify_derivative_rule(term, var)
                self._track_step(
                    term_derivative,
                    f'd/d{var}({term}) using {rule_used}',
                    rule_used,
                )
                derivatives.append(term_derivative)
            combined = sp.simplify(sp.Add(*derivatives))
            self._track_step(combined, f'Combine terms: f^({step_number})({var}) = {combined}', 'Simplify')
            return combined

        derivative = sp.diff(expr, var)
        rule_used = self._identify_derivative_rule(expr, var)
        self._track_step(
            derivative,
            f'Step {step_number}: Apply {rule_used}',
            rule_used,
        )
        return derivative

    def _identify_derivative_rule(self, expr: sp.Expr, var: sp.Symbol) -> str:
        if expr.is_Pow:
            if expr.base == var:
                return 'Power rule'
            return 'Chain rule'
        if expr.is_Mul:
            return 'Product rule'
        if expr.is_Add:
            return 'Sum rule'
        if isinstance(expr, sp.Function):
            return 'Chain rule'
        return 'Differentiation'

    # ============ INTEGRATION ============

    def integrate(
        self,
        expression: str,
        variable: str = 'x',
        lower: Optional[str] = None,
        upper: Optional[str] = None,
    ) -> Dict:
        self._begin_problem()
        expr = parse_math(expression)
        var = symbols(variable)

        self._track_step(expr, f'Original function: f({variable}) = {expression}', 'Setup')

        result = None
        methods_used: List[str] = []

        if self._is_proper_rational(expr, var):
            result = self._integrate_rational_with_partials(expr, var)
            if result is not None:
                methods_used.append('Partial fractions')

        if result is None and len(self._integration_factors(expr)) >= 2:
            parts_result = self._integrate_by_parts(expr, var)
            if parts_result is not None:
                result = parts_result
                methods_used.append('Integration by parts')

        if result is None:
            try:
                candidate = sp.integrate(expr, var)
                if candidate != expr:
                    result = candidate
                    self._track_step(result, 'Apply standard integration rules', 'Direct integration')
                    methods_used.append('Direct integration')
            except Exception:
                result = None

        if result is None:
            substitution_result = self._integrate_by_substitution(expr, var)
            if substitution_result is not None:
                result = substitution_result
                methods_used.append('Substitution')

        if result is None:
            parts_result = self._integrate_by_parts(expr, var)
            if parts_result is not None:
                result = parts_result
                methods_used.append('Integration by parts')

        if lower is not None and upper is not None:
            lower_val = float(lower)
            upper_val = float(upper)
            definite = sp.integrate(expr, (var, lower_val, upper_val))
            self._track_step(
                definite,
                f'Evaluate from {lower} to {upper}',
                'Definite integration',
            )
            result = definite

        return {
            'steps': self.step_tracker.get_solution(),
            'integral': result,
            'methods': methods_used,
            'integral_latex': sp.latex(result) if result is not None else 'Unable to integrate',
        }

    def _integration_factors(self, expr: sp.Expr) -> List[sp.Expr]:
        if expr.is_Mul:
            return list(expr.args)
        return [expr]

    def _is_proper_rational(self, expr: sp.Expr, var: sp.Symbol) -> bool:
        try:
            numer, denom = sp.fraction(sp.together(expr))
            return (
                numer.is_polynomial(var)
                and denom.is_polynomial(var)
                and sp.degree(numer, var) < sp.degree(denom, var)
                and sp.degree(denom, var) > 0
            )
        except Exception:
            return False

    def _integrate_rational_with_partials(self, expr: sp.Expr, var: sp.Symbol) -> Optional[sp.Expr]:
        together = sp.together(expr)
        numer, denom = sp.fraction(together)
        self._track_step(
            together,
            f'Proper rational function: ({numer}) / ({denom})',
            'Rational form',
        )

        factored_denom = sp.factor(denom)
        self._track_step(
            factored_denom,
            f'Factor the denominator: {factored_denom}',
            'Factor denominator',
        )

        decomposed = sp.apart(together, var)
        self._track_step(
            decomposed,
            'Decompose into partial fractions before integrating',
            'Partial fraction decomposition',
        )

        integrated = sp.integrate(decomposed, var)
        self._track_step(
            integrated,
            'Integrate each partial fraction term separately',
            'Integrate terms',
        )
        return sp.simplify(integrated)

    def _integrate_by_substitution(self, expr: sp.Expr, var: sp.Symbol) -> Optional[sp.Expr]:
        """Simple u-substitution when the integrand is g(f(x))*f'(x) up to a constant."""
        if not expr.has(var):
            return None

        inner = None
        if expr.is_Mul:
            for factor in expr.args:
                if factor.is_Pow and factor.exp == -1 and factor.base.is_Add:
                    inner = factor.base
                    break

        if inner is None:
            return None

        derivative = sp.diff(inner, var)
        if not expr.has(derivative):
            return None

        u = sp.Symbol('u')
        self._track_step(
            inner,
            f'Substitution: let u = {inner}, so du = ({derivative}) d{var}',
            'u-substitution',
        )
        substituted = sp.simplify(expr / derivative)
        self._track_step(
            substituted,
            f'Rewrite the integral in terms of u: ∫ {substituted} du',
            'u-substitution',
        )
        integrated_u = sp.integrate(substituted, u)
        result = sp.simplify(integrated_u.subs(u, inner))
        self._track_step(
            result,
            f'Substitute back u = {inner}: result = {result}',
            'u-substitution',
        )
        return result

    def _integrate_by_parts(self, expr: sp.Expr, var: sp.Symbol) -> Optional[sp.Expr]:
        factors = self._integration_factors(expr)
        if len(factors) < 2:
            return None

        ranked = sorted((liate_rank(f, var), f) for f in factors)
        u = ranked[0][1]
        dv_factors = [f for f in factors if f != u]
        dv = sp.Mul(*dv_factors) if len(dv_factors) > 1 else dv_factors[0]
        rank = ranked[0][0]
        liate_label = LIATE_LABELS[rank] if 0 <= rank < len(LIATE_LABELS) else 'Algebraic'

        self._track_step(
            u,
            f'Choose u = {u} ({liate_label} term has LIATE priority for u)',
            'LIATE: choose u',
        )
        self._track_step(
            dv,
            f'Choose dv = {dv} d{var} (remaining factor)',
            'Integration by parts setup',
        )

        du = sp.diff(u, var)
        self._track_step(
            du,
            f'Differentiate u: du = {du} d{var}',
            'Differentiate u',
        )

        v = sp.integrate(dv, var)
        if v == 0 and dv != 0:
            return None
        self._track_step(
            v,
            f'Integrate dv: v = ∫ {dv} d{var} = {v}',
            'Integrate dv',
        )

        uv = sp.expand(u * v)
        self._track_step(
            uv,
            f'Apply formula ∫ u dv = u·v − ∫ v du; first term u·v = {uv}',
            'Integration by parts',
        )

        remaining_integrand = sp.expand(v * du)
        self._track_step(
            remaining_integrand,
            f'Remaining integral: ∫ v du = ∫ {remaining_integrand} d{var}',
            'Integration by parts',
        )

        remaining = sp.integrate(remaining_integrand, var)
        self._track_step(
            remaining,
            f'Evaluate ∫ v du = {remaining}',
            'Integrate remainder',
        )

        result = sp.simplify(uv - remaining)
        self._track_step(
            result,
            f'Final result: u·v − ∫ v du = {result}',
            'Integration by parts result',
        )
        return result

    # ============ LIMITS ============

    def limit(
        self,
        expression: str,
        variable: str = 'x',
        point: str = '0',
        direction: str = '+',
    ) -> Dict:
        self._begin_problem()
        expr = parse_math(expression)
        var = symbols(variable)
        point_val = parse_math(point)

        self._track_step(expr, f'Original expression: {expression}', 'Setup')
        self._track_step(
            expr,
            f'Find limit as {variable} -> {point} (direction: {direction})',
            'Limit setup',
        )

        direct = expr.subs(var, point_val)
        if direct.is_finite and direct not in (sp.zoo, sp.nan):
            self._track_step(direct, f'Direct substitution: {direct}', 'Direct substitution')

        result = sp.limit(expr, var, point_val, dir=direction)
        self._track_step(result, f'Limit = {result}', 'Final result')

        return {
            'steps': self.step_tracker.get_solution(),
            'limit': result,
            'limit_latex': sp.latex(result),
        }

    # ============ LINEAR ALGEBRA ============

    def _normalize_equations(self, equations) -> List[str]:
        if isinstance(equations, str):
            parts = re.split(r'[;\n]+', equations)
            return [part.strip() for part in parts if part.strip()]
        return [str(eq).strip() for eq in equations if str(eq).strip()]

    def solve_linear_system(self, equations, variables) -> Dict:
        """
        Solve a linear system using Gaussian elimination with row-operation steps.

        Example:
        >>> solver = SmartSolver()
        >>> result = solver.solve_linear_system(
        ...     ["2x + 3y = 7", "x - y = 1"],
        ...     ["x", "y"],
        ... )
        """
        from .linear_algebra import solve_linear_system_with_steps

        self._begin_problem()
        eqs = self._normalize_equations(equations)
        var_list = [str(v).strip() for v in variables]
        solution, rref, syms, consistent = solve_linear_system_with_steps(
            eqs,
            var_list,
            self._track_step,
        )

        if not consistent:
            self._track_step(
                sp.Integer(0),
                'The system is inconsistent (parallel lines / no solution).',
                'No solution',
            )
            return {
                'steps': self.step_tracker.get_solution(),
                'solutions': [],
                'solutions_by_variable': {},
                'solution_latex': 'No solution',
                'rref': rref,
                'rref_latex': sp.latex(rref),
                'consistent': False,
            }

        if not solution:
            self._track_step(
                sp.Integer(0),
                'The system has infinitely many solutions (underdetermined).',
                'Infinite solutions',
            )
            return {
                'steps': self.step_tracker.get_solution(),
                'solutions': [],
                'solutions_by_variable': {},
                'solution_latex': 'Infinitely many solutions',
                'rref': rref,
                'rref_latex': sp.latex(rref),
                'consistent': True,
            }

        by_variable = {str(sym): sp.simplify(val) for sym, val in zip(syms, solution)}
        for sym, val in by_variable.items():
            self._track_step(
                sp.Eq(symbols(sym), val),
                f'Solution: {sym} = {val}',
                'Back substitution',
            )

        return {
            'steps': self.step_tracker.get_solution(),
            'solutions': solution,
            'solutions_by_variable': {k: str(v) for k, v in by_variable.items()},
            'solution_latex': sp.latex(sp.Matrix(solution)),
            'rref': rref,
            'rref_latex': sp.latex(rref),
            'consistent': True,
        }

    def matrix_rref(self, matrix) -> Dict:
        """Row-reduce a matrix to RREF with elementary row operation steps."""
        from .linear_algebra import matrix_rref_with_steps

        self._begin_problem()
        rref = matrix_rref_with_steps(matrix, self._track_step)
        return {
            'steps': self.step_tracker.get_solution(),
            'matrix': rref,
            'matrix_latex': sp.latex(rref),
        }

    # ============ MULTIVARIABLE CALCULUS ============

    def partial_derivative(
        self,
        expression: str,
        wrt: str,
        variables: Optional[List[str]] = None,
        order: int = 1,
    ) -> Dict:
        """
        Partial derivative with respect to one variable, holding others constant.

        Example:
        >>> SmartSolver().partial_derivative("x*y + y^2", "x", ["x", "y"])
        """
        from .multivariable import partial_derivative_with_steps

        if order < 1:
            raise ValueError('order must be at least 1.')

        self._begin_problem()
        result, _ = partial_derivative_with_steps(
            expression,
            wrt,
            variables,
            order,
            self._track_step,
            self._identify_derivative_rule,
        )
        return {
            'steps': self.step_tracker.get_solution(),
            'derivative': result,
            'derivative_latex': sp.latex(result),
        }

    def gradient(self, expression: str, variables: List[str]) -> Dict:
        """Gradient vector of partial derivatives."""
        from .multivariable import gradient_with_steps

        self._begin_problem()
        result, _ = gradient_with_steps(
            expression,
            variables,
            self._track_step,
            self._identify_derivative_rule,
        )
        return {
            'steps': self.step_tracker.get_solution(),
            'gradient': result,
            'gradient_latex': sp.latex(result),
        }

    def integrate_multivariable(
        self,
        expression: str,
        variables: List[str],
        bounds: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """
        Multiple integration over listed variables (definite bounds optional).

        Example:
        >>> SmartSolver().integrate_multivariable(
        ...     "x*y",
        ...     ["x", "y"],
        ...     {"x": (0, 1), "y": (0, 2)},
        ... )
        """
        from .multivariable import multiple_integral_with_steps

        self._begin_problem()
        result, _ = multiple_integral_with_steps(
            expression,
            variables,
            bounds,
            self._track_step,
        )
        return {
            'steps': self.step_tracker.get_solution(),
            'integral': result,
            'integral_latex': sp.latex(result),
        }


class StepRenderer:
    """Render steps in various formats."""

    @staticmethod
    def to_text(steps: List[Step]) -> str:
        output = []
        for i, step in enumerate(steps, 1):
            output.append(f'Step {i}: {step.description}')
            output.append(f'  -> {step.expression}')
            if step.rule_applied:
                output.append(f'  Rule: {step.rule_applied}')
            output.append('')
        return '\n'.join(output)

    @staticmethod
    def to_latex(steps: List[Step]) -> str:
        latex_parts = []
        for i, step in enumerate(steps, 1):
            latex_parts.append(f'\\text{{Step {i}: }} {step.latex}')
            if step.rule_applied:
                latex_parts.append(f'\\quad \\text{{({step.rule_applied})}}')
        return ' \\\\ '.join(latex_parts)

    @staticmethod
    def to_html(steps: List[Step]) -> str:
        html = ["<div class='solution-steps'>"]
        for i, step in enumerate(steps, 1):
            html.append("<div class='step'>")
            html.append(f"<div class='step-number'>Step {i}</div>")
            html.append(f"<div class='step-desc'>{step.description}</div>")
            html.append(f"<div class='step-expression'>{step.expression}</div>")
            if step.rule_applied:
                html.append(f"<div class='step-rule'>Rule: {step.rule_applied}</div>")
            html.append('</div>')
        html.append('</div>')
        return '\n'.join(html)


def serialize_solver_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a SmartSolver result into JSON-serializable data."""
    steps = result.get('steps', [])
    payload: Dict[str, Any] = {}

    for key, value in result.items():
        if key == 'steps':
            payload['steps'] = [step_to_dict(step) for step in value]
        elif key in ('derivative', 'integral', 'limit', 'decomposition', 'gradient'):
            payload[key] = str(value) if value is not None else None
        elif key in ('matrix', 'rref'):
            payload[key] = str(value) if value is not None else None
        elif key == 'solutions':
            payload['solutions'] = [str(s) for s in value]
        elif key == 'solutions_by_variable' and isinstance(value, dict):
            payload[key] = value
        else:
            payload[key] = value

    if 'derivative' in result:
        payload['result'] = str(result['derivative'])
        payload['result_latex'] = result.get('derivative_latex')
    elif 'gradient' in result:
        payload['result'] = str(result['gradient'])
        payload['result_latex'] = result.get('gradient_latex')
    elif 'integral' in result:
        payload['result'] = str(result['integral']) if result['integral'] is not None else None
        payload['result_latex'] = result.get('integral_latex')
    elif 'matrix' in result:
        payload['result'] = str(result['matrix'])
        payload['result_latex'] = result.get('matrix_latex')
    elif 'decomposition' in result:
        payload['result'] = str(result['decomposition'])
        payload['result_latex'] = result.get('decomposition_latex')
    elif 'limit' in result:
        payload['result'] = str(result['limit'])
        payload['result_latex'] = result.get('limit_latex')
    elif 'solutions_by_variable' in result and result.get('solutions_by_variable'):
        payload['result'] = result['solutions_by_variable']
        payload['result_latex'] = result.get('solution_latex')
    elif 'solutions' in result:
        payload['result'] = [str(s) for s in result['solutions']]
        payload['result_latex'] = result.get('solution_latex')

    payload['steps_text'] = StepRenderer.to_text(steps)
    payload['steps_html'] = StepRenderer.to_html(steps)
    payload['steps_latex'] = StepRenderer.to_latex(steps)
    return payload
