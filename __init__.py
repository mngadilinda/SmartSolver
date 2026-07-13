"""Ed-Master SmartSolver — step-by-step mathematics on SymPy."""

from smartsolver.math_step_tracker import (
    SmartSolver,
    Step,
    StepRenderer,
    StepTracker,
    contains_log_or_exp,
    contains_trig,
    liate_rank,
    normalize_math_input,
    parse_math,
    serialize_solver_result,
    solution_to_latex,
    step_to_dict,
)

__version__ = "0.1.0"

__all__ = [
    "SmartSolver",
    "Step",
    "StepRenderer",
    "StepTracker",
    "contains_log_or_exp",
    "contains_trig",
    "liate_rank",
    "normalize_math_input",
    "parse_math",
    "serialize_solver_result",
    "solution_to_latex",
    "step_to_dict",
    "__version__",
]
