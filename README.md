# SmartSolver

**SmartSolver** is a step-by-step mathematics engine built on [SymPy](https://www.sympy.org/). It powers worked solutions inside [Ed-Master](https://www.ed-master.co.za) — showing not just the final answer, but the rules and identities applied along the way.

The implementation lives in `math_step_tracker.py` (class `SmartSolver`).

---

## Features

| Area | What SmartSolver shows |
|------|------------------------|
| **Equations** | Linear & quadratic solving, discriminant steps |
| **Trigonometry** | Quotient, Pythagorean, double-angle identities; tan reduction |
| **Logarithms & exponentials** | Log product/power rules, `log(x) = c → x = e^c`, `a^x = b` |
| **Differentiation** | Sum, product, power, chain rule labels |
| **Integration** | Partial fractions, LIATE-based integration by parts, direct rules |
| **Limits** | Setup, direct substitution where valid, final limit |
| **Partial fractions** | Factor denominator, decompose, integrate term-by-term |

Each step includes a **description**, **expression**, **LaTeX**, and **rule applied**.

---

## Requirements

- **Python 3.10+** (`requires-python` in `pyproject.toml`)
- **SymPy ≥ 1.13**

### Supported Python versions

| Status | Versions |
|--------|----------|
| Minimum | Python **3.10** |
| Tested | Python **3.10**, **3.11**, **3.12**, **3.13** |

Only list versions you actually test. Do **not** claim support for unreleased major versions (e.g. Python 4.x or fictional 7.x).

```bash
pip install edmaster-smartsolver
```

Or install SymPy only when using from source:

```bash
pip install sympy
```

---

## Quick start

```python
from smartsolver import SmartSolver, StepRenderer, serialize_solver_result

solver = SmartSolver()

# Solve a quadratic
result = solver.solve_equation("x^2 + 5x + 6 = 0", "x")
print(StepRenderer.to_text(result["steps"]))
print("Solutions:", result["solutions"])

# JSON-friendly payload (for APIs)
payload = serialize_solver_result(result)
print(payload["steps_latex"])
```

When running from a git checkout (editable install):

```bash
pip install -e .
```

```python
from smartsolver import SmartSolver
```

---

## Input notation

SmartSolver accepts student-style strings:

| You type | Parsed as |
|----------|-----------|
| `x^2` or `x**2` | x squared |
| `5x` | `5*x` |
| `2x - 4` | `2x - 4 = 0` (equation assumed zero) |
| `sin(x)`, `cos(x)`, `log(x)` | SymPy trig / natural log |
| `2**x` | exponential |

---

## Operations

### `solve_equation(equation, variable='x')`

Solve an equation with step-by-step working.

```python
solver.solve_equation("sin(x) - cos(x) = 0", "x")
solver.solve_equation("2**x - 8 = 0", "x")
solver.solve_equation("log(x) - 3 = 0", "x")
```

**Returns:** `steps`, `solutions`, `solution_latex`

---

### `differentiate(expression, variable='x', order=1)`

Differentiate with rule labels (sum, power, chain, etc.).

```python
solver.differentiate("x**2 + 3*x", "x")
solver.differentiate("sin(x)**2", "x", order=1)
```

**Returns:** `steps`, `derivative`, `derivative_latex`

---

### `integrate(expression, variable='x', lower=None, upper=None)`

Integrate with method tracking.

```python
solver.integrate("x*exp(x)", "x")           # integration by parts (LIATE)
solver.integrate("1/(x**2 - 1)", "x")      # partial fractions
solver.integrate("x**2", "x", lower="0", upper="1")  # definite
```

**Returns:** `steps`, `integral`, `integral_latex`, `methods` (e.g. `['Integration by parts']`)

---

### `partial_fractions(expression, variable='x')`

Decompose a rational expression without integrating.

```python
solver.partial_fractions("(2*x + 3)/((x - 1)*(x + 2))")
```

**Returns:** `steps`, `decomposition`, `decomposition_latex`

---

### `limit(expression, variable='x', point='0', direction='+')`

Evaluate a limit with setup steps.

```python
solver.limit("sin(x)/x", "x", point="0", direction="+")
```

**Returns:** `steps`, `limit`, `limit_latex`

---

## Helper utilities

```python
from smartsolver import (
    StepRenderer,
    serialize_solver_result,
    normalize_math_input,
    parse_math,
    step_to_dict,
)

# Human-readable steps
StepRenderer.to_text(steps)
StepRenderer.to_latex(steps)
StepRenderer.to_html(steps)

# API / JSON serialization
serialize_solver_result(result)
```

### Optional helpers (Ed-Master Math Lab)

If you also ship `edmathlab.py` alongside this package, it provides stdout-friendly wrappers. They are not included in the PyPI wheel by default.

---

## REST API (Ed-Master platform)

When the Django backend is running, authenticated users can call:

```
POST /api/math/steps/
Content-Type: application/json
```

**Body:**

```json
{
  "operation": "solve",
  "expression": "x^2 + 5x + 6 = 0",
  "variable": "x"
}
```

**Operations:** `solve`, `differentiate`, `integrate`, `limit`, `partial_fractions`

**Response fields:** `steps`, `steps_text`, `steps_html`, `steps_latex`, `result`, `result_latex`, `operation`

Requires a signed-in Ed-Master session (JWT cookie).

---

## Result structure

Each step is a `Step` dataclass:

```python
@dataclass
class Step:
    description: str   # e.g. "Apply the quotient identity"
    expression: str    # SymPy string at this step
    latex: str         # LaTeX rendering
    rule_applied: str  # e.g. "Quotient identity", "LIATE: choose u"
    substeps: list     # nested steps (reserved)
```

`serialize_solver_result()` flattens this for JSON APIs and adds rendered `steps_text`, `steps_html`, and `steps_latex`.

---

## Example session

```python
from smartsolver import SmartSolver, StepRenderer

s = SmartSolver()

print("=== Quadratic ===")
r = s.solve_equation("x^2 + 5x + 6 = 0")
print(StepRenderer.to_text(r["steps"]))

print("=== Trig ===")
r = s.solve_equation("sin(x) - cos(x) = 0")
print(StepRenderer.to_text(r["steps"]))

print("=== Integration by parts ===")
r = s.integrate("x*exp(x)")
print(StepRenderer.to_text(r["steps"]))
print("Answer:", r["integral"])
print("Methods:", r["methods"])
```

---

## Scope & limitations

SmartSolver is designed for **education**, not as a replacement for a full computer algebra system.

- Trig: covers common identities and reduction; general periodic solution sets are simplified.
- Logs: strongest on combinable logs and `log(f(x)) = constant` forms.
- Partial fractions: requires a proper rational form; improper rationals use polynomial division first.
- Integration by parts: one LIATE-guided pass; does not recurse automatically.
- u-substitution: basic patterns only.

SymPy still performs the underlying symbolic work; SmartSolver adds **annotated steps** around it.

---

## Running tests

```bash
pip install -e ".[dev]"
python -m unittest discover -s tests -v
```

Or a quick smoke test:

```bash
python -c "from smartsolver import SmartSolver; print(SmartSolver().solve_equation('x-2=0')['solutions'])"
```

---

## Project layout

```
.                            # repository root (this folder)
├── __init__.py              # public exports
├── math_step_tracker.py     # SmartSolver core
├── README.md
├── LICENSE
├── pyproject.toml           # PEP 517 build (no setup.py)
└── tests/
    └── test_smartsolver.py
```

---

## Roadmap

- [x] PyPI packaging scaffold (`edmaster-smartsolver`)
- [ ] Recursive integration by parts
- [ ] Richer trig general solutions
- [ ] Public “Try SmartSolver” demo page on Ed-Master

---

## Links

- **Platform:** [ed-master.co.za](https://www.ed-master.co.za)
- **Try Math Lab:** [ed-master.co.za/try/math-lab](https://www.ed-master.co.za/try/math-lab)
- **Research projects:** [ed-master.co.za/projects](https://www.ed-master.co.za/projects)

---

## License

MIT — see [LICENSE](LICENSE).
