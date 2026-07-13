"""Pure-Python tests for SmartSolver (no Django)."""

import unittest

from smartsolver import SmartSolver, StepRenderer, serialize_solver_result


class SmartSolverTests(unittest.TestCase):
    def test_quadratic_solve(self):
        result = SmartSolver().solve_equation("x^2 + 5x + 6 = 0", "x")
        solutions = sorted(int(s) for s in result["solutions"])
        self.assertEqual(solutions, [-3, -2])
        self.assertGreaterEqual(len(result["steps"]), 4)

    def test_trig_quotient_identity(self):
        result = SmartSolver().solve_equation("sin(x) - cos(x) = 0", "x")
        rules = [step.rule_applied for step in result["steps"]]
        self.assertIn("Quotient identity", rules)
        self.assertEqual(str(result["solutions"][0]), "pi/4")

    def test_exponential_solve(self):
        result = SmartSolver().solve_equation("2**x - 8 = 0", "x")
        rules = [step.rule_applied for step in result["steps"]]
        self.assertIn("Log both sides", rules)
        self.assertEqual(str(result["solutions"][0]), "3")

    def test_integration_by_parts_liate(self):
        result = SmartSolver().integrate("x*exp(x)", "x")
        rules = [step.rule_applied for step in result["steps"]]
        self.assertIn("LIATE: choose u", rules)
        self.assertEqual(str(result["integral"]), "(x - 1)*exp(x)")

    def test_partial_fractions(self):
        result = SmartSolver().partial_fractions("(2*x + 3)/((x - 1)*(x + 2))")
        rules = [step.rule_applied for step in result["steps"]]
        self.assertIn("Partial fraction decomposition", rules)
        self.assertIn("1/(3*(x + 2))", str(result["decomposition"]))

    def test_serialize_result(self):
        raw = SmartSolver().solve_equation("x - 2 = 0", "x")
        payload = serialize_solver_result(raw)
        self.assertEqual(payload["result"], ["2"])
        self.assertTrue(payload["steps_text"])
        self.assertTrue(StepRenderer.to_latex(raw["steps"]))


if __name__ == "__main__":
    unittest.main()
