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

    def test_linear_system_gaussian_elimination(self):
        result = SmartSolver().solve_linear_system(
            ["2x + 3y = 7", "x - y = 1"],
            ["x", "y"],
        )
        rules = [step.rule_applied for step in result["steps"]]
        self.assertIn("Augmented matrix", rules)
        self.assertIn("Row elimination", rules)
        self.assertIn("Reduced row echelon form", rules)
        self.assertIn("Back substitution", rules)
        self.assertEqual(result["solutions_by_variable"], {"x": "2", "y": "1"})

    def test_matrix_rref(self):
        result = SmartSolver().matrix_rref("Matrix([[2, 4, 6], [1, 3, 5]])")
        rules = [step.rule_applied for step in result["steps"]]
        self.assertIn("Setup", rules)
        self.assertIn("Reduced row echelon form", rules)
        self.assertIn("1", str(result["matrix"]))

    def test_partial_derivative(self):
        result = SmartSolver().partial_derivative("x*y + y^2", "x", ["x", "y"])
        rules = [step.rule_applied for step in result["steps"]]
        self.assertIn("Partial derivative setup", rules)
        self.assertEqual(str(result["derivative"]), "y")

    def test_gradient(self):
        result = SmartSolver().gradient("x^2 + x*y", ["x", "y"])
        rules = [step.rule_applied for step in result["steps"]]
        self.assertIn("Gradient setup", rules)
        self.assertIn("2*x + y", str(result["gradient"][0]))
        self.assertIn("x", str(result["gradient"][1]))

    def test_integrate_multivariable(self):
        result = SmartSolver().integrate_multivariable(
            "x*y",
            ["x", "y"],
            {"x": (0, 1), "y": (0, 2)},
        )
        rules = [step.rule_applied for step in result["steps"]]
        self.assertIn("Definite integration", rules)
        self.assertEqual(str(result["integral"]), "1")


if __name__ == "__main__":
    unittest.main()
