import unittest
from Experiment import Experiment
from SignalDetection import SignalDetection

class TestExperiment(unittest.TestCase):

    def setUp(self):
        self.exp = Experiment()

    def test_add_condition(self):
        sdt_obj = SignalDetection(40, 10, 20, 30)
        self.exp.add_condition(sdt_obj, label="Condition A")
        self.assertEqual(len(self.exp.conditions), 1)
        self.assertEqual(self.exp.conditions[0][1], "Condition A")

    def test_add_condition_without_label(self):
        # Test that if no label is provided, the condition is stored with a None label.
        sdt_obj = SignalDetection(40, 10, 20, 30)
        self.exp.add_condition(sdt_obj)
        self.assertIsNone(self.exp.conditions[0][1])

    def test_sorted_roc_points(self):
        self.exp.add_condition(SignalDetection(40, 10, 20, 30), label="Condition A")
        self.exp.add_condition(SignalDetection(30, 20, 15, 35), label="Condition B")
        false_alarm_rate, hit_rate = self.exp.sorted_roc_points()
        self.assertEqual(false_alarm_rate, sorted(false_alarm_rate))
        self.assertEqual(len(false_alarm_rate), len(hit_rate))

    def test_compute_auc_two_points(self):
        # Test case for AUC = 0.5 with two points: (0,0) and (1,1)
        exp = Experiment()  # New instance to avoid interference
        # (0,0): hit_rate=0 (0/(0+1)) and false alarm rate=0 (0/(0+1))
        exp.add_condition(SignalDetection(0, 1, 0, 1))
        # (1,1): hit_rate=1 (1/(1+0)) and false alarm rate=1 (1/(1+0))
        exp.add_condition(SignalDetection(1, 0, 1, 0))
        self.assertAlmostEqual(exp.compute_auc(), 0.5, places=3)

    def test_compute_auc_three_points(self):
        # Test case for AUC = 1 with three points: (0,0), (0,1), and (1,1)
        exp = Experiment()  # Fresh instance
        # (0,0)
        exp.add_condition(SignalDetection(0, 1, 0, 1))
        # (0,1): hit_rate=1 (1/1) and false alarm rate=0 (0/1)
        exp.add_condition(SignalDetection(1, 0, 0, 1))
        # (1,1)
        exp.add_condition(SignalDetection(1, 0, 1, 0))
        self.assertAlmostEqual(exp.compute_auc(), 1.0, places=3)

    def test_empty_experiment(self):
        with self.assertRaises(ValueError):
            self.exp.sorted_roc_points()
        with self.assertRaises(ValueError):
            self.exp.compute_auc()
    
    def test_conditions_storage(self):
        # Test that multiple conditions are stored properly.
        sdt1 = SignalDetection(10, 5, 3, 7)
        sdt2 = SignalDetection(20, 10, 8, 12)
        self.exp.add_condition(sdt1, label="First")
        self.exp.add_condition(sdt2, label="Second")
        self.assertEqual(len(self.exp.conditions), 2)
        self.assertEqual(self.exp.conditions[0][1], "First")
        self.assertEqual(self.exp.conditions[1][1], "Second")

    def test_plot_roc_curve(self):
        """Test that plot_roc_curve runs without errors."""
        sdt = SignalDetection(50, 50, 50, 50)
        self.exp.add_condition(sdt, label="Test")
        try:
            self.exp.plot_roc_curve(show_plot=False)
        except Exception as e:
            self.fail(f"plot_roc_curve() raised an exception: {e}")

    def test_compute_auc_single_condition(self):
        # Test that when only one condition is added, compute_auc returns 0.0
        self.exp.add_condition(SignalDetection(40, 10, 20, 30), label="Single Condition")
        self.assertEqual(self.exp.compute_auc(), 0.0)

if __name__ == '__main__':
    unittest.main()

