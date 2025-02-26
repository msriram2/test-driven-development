from SignalDetection import SignalDetection
import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore

class Experiment:
    """Initializes self.conditions. self.conditions is a list containing signal-detection theory
    objects.
    
    Accepts no parameters and returns nothing.
    """
    def __init__(self):
        # Use a consistent name for the list of conditions.
        self.conditions = []  

    def add_condition(self, sig_det_obj: SignalDetection, label: str = None) -> None:
        self.conditions.append((sig_det_obj, label))

    def sorted_roc_points(self) -> tuple[list[float], list[float]]:
        if not self.conditions:
            raise ValueError("No conditions have been added to the experiment.")
        
        # Note: use sdt.fa_rate() since that’s the method defined in SignalDetection.
        false_alarm_rates = [sdt.fa_rate() for sdt, _ in self.conditions]
        hit_rates = [sdt.hit_rate() for sdt, _ in self.conditions]
        
        sorted_indices = np.argsort(false_alarm_rates)
        sorted_false_alarm_rates = [false_alarm_rates[i] for i in sorted_indices]
        sorted_hit_rates = [hit_rates[i] for i in sorted_indices]
        
        return sorted_false_alarm_rates, sorted_hit_rates

    def compute_auc(self) -> float:
        if not self.conditions:
            raise ValueError("No conditions available to compute AUC.")
        
        false_alarm_rates, hit_rates = self.sorted_roc_points()
        return np.trapz(hit_rates, false_alarm_rates)
    
    def plot_roc_curve(self, show_plot: bool = True):
        false_alarm_rates, hit_rates = self.sorted_roc_points()
        
        plt.figure(figsize=(6, 6))
        plt.plot(false_alarm_rates, hit_rates, marker='o', linestyle='-', label='ROC Curve')
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Chance Level')
        plt.xlabel("False Alarm Rate")
        plt.ylabel("Hit Rate")
        plt.title("ROC Curve")
        plt.legend()
        
        if show_plot:
            plt.show()
