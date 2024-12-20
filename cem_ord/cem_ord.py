import numpy as np
from .confusion_matrix import ConfusionMatrix


class CEMOrd:
    def __init__(self, gold_labels, system_labels):
        """
        Initialize the CEM-Ord metric with gold and system labels.
        
        :param gold_labels: List of gold standard labels.
        :param system_labels: List of system output labels.
        """
        if len(gold_labels) != len(system_labels):
            raise ValueError("Gold labels and system labels must have the same length.")

        self.gold_labels = gold_labels
        self.system_labels = system_labels
        self.confusion_matrix = ConfusionMatrix(gold_labels, system_labels)

    def evaluate(self):
        """
        Evaluate the system predictions using the CEM-Ord metric.
        
        :return: The overall CEM-Ord score.
        """
        sum_numerator = 0.0
        sum_denominator = 0.0

        # Iterate through gold and system labels to compute proximity
        for gold, system in zip(self.gold_labels, self.system_labels):
            sum_numerator += self.confusion_matrix.proximity(system, gold)
            sum_denominator += self.confusion_matrix.proximity(gold, gold)

        # Compute the final CEM-Ord score
        cem_ord_score = sum_numerator / sum_denominator if sum_denominator > 0 else 0.0
        return cem_ord_score
