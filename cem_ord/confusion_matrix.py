import numpy as np
import math
from collections import Counter


class ConfusionMatrix:
    def __init__(self, gold_labels, system_labels):
        """
        Initialize the confusion matrix.
        
        :param gold_labels: List of gold standard labels.
        :param system_labels: List of system output labels.
        """
        self.gold_labels = np.array(gold_labels)
        self.system_labels = np.array(system_labels)
        self.classes = sorted(set(self.gold_labels) | set(self.system_labels))
        self.class_indices = {cls: i for i, cls in enumerate(self.classes)}
        self.matrix = np.zeros((len(self.classes), len(self.classes)), dtype=int)

        self._generate_matrix()

    def _generate_matrix(self):
        """
        Populate the confusion matrix based on gold and system labels.
        """
        for gold, system in zip(self.gold_labels, self.system_labels):
            gold_idx = self.class_indices[gold]
            system_idx = self.class_indices[system]
            self.matrix[gold_idx, system_idx] += 1

    def proximity(self, ci_class, cj_class):
        """
        Compute the proximity between two classes ci_class and cj_class.
        
        :param ci_class: The predicted class.
        :param cj_class: The gold class.
        :return: Proximity value.
        """
        ci_index = self.class_indices.get(ci_class)
        cj_index = self.class_indices.get(cj_class)

        if ci_index is None or cj_index is None:
            return 0.0

        # Compute the distance between ci_class and cj_class in terms of ordinal indices
        distance = abs(ci_index - cj_index)

        # Compute the proximity based on the ordinal distance and class distributions
        num_instances = len(self.gold_labels)
        gold_counts = Counter(self.gold_labels)

        items_gold_class_ci = gold_counts[ci_class] if ci_class in gold_counts else 0
        sum_items_classes = sum(
            gold_counts[self.classes[i]]
            for i in range(min(ci_index, cj_index) + 1, max(ci_index, cj_index))
        ) if ci_index != cj_index else 0

        proximity = 0.0
        if num_instances > 0:
            proximity = ((items_gold_class_ci / 2) + sum_items_classes) / num_instances

        if proximity > 0.0:
            proximity = -1 * math.log2(proximity)

        return proximity
