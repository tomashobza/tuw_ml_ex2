

class RegressionTreeNode:
    def __init__(self, threshold=None, feature_index=None, right=None, left=None, value=None):
        self.threshold = threshold
        self.right_child = right
        self.left_child = left
        self.value = value
        self.feature = feature_index