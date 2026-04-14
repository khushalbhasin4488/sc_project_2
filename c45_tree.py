import numpy as np

def entropy(y):
    counts = np.bincount(y)
    probs = counts[counts > 0] / len(y)
    return -np.sum(probs * np.log2(probs))

class C45Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None,
                 is_leaf=False, class_label=None, cc=0, ic=0):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.is_leaf = is_leaf
        self.class_label = class_label
        self.cc = cc
        self.ic = ic

class C45Tree:
    """Custom implementation of C4.5 Decision Tree for numerical features."""
    def __init__(self, min_samples_leaf=5, max_depth=None):
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.root = None

    def fit(self, X, y):
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples = len(y)
        if num_samples == 0:
            return C45Node(is_leaf=True, class_label=0, cc=0, ic=0)

        counts = np.bincount(y)
        class_label = int(np.argmax(counts))
        cc = counts[class_label]
        ic = num_samples - cc

        # Stopping criteria
        if ic == 0 or num_samples < 2 * self.min_samples_leaf or (self.max_depth and depth >= self.max_depth):
            return C45Node(is_leaf=True, class_label=class_label, cc=cc, ic=ic)

        best_gain_ratio = -1
        best_feature = None
        best_threshold = None

        initial_entropy = entropy(y)

        # Iterate over features to find best split using Information Gain Ratio
        for feat_idx in range(X.shape[1]):
            x_col = X[:, feat_idx]
            unique_vals = np.unique(x_col)
            if len(unique_vals) <= 1:
                continue

            sorted_indices = np.argsort(x_col)
            x_sorted, y_sorted = x_col[sorted_indices], y[sorted_indices]

            for i in range(1, len(x_sorted)):
                if x_sorted[i] == x_sorted[i-1]:
                    continue
                
                if i < self.min_samples_leaf or (len(x_sorted) - i) < self.min_samples_leaf:
                    continue

                threshold = (x_sorted[i] + x_sorted[i-1]) / 2.0
                y_left = y_sorted[:i]
                y_right = y_sorted[i:]

                p_left = len(y_left) / num_samples
                p_right = len(y_right) / num_samples

                gain = initial_entropy - (p_left * entropy(y_left) + p_right * entropy(y_right))
                
                # C4.5 Split Info
                split_info = -(p_left * np.log2(p_left) + p_right * np.log2(p_right))
                gain_ratio = gain / split_info if split_info > 0 else 0

                if gain_ratio > best_gain_ratio:
                    best_gain_ratio = gain_ratio
                    best_feature = feat_idx
                    best_threshold = threshold

        if best_feature is None:
            return C45Node(is_leaf=True, class_label=class_label, cc=cc, ic=ic)

        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask

        left_child = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(X[right_mask], y[right_mask], depth + 1)

        return C45Node(feature_idx=best_feature, threshold=best_threshold,
                       left=left_child, right=right_child)

    def extract_rules(self, feature_idx_mapping=None):
        rules = []
        def dfs(node, current_conds):
            if node.is_leaf:
                rules.append({
                    'conditions': current_conds.copy(),
                    'predicted_class': node.class_label,
                    'CC': float(node.cc),
                    'IC': float(node.ic),
                    'RL': max(len(current_conds), 1)
                })
                return
            
            f_idx = feature_idx_mapping[node.feature_idx] if feature_idx_mapping is not None else node.feature_idx
            
            dfs(node.left, current_conds + [(f_idx, '<=', node.threshold)])
            dfs(node.right, current_conds + [(f_idx, '>', node.threshold)])

        dfs(self.root, [])
        return rules

    def predict(self, X):
        return np.array([self._predict_one(x, self.root) for x in X])

    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.class_label
        if x[node.feature_idx] <= node.threshold:
            return self._predict_one(x, node.left)
        else:
            return self._predict_one(x, node.right)
