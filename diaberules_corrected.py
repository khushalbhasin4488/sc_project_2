"""
DiabeRules — Re-Implementation (Fixing Deviations)
Authentic C4.5 integration, correct temporal split, proper pruning, and Risk Factor Analysis.

Note regarding C4.5:
Since libraries like `chefboost` do not expose leaf-node `CC` and `IC` statistics (which are vital for calculating the WOR metric described in the paper), an internal custom C4.5 builder `c45_tree` is utilized here. This ensures exact calculation of correct/incorrect samples at the nodes.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
DEFAULT_CLASS = 1
N_FOLDS = 10
HOLDOUT_SIZE = 0.2

# ─── 1. Load & Preprocess ────────────────────────────────────────────────────
df = pd.read_csv('/Users/khushalbhasin/Documents/code/sc_project/diabetes.csv')
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values
feature_names = list(df.drop('Outcome', axis=1).columns)

def temporal_split(X, y, holdout_size=HOLDOUT_SIZE):
    n_holdout = int(round(len(X) * holdout_size))
    n_dev = len(X) - n_holdout
    return X[:n_dev], X[n_dev:], y[:n_dev], y[n_dev:]


X_dev, X_holdout, y_dev, y_holdout = temporal_split(X, y)
# Compute medians from dev cohort only to prevent data leakage
for col_name in cols_with_zeros:
    col_idx = feature_names.index(col_name)
    col_data = X_dev[:, col_idx]
    median_val = np.nanmedian(col_data[col_data != 0])
    X_dev[:, col_idx] = np.where(X_dev[:, col_idx] == 0, median_val, X_dev[:, col_idx])
    X_holdout[:, col_idx] = np.where(X_holdout[:, col_idx] == 0, median_val, X_holdout[:, col_idx])
print(f"Development Cohort: {len(X_dev)} samples")
print(f"Holdout Cohort: {len(X_holdout)} samples")
print(f"Development class distribution: 0={np.sum(y_dev==0)}, 1={np.sum(y_dev==1)}")
print(f"Holdout class distribution: 0={np.sum(y_holdout==0)}, 1={np.sum(y_holdout==1)}")

# ─── 2. SHCK Feature Selection ───────────────────────────────────────────────
def compute_cluster_scores(X_sub, y, K):
    n = X_sub.shape[1]
    lo, hi = float(np.min(y)), float(np.max(y))
    centroids = np.random.uniform(lo, hi, size=K)
    mv = np.mean(X_sub, axis=0)
    cl = np.array([int(np.argmin(np.abs(mv[i] - centroids))) for i in range(n)])
    for j in range(K):
        a = [mv[i] for i in range(n) if cl[i] == j]
        if a: centroids[j] = np.mean(a)
    return np.array([np.abs(mv[i] - centroids[cl[i]]) for i in range(n)])

def shck(X_tr, y_tr, X_val, y_val, max_iter=50):
    n = X_tr.shape[1]
    F = list(range(n))
    K = len(np.unique(y_tr))
    
    dt = C45Tree(min_samples_leaf=5)
    dt.fit(X_tr[:, F], y_tr)
    best_acc = accuracy_score(y_val, dt.predict(X_val[:, F]))
    S = F.copy()
    
    for _ in range(max_iter):
        if len(F) <= 3: break
        sv = compute_cluster_scores(X_tr[:, F], y_tr, K)
        # Fitness proportionate selection: highest score = highest probability to remove
        p = sv / np.sum(sv) if np.sum(sv) > 0 else np.ones(len(F)) / len(F)
        
        idx = np.random.choice(len(F), p=p)
        rm = F[idx]
        F_new = [f for f in F if f != rm]
        
        dt2 = C45Tree(min_samples_leaf=5)
        dt2.fit(X_tr[:, F_new], y_tr)
        a = accuracy_score(y_val, dt2.predict(X_val[:, F_new]))
        
        if a >= best_acc:
            best_acc = a
            S = F_new.copy()
            F = F_new.copy()
            
    return S

# ─── 3. Rule Extraction ──────────────────────────────────────────────────────
def compute_wor(CC, IC, RL):
    if CC <= 0: return -9999.0
    return (CC-IC)/(CC+IC) + CC/(IC+1) - IC/CC + CC/RL

def simplify_conditions(conditions):
    bounds_by_feature = {}
    for feat_idx, op, threshold in conditions:
        bounds = bounds_by_feature.setdefault(feat_idx, {'lower': -np.inf, 'upper': np.inf})
        if op == '<=':
            bounds['upper'] = min(bounds['upper'], threshold)
        else:
            bounds['lower'] = max(bounds['lower'], threshold)

    simplified = []
    for feat_idx in sorted(bounds_by_feature):
        bounds = bounds_by_feature[feat_idx]
        if bounds['lower'] >= bounds['upper']:
            return []
        if np.isfinite(bounds['lower']):
            simplified.append((feat_idx, '>', bounds['lower']))
        if np.isfinite(bounds['upper']):
            simplified.append((feat_idx, '<=', bounds['upper']))
    return simplified

def normalize_rule(rule):
    normalized = dict(rule)
    normalized['conditions'] = simplify_conditions(rule['conditions'])
    normalized['RL'] = max(len(normalized['conditions']), 1)
    return normalized

def ruleset_stats(ruleset):
    used_features = sorted({feat_idx for rule in ruleset for feat_idx, _, _ in rule['conditions']})
    total_conditions = sum(len(rule['conditions']) for rule in ruleset)
    return {
        'rule_count': len(ruleset),
        'feature_count': len(used_features),
        'features': used_features,
        'total_conditions': total_conditions,
    }

def score_ruleset(ruleset, X, y):
    stats = ruleset_stats(ruleset)
    acc = ruleset_accuracy(ruleset, X, y)
    return (
        acc,
        -stats['feature_count'],
        -stats['total_conditions'],
        -stats['rule_count'],
    )

def rule_matches(conds, sample):
    for (gi, op, thr) in conds:
        v = sample[gi]
        if op == '<=' and v > thr: return False
        if op == '>' and v <= thr: return False
    return True

# ─── 4. Prediction & Pruning ─────────────────────────────────────────────────
def predict_with_rules(ruleset, X, default_class=DEFAULT_CLASS):
    preds = []
    for s in X:
        m = False
        for r in ruleset:
            if rule_matches(r['conditions'], s):
                preds.append(r['predicted_class'])
                m = True
                break
        if not m: preds.append(default_class)
    return np.array(preds)

def ruleset_accuracy(rs, X, y):
    if not rs: return 0.0
    return accuracy_score(y, predict_with_rules(rs, X))

def build_initial_ruleset(X_train, y_train, verbose=False):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    init_rules = []

    for fi, (inner_train_idx, inner_val_idx) in enumerate(skf.split(X_train, y_train)):
        X_inner_train = X_train[inner_train_idx]
        y_inner_train = y_train[inner_train_idx]
        X_inner_val = X_train[inner_val_idx]
        y_inner_val = y_train[inner_val_idx]

        sel = shck(X_inner_train, y_inner_train, X_inner_val, y_inner_val, max_iter=50)

        dt = C45Tree(min_samples_leaf=5)
        dt.fit(X_inner_train[:, sel], y_inner_train)

        raw_rules = dt.extract_rules(feature_idx_mapping=sel)
        # Compute WOR using original tree-path RL before normalization
        for r in raw_rules:
            r['WOR'] = compute_wor(r['CC'], r['IC'], r['RL'])
        rules = [normalize_rule(r) for r in raw_rules]

        c0_rules = [r for r in rules if r['predicted_class'] == 0]
        c0_rules.sort(key=lambda x: (x['WOR'], -len({f for f, _, _ in x['conditions']}), -x['RL']), reverse=True)

        if c0_rules:
            best = c0_rules[0]
            init_rules.append(best)
            if verbose:
                print(f"  Inner Fold {fi+1}: {len(sel)} feats, extracted best rule | "
                      f"WOR={best['WOR']:.2f}, CC={best['CC']:.0f}, IC={best['IC']:.0f}")

    return init_rules


def sequential_hill_climbing_prune(init_rules, X_train, y_train):
    P = init_rules.copy()
    acc = ruleset_accuracy(P, X_train, y_train)
    changed = True
    while changed:
        changed = False
        for i in range(len(P)):
            Pn = [r for j, r in enumerate(P) if j != i]
            an = ruleset_accuracy(Pn, X_train, y_train)
            if an >= acc:
                acc = an
                P = Pn
                changed = True
                break
    return P, acc

# ─── 5. Paper-style pipeline ─────────────────────────────────────────────────
def fit_diaberules(X_train, y_train, verbose=False):
    if verbose:
        print("\n=== Phase 1 & 2: SHCK + Rule Generation (C4.5) ===")

    init_rules = build_initial_ruleset(X_train, y_train, verbose=verbose)

    pre_acc = ruleset_accuracy(init_rules, X_train, y_train)
    if verbose:
        print(f"\nInitial Transparent Ruleset: {len(init_rules)} rules")
        print(f"  Pre-prune accuracy (training split): {pre_acc*100:.2f}%")

    final_rules, train_acc = sequential_hill_climbing_prune(init_rules, X_train, y_train)
    if verbose:
        stats = ruleset_stats(final_rules)
        print(f"\n=== After Pruning ===")
        print(f"  Rules: {len(final_rules)}, Accuracy on training split: {train_acc*100:.2f}%")
        print(f"  Features used: {stats['feature_count']} -> {[feature_names[i] for i in stats['features']]}")

    return final_rules, train_acc


def evaluate_development_cv(X_dev, y_dev):
    outer_skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    accs, precs, recs, f1s = [], [], [], []

    print("\n=== 10-Fold CV on Development Cohort ===")
    for fold_idx, (train_idx, test_idx) in enumerate(outer_skf.split(X_dev, y_dev)):
        X_train_fold = X_dev[train_idx]
        y_train_fold = y_dev[train_idx]
        X_test_fold = X_dev[test_idx]
        y_test_fold = y_dev[test_idx]

        final_rules, train_acc = fit_diaberules(X_train_fold, y_train_fold, verbose=False)
        preds = predict_with_rules(final_rules, X_test_fold)

        acc = accuracy_score(y_test_fold, preds)
        prec = precision_score(y_test_fold, preds, zero_division=0)
        rec = recall_score(y_test_fold, preds, zero_division=0)
        f1 = f1_score(y_test_fold, preds, zero_division=0)

        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)

        print(f"  Fold {fold_idx+1}: Acc={acc*100:.2f}% | Rules={len(final_rules)} | TrainAcc={train_acc*100:.2f}%")

    print("\nDevelopment CV Summary")
    print(f"  Accuracy  : {np.mean(accs)*100:.2f}% ± {np.std(accs)*100:.2f}")
    print(f"  Recall    : {np.mean(recs)*100:.2f}% ± {np.std(recs)*100:.2f}")
    print(f"  Precision : {np.mean(precs)*100:.2f}% ± {np.std(precs)*100:.2f}")
    print(f"  F1 Score  : {np.mean(f1s)*100:.2f}% ± {np.std(f1s)*100:.2f}")

def display_rules(rules):
    print("\n=== Intelligible Insight Rules ===")
    for i, r in enumerate(rules):
        c = " AND ".join(f"{feature_names[f]} {op} {t:.2f}" for f, op, t in r['conditions'])
        print(f"  Rule {i+1}: IF {c} THEN Class={r['predicted_class']} "
              f"(WOR={r['WOR']:.2f}, CC={r['CC']:.0f}, IC={r['IC']:.0f})")

def save_rules_to_file(rules, filename="/Users/khushalbhasin/Documents/code/sc_project/intelligible_rules.txt"):
    with open(filename, 'w') as f:
        f.write("=== Intelligible Insight Rules ===\n")
        for i, r in enumerate(rules):
            c = " AND ".join(f"{feature_names[feat]} {op} {t:.2f}" for feat, op, t in r['conditions'])
            f.write(f"  Rule {i+1}: IF {c} THEN Class={r['predicted_class']} "
                  f"(WOR={r['WOR']:.2f}, CC={r['CC']:.0f}, IC={r['IC']:.0f})\n")
    print(f"\n[+] Rules successfully saved to '{filename}'")

def summarize_merged_rule(final_rules, max_features=2):
    if not final_rules:
        return {}

    feature_frequency = {}
    for rule in final_rules:
        for feat_idx, op, threshold in rule['conditions']:
            feature_frequency[feat_idx] = feature_frequency.get(feat_idx, 0) + 1

    ranked_features = sorted(
        feature_frequency,
        key=lambda feat_idx: (feature_frequency[feat_idx], -feat_idx),
        reverse=True
    )[:max_features]

    merged = {}
    for feat_idx in ranked_features:
        lower_bounds = []
        upper_bounds = []
        for rule in final_rules:
            for rule_feat_idx, op, threshold in rule['conditions']:
                if rule_feat_idx != feat_idx:
                    continue
                if op == '<=':
                    upper_bounds.append(threshold)
                else:
                    lower_bounds.append(threshold)

        lower = max(lower_bounds) if lower_bounds else -np.inf
        upper = max(upper_bounds) if upper_bounds else np.inf
        if lower < upper:
            merged[feat_idx] = {'lower': lower, 'upper': upper}

    return merged


def sample_outside_safe_range(sample_values, bound):
    lower = bound['lower']
    upper = bound['upper']
    outside = np.zeros(len(sample_values), dtype=bool)
    if np.isfinite(lower):
        outside |= sample_values <= lower
    if np.isfinite(upper):
        outside |= sample_values > upper
    return outside


def development_fold_indices(X_dev, y_dev):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return list(skf.split(X_dev, y_dev))


def risk_factor_analysis(final_rules, X_test, y_test):
    print("\n=== Phase 5: Risk Factor Analysis ===")
    merged_rule = summarize_merged_rule(final_rules)
    if not merged_rule:
        print("  Skipped: final rules do not collapse into a single consistent merged rule.")
        print("  This remains a paper mismatch; the current model still lacks the paper-style merged-rule risk analysis.")
        return

    print("  Merged rule summary:")
    for feat_idx, bounds in merged_rule.items():
        lower = bounds['lower']
        upper = bounds['upper']
        lower_text = f"> {lower:.2f}" if np.isfinite(lower) else "unbounded below"
        upper_text = f"<= {upper:.2f}" if np.isfinite(upper) else "unbounded above"
        print(f"    {feature_names[feat_idx]}: {lower_text}, {upper_text}")

    negative_mask = (y_test == 0)
    X_negative = X_test[negative_mask]
    if len(X_negative) == 0:
        print("  Skipped: no negative-class samples available in holdout cohort.")
        return

    ranked_features = list(merged_rule.keys())

    print("\n  Single-factor reversed-range misclassification on development folds:")
    dev_folds = development_fold_indices(X_dev, y_dev)
    for feat_idx in ranked_features:
        fold_rates = []
        for _, test_idx in dev_folds:
            X_fold = X_dev[test_idx]
            y_fold = y_dev[test_idx]
            X_fold_neg = X_fold[y_fold == 0]
            if len(X_fold_neg) == 0:
                continue
            misclassified = sample_outside_safe_range(X_fold_neg[:, feat_idx], merged_rule[feat_idx])
            fold_rates.append(np.mean(misclassified) * 100)

        label = feature_names[feat_idx]
        formatted_rates = [round(float(rate), 2) for rate in fold_rates]
        print(f"    {label}: {formatted_rates}")
        if fold_rates:
            print(f"    Average {label}: {np.mean(fold_rates):.2f}%")

    if len(ranked_features) >= 2:
        print("\n  Two-factor reversed-range misclassification on development folds:")
        primary = ranked_features[:2]
        fold_rates = []
        for _, test_idx in dev_folds:
            X_fold = X_dev[test_idx]
            y_fold = y_dev[test_idx]
            X_fold_neg = X_fold[y_fold == 0]
            if len(X_fold_neg) == 0:
                continue

            misclassified = np.zeros(len(X_fold_neg), dtype=bool)
            for feat_idx in primary:
                misclassified |= sample_outside_safe_range(X_fold_neg[:, feat_idx], merged_rule[feat_idx])
            fold_rates.append(np.mean(misclassified) * 100)

        labels = ", ".join(feature_names[idx] for idx in primary)
        formatted_rates = [round(float(rate), 2) for rate in fold_rates]
        print(f"    {labels}: {formatted_rates}")
        if fold_rates:
            print(f"    Average {labels}: {np.mean(fold_rates):.2f}%")
# ------------------------------------

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



if __name__ == "__main__":
    evaluate_development_cv(X_dev, y_dev)

    final, dev_train_acc = fit_diaberules(X_dev, y_dev, verbose=True)
    display_rules(final)
    save_rules_to_file(final, "/Users/khushalbhasin/Documents/code/sc_project/intelligible_rules.txt")

    preds = predict_with_rules(final, X_holdout)
    acc = accuracy_score(y_holdout, preds)
    prec = precision_score(y_holdout, preds, zero_division=0)
    rec = recall_score(y_holdout, preds, zero_division=0)
    f1 = f1_score(y_holdout, preds, zero_division=0)

    print("\n" + "=" * 60)
    print("Final Model Metrics on Holdout Cohort")
    print("=" * 60)
    print(f"  Development Train Accuracy : {dev_train_acc*100:.2f}%")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")

    risk_factor_analysis(final, X_holdout, y_holdout)
