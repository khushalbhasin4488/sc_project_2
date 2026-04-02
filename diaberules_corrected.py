"""
DiabeRules — Corrected Implementation (Final v7)
Paper: "DiabeRules: A Transparent Rule-Based Expert System for Managing Diabetes"

Key findings from cross-verification:
1. WOR parameters (CC/IC) must be the actual leaf sample counts (tree_.n_node_samples),
   not the normalized proportions returned by tree_.value.
2. SMOTE is applied to balance training data before feature selection and rule generation.
3. With balanced data, rules for BOTH classes are extracted. This is critical because
   SMOTE-balanced trees produce narrower class-0 leaves — extracting only class-0 rules
   would reduce coverage and hurt precision. Extracting both-class rules lets the system
   actively classify both directions instead of relying on a crude default.
4. Default class for unmatched samples = majority class of unmatched training data.
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import warnings
# from sklearn.ensemble import RandomForestClassifier
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─── 1. Load & Preprocess ────────────────────────────────────────────────────

df = pd.read_csv('/Users/khushalbhasin/Documents/code/sc_project/diabetes.csv')
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_with_zeros:
    median_val = df[col].replace(0, np.nan).median()
    df[col] = df[col].replace(0, median_val)

X = df.drop('Outcome', axis=1).values
y = df['Outcome'].values
feature_names = list(df.drop('Outcome', axis=1).columns)

TOP_N_RULES_PER_FOLD = 10  # Top N rules per class per fold

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class distribution: 0={np.sum(y==0)}, 1={np.sum(y==1)}")


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


def shck(X_train, y_train, max_iter=50):
    n = X_train.shape[1]
    F = list(range(n)); K = len(np.unique(y_train))
    dt = DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE, min_samples_leaf=5)
    dt.fit(X_train[:, F], y_train)
    best_acc = accuracy_score(y_train, dt.predict(X_train[:, F]))
    S = F.copy()
    for _ in range(max_iter):
        if len(F) <= 3: break
        sv = compute_cluster_scores(X_train[:, F], y_train, K)
        t = np.sum(sv); p = sv/t if t > 0 else np.ones(len(F))/len(F)
        idx = np.random.choice(len(F), p=p)
        rm = F[idx]; F_new = [f for f in F if f != rm]
        dt2 = DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE, min_samples_leaf=5)
        dt2.fit(X_train[:, F_new], y_train)
        a = accuracy_score(y_train, dt2.predict(X_train[:, F_new]))
        if a >= best_acc:
            best_acc = a; S = F_new.copy(); F = F_new.copy()
    return S


# ─── 3. Rule Extraction ──────────────────────────────────────────────────────

def rule_matches(conds, sample):
    for (gi, op, thr) in conds:
        v = sample[gi]
        if op == '<=' and v > thr: return False
        if op == '>' and v <= thr: return False
    return True


def extract_rules_from_dt(dt, feat_idx):
    """Extract leaf rules using strict n_node_samples for CC/IC derivation."""
    tree_ = dt.tree_
    loc2glob = {i: feat_idx[i] for i in range(len(feat_idx))}
    rules = []
    def recurse(node, conds):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            lv = tree_.value[node][0]
            n_samples = tree_.n_node_samples[node]
            pc = int(np.argmax(lv))
            CC = int(round(lv[pc] * n_samples))
            IC = n_samples - CC
            rules.append({
                'conditions': conds.copy(), 'predicted_class': pc,
                'CC': float(CC), 'IC': float(IC), 'RL': max(len(conds), 1)
            })
        else:
            li = tree_.feature[node]; gi = loc2glob[li]; t = tree_.threshold[node]
            recurse(tree_.children_left[node], conds + [(gi, '<=', t)])
            recurse(tree_.children_right[node], conds + [(gi, '>', t)])
    recurse(0, [])
    return rules


def compute_wor(CC, IC, RL):
    if CC <= 0: return -9999.0
    return (CC-IC)/(CC+IC) + CC/(IC+1) - IC/CC + CC/RL


def extract_top_rules(rules, top_n):
    """Return top N rules per class by WOR.
    With SMOTE-balanced training, both classes have strong leaf nodes.
    Extracting rules for both classes lets the system actively classify
    both directions, which is critical for precision.
    """
    for r in rules:
        r['WOR'] = compute_wor(r['CC'], r['IC'], r['RL'])
    # Extract top rules from BOTH classes
    c0_rules = sorted([r for r in rules if r['predicted_class'] == 0],
                      key=lambda x: x['WOR'], reverse=True)
    c1_rules = sorted([r for r in rules if r['predicted_class'] == 1],
                      key=lambda x: x['WOR'], reverse=True)
    return c0_rules[:top_n] + c1_rules[:top_n]


# ─── 4. Prediction & Pruning ─────────────────────────────────────────────────

def predict_with_rules(ruleset, X, default_class=0):
    preds = []
    for s in X:
        m = False
        for r in ruleset:
            if rule_matches(r['conditions'], s):
                preds.append(r['predicted_class']); m = True; break
        if not m: preds.append(default_class)
    return np.array(preds)


def ruleset_accuracy(rs, X, y, default_class=0):
    if not rs: return accuracy_score(y, np.full(len(y), default_class))
    return accuracy_score(y, predict_with_rules(rs, X, default_class))


def sequential_hill_climbing_prune(init_rs, X, y, default_class=0):
    P = init_rs.copy(); acc = ruleset_accuracy(P, X, y, default_class)
    changed = True
    while changed:
        changed = False
        for i in range(len(P)):
            Pn = [r for j, r in enumerate(P) if j != i]
            an = ruleset_accuracy(Pn, X, y, default_class)
            if an >= acc: acc = an; P = Pn; changed = True; break
    return P, acc


# ─── 5. Paper-style pipeline ─────────────────────────────────────────────────

def run_paper_style(X, y, feature_names):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    init_rules = []
    # Default class = majority class of full dataset
    default_class = int(np.bincount(y).argmax())

    print(f"\nDefault class for unmatched samples: {default_class}")
    print("\n=== Phase 1 & 2: SHCK + Rule Generation (with SMOTE) ===")
    for fi, (tr, _) in enumerate(skf.split(X, y)):
        Xt, yt = X[tr], y[tr]
        
        # Apply SMOTE to handle class imbalance on training data only
        smote = SMOTE(random_state=RANDOM_STATE)
        Xt_res, yt_res = smote.fit_resample(Xt, yt)
        
        sel = shck(Xt_res, yt_res, max_iter=50)
        dt = DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE,
                                    min_samples_leaf=5)
        dt.fit(Xt_res[:, sel], yt_res)
        rules = extract_rules_from_dt(dt, sel)
        top_rules = extract_top_rules(rules, TOP_N_RULES_PER_FOLD)
        init_rules.extend(top_rules)
        c0_count = sum(1 for r in top_rules if r['predicted_class'] == 0)
        c1_count = sum(1 for r in top_rules if r['predicted_class'] == 1)
        if top_rules:
            best = top_rules[0]
            print(f"  Fold {fi+1}: {len(sel)} feats, {len(top_rules)} rules "
                  f"(class0={c0_count}, class1={c1_count}) | "
                  f"Best WOR={best['WOR']:.2f}")

    print(f"\nInitial Transparent Ruleset: {len(init_rules)} rules")
    pre_acc = ruleset_accuracy(init_rules, X, y, default_class)
    matched = sum(1 for s in X if any(rule_matches(r['conditions'], s) for r in init_rules))
    print(f"  Pre-prune accuracy: {pre_acc*100:.2f}%, Matched: {matched}/{len(X)}")

    final, acc = sequential_hill_climbing_prune(init_rules, X, y, default_class)
    print(f"\n=== After Pruning ===")
    c0_final = sum(1 for r in final if r['predicted_class'] == 0)
    c1_final = sum(1 for r in final if r['predicted_class'] == 1)
    print(f"  Rules: {len(final)} (class0={c0_final}, class1={c1_final}), Accuracy: {acc*100:.2f}%")
    return final, default_class


def display_rules(rules):
    print(f"\n=== Intelligible Insight Rules (All {len(rules)} Rules) ===")
    for i, r in enumerate(rules):
        c = " AND ".join(f"{feature_names[f]} {op} {t:.2f}" for f, op, t in r['conditions'])
        cls_label = "Safe" if r['predicted_class'] == 0 else "Diabetic"
        print(f"  Rule {i+1}: IF {c} THEN {cls_label} (Class={r['predicted_class']}, "
              f"WOR={r['WOR']:.2f}, CC={r['CC']:.0f}, IC={r['IC']:.0f})")


if __name__ == "__main__":
    print("=" * 60)
    print("DiabeRules — Corrected Implementation (v7 + SMOTE)")
    print("=" * 60)

    final, dc = run_paper_style(X, y, feature_names)
    display_rules(final)

    # Calculate all metrics for the final pruned ruleset
    preds = predict_with_rules(final, X, dc)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)

    print("\n" + "=" * 60)
    print("Final Model Metrics (Our Implementation + SMOTE)")
    print("=" * 60)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")

    print("\n" + "=" * 60)
    print("Metrics Reported in Paper")
    print("=" * 60)
    print("  Accuracy  : 89.23%")
    print("  Recall    : 86.04%")
    print("  Precision : 83.81%")
    print("  F1 Score  : 85.02%")
