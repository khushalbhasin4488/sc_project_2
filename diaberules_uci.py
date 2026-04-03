"""
DiabeRules — UCI Hospital Dataset Implementation
Paper: "DiabeRules: A Transparent Rule-Based Expert System for Managing Diabetes"

This script mirrors the structure of diaberules_corrected.py and adapts the
same pipeline to the UCI Diabetes 130-US hospitals dataset.

Target choice for this dataset:
1. Every record is already a diabetes-related hospital encounter.
2. The practical binary target is hospital readmission.
3. Outcome = 1 if readmitted ('>30' or '<30'), else 0 for 'NO'.

Dataset analysis used for preprocessing:
1. 101,766 rows and 50 columns.
2. High-missing raw columns: weight, payer_code, medical_specialty.
3. Very high-cardinality diagnosis codes are grouped into clinical buckets.
4. Near-constant medication columns are removed automatically.
"""

from pathlib import Path
from collections import Counter
import csv
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score,
)
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "uci_dataset" / "diabetic_data.csv"
ID_MAP_PATH = BASE_DIR / "uci_dataset" / "IDS_mapping.csv"

TOP_N_RULES_PER_FOLD = 10
MIN_SAMPLES_LEAF = 50
MAX_DEPTH = 5
SHCK_MAX_ITER = 30
SHCK_N_RESTARTS = 5
LAMBDA_MINORITY = 0.3
HIGH_MISSING_COLUMNS = ["weight", "payer_code", "medical_specialty"]
ID_COLUMNS = ["encounter_id", "patient_nbr"]
LOW_VARIANCE_THRESHOLD = 0.995


# ─── 1. Load & Preprocess ────────────────────────────────────────────────────

def categorize_diagnosis(code):
    code = str(code).strip()
    if code in ["nan", "?", "Missing", ""]: return "Missing"
    if code.startswith(("V", "E")): return "Other"
    try:
        v = float(code)
        if 390 <= v < 460 or v == 785: return "Circulatory"
        if 460 <= v < 520 or v == 786: return "Respiratory"
        if 520 <= v < 580 or v == 787: return "Digestive"
        if int(v) == 250: return "Diabetes"
        if 800 <= v < 1000: return "Injury"
        if 710 <= v < 740: return "Musculoskeletal"
        if 580 <= v < 630 or v == 788: return "Genitourinary"
        if 140 <= v < 240: return "Neoplasms"
    except ValueError:
        pass
    return "Other"

def load_uci_data():
    # Load dataset with standardized NA values
    df = pd.read_csv(DATA_PATH, na_values=["?", "Unknown/Invalid"])
    df["Outcome"] = (df["readmitted"] != "NO").astype(int)
    df.drop(columns=ID_COLUMNS + HIGH_MISSING_COLUMNS + ["readmitted"], inplace=True)

    # Rapid ID Mapping parser
    id_maps = {"admission_type_id": {}, "discharge_disposition_id": {}, "admission_source_id": {}}
    current = None
    with open(ID_MAP_PATH) as f:
        for line in map(str.strip, f):
            if not line: continue
            if "description" in line:
                current = line.split(",")[0]
            elif current in id_maps:
                parts = line.split(",", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    id_maps[current][parts[0]] = parts[1].strip('"')

    for col, mapping in id_maps.items():
        clean_col = col.replace("_id", "")
        df[clean_col] = df[col].astype(str).map(mapping).fillna("Unknown")
        df.drop(columns=[col], inplace=True)

    # Vectorized age parsing & diagnostic categorization
    df["age_midpoint"] = df["age"].str.extract(r'\[(\d+)-(\d+)\)').astype(float).mean(axis=1)
    for col in ["diag_1", "diag_2", "diag_3"]:
        df[f"{col}_group"] = df[col].apply(categorize_diagnosis)
    df.drop(columns=["age", "diag_1", "diag_2", "diag_3"], inplace=True)

    # Drop low variance columns dynamically (>99.5% match)
    for col in list(df.columns.drop("Outcome")):
        if df[col].value_counts(normalize=True, dropna=False).iloc[0] >= LOW_VARIANCE_THRESHOLD:
            df.drop(columns=[col], inplace=True)

    # Impute missing values dynamically based on type
    numeric_cols = df.select_dtypes(include=np.number).columns.drop("Outcome", errors="ignore")
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(exclude=np.number).columns
    df[categorical_cols] = df[categorical_cols].fillna("Missing").astype(str)

    # Generate one-hot encoded feature matrix
    X_df = pd.get_dummies(df.drop(columns=["Outcome"]))
    return X_df.values.astype(np.float32), df["Outcome"].values, list(X_df.columns)


X, y, feature_names = load_uci_data()


# ─── 2. SHCK Feature Selection (majority-vote restarts) ──────────────────────

_NUMERIC_FEATURE_NAMES = {
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "age_midpoint",
}

def compute_cluster_scores(X_sub, y, K):
    n  = X_sub.shape[1]
    lo, hi = float(np.min(y)), float(np.max(y))
    centroids = np.random.uniform(lo, hi, size=K)
    mv = np.mean(X_sub, axis=0)
    cl = np.array([int(np.argmin(np.abs(mv[i] - centroids))) for i in range(n)])
    for j in range(K):
        a = [mv[i] for i in range(n) if cl[i] == j]
        if a:
            centroids[j] = np.mean(a)
    return np.array([np.abs(mv[i] - centroids[cl[i]]) for i in range(n)])

def _single_shck_run(X_train, y_train, seed):
    rng = np.random.RandomState(seed)
    n   = X_train.shape[1]
    F   = list(range(n))
    K   = len(np.unique(y_train))
    dt = DecisionTreeClassifier(
        criterion="entropy", random_state=seed,
        min_samples_leaf=MIN_SAMPLES_LEAF, max_depth=MAX_DEPTH,
    )
    dt.fit(X_train[:, F], y_train)
    best_acc = accuracy_score(y_train, dt.predict(X_train[:, F]))
    S = F.copy()
    for _ in range(SHCK_MAX_ITER):
        if len(F) <= 3: break
        sv = compute_cluster_scores(X_train[:, F], y_train, K)
        t  = np.sum(sv)
        p  = sv / t if t > 0 else np.ones(len(F)) / len(F)
        idx   = rng.choice(len(F), p=p)
        rm    = F[idx]
        F_new = [f for f in F if f != rm]
        dt2   = DecisionTreeClassifier(
            criterion="entropy", random_state=seed,
            min_samples_leaf=MIN_SAMPLES_LEAF, max_depth=MAX_DEPTH,
        )
        dt2.fit(X_train[:, F_new], y_train)
        a = accuracy_score(y_train, dt2.predict(X_train[:, F_new]))
        if a >= best_acc:
            best_acc = a
            S, F = F_new.copy(), F_new.copy()
    return S

def shck_majority_vote(X_train, y_train):
    n = X_train.shape[1]
    vote_counts = np.zeros(n, dtype=int)
    for restart in range(SHCK_N_RESTARTS):
        seed = RANDOM_STATE + restart * 17
        sel  = _single_shck_run(X_train, y_train, seed)
        for f in sel:
            vote_counts[f] += 1
    selected = [i for i in range(n) if vote_counts[i] > (SHCK_N_RESTARTS / 2)]
    if len(selected) < 3:
        selected = list(np.argsort(vote_counts)[::-1][:3])
    return selected

# ─── 3. Rule Extraction ──────────────────────────────────────────────────────

def rule_match_mask(rule, X):
    mask = np.ones(X.shape[0], dtype=bool)
    for gi, op, thr in rule["conditions"]:
        mask &= (X[:, gi] <= thr) if op == "<=" else (X[:, gi] > thr)
    return mask

def rule_matches(conds, sample):
    for gi, op, thr in conds:
        v = sample[gi]
        if op == "<=" and v > thr: return False
        if op == ">" and v <= thr: return False
    return True

def extract_rules_from_dt(dt, feat_idx):
    tree_ = dt.tree_
    loc2glob = {i: feat_idx[i] for i in range(len(feat_idx))}
    rules = []
    def recurse(node, conds):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            lv = tree_.value[node][0]
            n_samp = tree_.n_node_samples[node]
            pc = int(np.argmax(lv))
            CC = int(round(lv[pc] * n_samp))
            cc_min = int(round(lv[1] * n_samp)) if len(lv) > 1 else 0
            rules.append({
                "conditions": conds.copy(), "predicted_class": pc,
                "CC": float(CC), "IC": float(n_samp - CC), "RL": max(len(conds), 1),
                "CC_minority": float(cc_min),
            })
        else:
            li, t = tree_.feature[node], tree_.threshold[node]
            recurse(tree_.children_left[node], conds + [(loc2glob[li], "<=", t)])
            recurse(tree_.children_right[node], conds + [(loc2glob[li], ">", t)])
    recurse(0, [])
    return rules

def compute_wor(CC, IC, RL, total_minority, CC_minority, lam=LAMBDA_MINORITY):
    if CC <= 0: return -9999.0
    base = (CC - IC) / (CC + IC) + CC / (IC + 1) - IC / CC + CC / RL
    return base + (lam * (CC_minority / max(total_minority, 1)))

def extract_top_rules(rules, top_n, total_minority):
    for r in rules: r["WOR"] = compute_wor(r["CC"], r["IC"], r["RL"], total_minority, r["CC_minority"])
    c0 = sorted([r for r in rules if r["predicted_class"] == 0], key=lambda x: x["WOR"], reverse=True)
    c1 = sorted([r for r in rules if r["predicted_class"] == 1], key=lambda x: x["WOR"], reverse=True)
    return c0[:top_n] + c1[:top_n]

# ─── 4. Prediction & F1-Based Pruning ────────────────────────────────────────

def predict_with_rules(ruleset, X, default_class=0):
    preds, unmatched = np.full(X.shape[0], default_class, dtype=int), np.ones(X.shape[0], dtype=bool)
    for rule in ruleset:
        if not unmatched.any(): break
        mask = rule_match_mask(rule, X) & unmatched
        preds[mask], unmatched[mask] = rule["predicted_class"], False
    return preds

def predict_with_masks(ruleset, rule_masks, n_samples, default_class=0):
    preds, unmatched = np.full(n_samples, default_class, dtype=int), np.ones(n_samples, dtype=bool)
    for rule, mask in zip(ruleset, rule_masks):
        if not unmatched.any(): break
        apply = mask & unmatched
        preds[apply], unmatched[apply] = rule["predicted_class"], False
    return preds

def compute_default_class(ruleset, X_train, y_train):
    if not ruleset: return int(np.bincount(y_train).argmax())
    unmatched_labels = y_train[~np.any(np.column_stack([rule_match_mask(r, X_train) for r in ruleset]), axis=1)]
    return int(Counter(unmatched_labels).most_common(1)[0][0]) if len(unmatched_labels) > 0 else int(np.bincount(y_train).argmax())

def ruleset_f1(rs, masks, y, default_class=0):
    return f1_score(y, predict_with_masks(rs, masks, len(y), default_class), zero_division=0)

def sequential_hill_climbing_prune(init_rs, X_train, y_train):
    if not init_rs: return [], int(np.bincount(y_train).argmax())
    P, M = init_rs.copy(), [rule_match_mask(r, X_train) for r in init_rs]
    dc = compute_default_class(P, X_train, y_train)
    cur_f1, changed = ruleset_f1(P, M, y_train, dc), True

    while changed:
        changed = False
        for i in range(len(P)):
            Pn, Mn = [r for j, r in enumerate(P) if j != i], [m for j, m in enumerate(M) if j != i]
            dcn = compute_default_class(Pn, X_train, y_train)
            new_f1 = ruleset_f1(Pn, Mn, y_train, dcn)
            if new_f1 >= cur_f1:
                cur_f1, P, M, dc, changed = new_f1, Pn, Mn, dcn, True
                break
    return P, dc

# ─── 5. Full Pipeline ────────────────────────────────────────────────────────

def run_pipeline(X, y, feature_names):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    init_rules = []
    print(f"\n{'='*60}\nPhase 1 & 2: Majority-Vote SHCK + Rule Gen (SMOTE-ENN)\n{'='*60}")

    for fi, (tr, _) in enumerate(skf.split(X, y)):
        X_train, y_train = X[tr], y[tr]
        try:
            X_res, y_res = SMOTEENN(random_state=RANDOM_STATE).fit_resample(X_train, y_train)
        except Exception as e:
            print(f"  [Fold {fi+1}] SMOTE-ENN failed ({e}), using raw data")
            X_res, y_res = X_train, y_train

        tot_min = int(np.sum(y_res == 1))
        sel = shck_majority_vote(X_res, y_res)
        dt = DecisionTreeClassifier(criterion="entropy", random_state=RANDOM_STATE, min_samples_leaf=MIN_SAMPLES_LEAF, max_depth=MAX_DEPTH)
        dt.fit(X_res[:, sel], y_res)

        top_rules = extract_top_rules(extract_rules_from_dt(dt, sel), TOP_N_RULES_PER_FOLD, tot_min)
        init_rules.extend(top_rules)
        c0, c1 = sum(r["predicted_class"] == 0 for r in top_rules), sum(r["predicted_class"] == 1 for r in top_rules)
        print(f"  Fold {fi+1:2d}: feats={len(sel):3d} | rules={len(top_rules):3d} (c0={c0}, c1={c1}) | SMOTE-ENN n={len(X_res)}")

    print(f"\nInitial Ruleset: {len(init_rules)} rules")
    if init_rules:
        masks = [rule_match_mask(r, X) for r in init_rules]
        print(f"  Pre-prune F1: {ruleset_f1(init_rules, masks, y, compute_default_class(init_rules, X, y))*100:.2f}%, Matched: {np.sum(np.any(np.column_stack(masks), axis=1))}")

    final, dc = sequential_hill_climbing_prune(init_rules, X, y)
    print(f"\n=== After Pruning ===\n  Rules: {len(final)} (c0={sum(r['predicted_class']==0 for r in final)}, c1={sum(r['predicted_class']==1 for r in final)}) | dc={dc}")
    return final, dc

def _format_condition(feature_name, op, threshold):
    if op in {"<=", ">"} and abs(threshold - 0.5) < 1e-9 and feature_name not in _NUMERIC_FEATURE_NAMES:
        return f"{feature_name} is {'False' if op == '<=' else 'True'}"
    return f"{feature_name} {op} {threshold:.2f}"

def display_rules(rules, feature_names):
    print(f"\n{'='*60}\nIntelligible Insight Rules ({len(rules)} total)\n{'='*60}")
    for i, r in enumerate(rules):
        cond = " AND ".join(_format_condition(feature_names[f], op, t) for f, op, t in r["conditions"])
        print(f"  Rule {i+1}: IF {cond}\n           THEN {'Readmitted' if r['predicted_class'] == 1 else 'No Readmission'} (WOR={r['WOR']:.2f}, CC={r['CC']:.0f})")

if __name__ == "__main__":
    final, dc = run_pipeline(X, y, feature_names)
    display_rules(final, feature_names)

    preds = predict_with_rules(final, X, dc)
    scores = []
    for s in X:
        m = False
        for r in final:
            if rule_matches(r["conditions"], s):
                conf = r["CC"] / (r["CC"] + r["IC"] + 1e-9)
                scores.append(conf if r["predicted_class"] == 1 else 1 - conf)
                m = True
                break
        if not m: scores.append(float(dc))

    print(f"\n{'='*60}\nFinal Model Metrics (UCI Implementation + SMOTE-ENN)\n{'='*60}")
    print(f"  Accuracy  : {accuracy_score(y, preds)*100:.2f}%")
    print(f"  Recall    : {recall_score(y, preds, zero_division=0)*100:.2f}%")
    print(f"  Precision : {precision_score(y, preds, zero_division=0)*100:.2f}%")
    print(f"  F1 Score  : {f1_score(y, preds, zero_division=0)*100:.2f}%")
    try: print(f"  AUC-ROC   : {roc_auc_score(y, scores):.4f}")
    except: pass
