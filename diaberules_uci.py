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
import csv
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
SHCK_MAX_ITER = 30
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


# ─── 2. SHCK Feature Selection ───────────────────────────────────────────────

def compute_cluster_scores(X_sub, y, K):
    n = X_sub.shape[1]
    lo, hi = float(np.min(y)), float(np.max(y))
    centroids = np.random.uniform(lo, hi, size=K)
    mv = np.mean(X_sub, axis=0)
    cl = np.array([int(np.argmin(np.abs(mv[i] - centroids))) for i in range(n)])
    for j in range(K):
        a = [mv[i] for i in range(n) if cl[i] == j]
        if a:
            centroids[j] = np.mean(a)
    return np.array([np.abs(mv[i] - centroids[cl[i]]) for i in range(n)])


def shck(X_train, y_train, max_iter=SHCK_MAX_ITER):
    n = X_train.shape[1]
    F = list(range(n))
    K = len(np.unique(y_train))
    dt = DecisionTreeClassifier(
        criterion="entropy",
        random_state=RANDOM_STATE,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        max_depth=6
    )
    dt.fit(X_train[:, F], y_train)
    best_acc = accuracy_score(y_train, dt.predict(X_train[:, F]))
    S = F.copy()
    for _ in range(max_iter):
        if len(F) <= 3:
            break
        sv = compute_cluster_scores(X_train[:, F], y_train, K)
        t = np.sum(sv)
        p = sv / t if t > 0 else np.ones(len(F)) / len(F)
        idx = np.random.choice(len(F), p=p)
        rm = F[idx]
        F_new = [f for f in F if f != rm]
        dt2 = DecisionTreeClassifier(
            criterion="entropy",
            random_state=RANDOM_STATE,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_depth=6
        )
        dt2.fit(X_train[:, F_new], y_train)
        a = accuracy_score(y_train, dt2.predict(X_train[:, F_new]))
        if a >= best_acc:
            best_acc = a
            S = F_new.copy()
            F = F_new.copy()
    return S


# ─── 3. Rule Extraction ──────────────────────────────────────────────────────

def rule_matches(conds, sample):
    for (gi, op, thr) in conds:
        v = sample[gi]
        if op == "<=" and v > thr:
            return False
        if op == ">" and v <= thr:
            return False
    return True


def rule_match_mask(rule, X):
    mask = np.ones(X.shape[0], dtype=bool)
    for gi, op, thr in rule["conditions"]:
        if op == "<=":
            mask &= X[:, gi] <= thr
        else:
            mask &= X[:, gi] > thr
    return mask


def extract_rules_from_dt(dt, feat_idx):
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
            rules.append(
                {
                    "conditions": conds.copy(),
                    "predicted_class": pc,
                    "CC": float(CC),
                    "IC": float(IC),
                    "RL": max(len(conds), 1),
                }
            )
        else:
            li = tree_.feature[node]
            gi = loc2glob[li]
            t = tree_.threshold[node]
            recurse(tree_.children_left[node], conds + [(gi, "<=", t)])
            recurse(tree_.children_right[node], conds + [(gi, ">", t)])

    recurse(0, [])
    return rules


def compute_wor(CC, IC, RL):
    if CC <= 0:
        return -9999.0
    return (CC - IC) / (CC + IC) + CC / (IC + 1) - IC / CC + CC / RL


def extract_top_rules(rules, top_n):
    for r in rules:
        r["WOR"] = compute_wor(r["CC"], r["IC"], r["RL"])
    c0_rules = sorted([r for r in rules if r["predicted_class"] == 0], key=lambda x: x["WOR"], reverse=True)
    c1_rules = sorted([r for r in rules if r["predicted_class"] == 1], key=lambda x: x["WOR"], reverse=True)
    return c0_rules[:top_n] + c1_rules[:top_n]


# ─── 4. Prediction & Pruning ─────────────────────────────────────────────────

def predict_with_rules(ruleset, X, default_class=0):
    preds = np.full(X.shape[0], default_class, dtype=int)
    unmatched = np.ones(X.shape[0], dtype=bool)
    for rule in ruleset:
        if not unmatched.any():
            break
        mask = rule_match_mask(rule, X) & unmatched
        preds[mask] = rule["predicted_class"]
        unmatched[mask] = False
    return preds


def predict_with_masks(ruleset, rule_masks, n_samples, default_class=0):
    preds = np.full(n_samples, default_class, dtype=int)
    if not rule_masks:
        return preds
    unmatched = np.ones(preds.shape[0], dtype=bool)
    for rule, mask in zip(ruleset, rule_masks):
        if not unmatched.any():
            break
        apply_mask = mask & unmatched
        preds[apply_mask] = rule["predicted_class"]
        unmatched[apply_mask] = False
    return preds


def ruleset_accuracy(rs, X, y, default_class=0):
    if not rs:
        return accuracy_score(y, np.full(len(y), default_class))
    return accuracy_score(y, predict_with_rules(rs, X, default_class))


def sequential_hill_climbing_prune(init_rs, X, y, default_class=0):
    if not init_rs:
        return [], accuracy_score(y, np.full(len(y), default_class))

    base_masks = [rule_match_mask(rule, X) for rule in init_rs]
    P = init_rs.copy()
    M = base_masks.copy()
    acc = accuracy_score(y, predict_with_masks(P, M, len(y), default_class))
    changed = True

    while changed:
        changed = False
        for i in range(len(P)):
            Pn = [r for j, r in enumerate(P) if j != i]
            Mn = [m for j, m in enumerate(M) if j != i]
            preds = predict_with_masks(Pn, Mn, len(y), default_class)
            an = accuracy_score(y, preds)
            if an >= acc:
                acc = an
                P = Pn
                M = Mn
                changed = True
                break
    return P, acc


# ─── 5. Paper-style pipeline ─────────────────────────────────────────────────

def run_paper_style(X, y, feature_names):
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    init_rules = []
    default_class = int(np.bincount(y).argmax())

    print(f"\nDefault class for unmatched samples: {default_class}")
    print("\n=== Phase 1 & 2: SHCK + Rule Generation (Optimized Multi-Class) ===")
    for fi, (tr, _) in enumerate(skf.split(X, y)):
        Xt, yt = X[tr], y[tr]

        smote = SMOTE(random_state=RANDOM_STATE)
        Xt_res, yt_res = smote.fit_resample(Xt, yt)

        sel = shck(Xt_res, yt_res, max_iter=SHCK_MAX_ITER)
        dt = DecisionTreeClassifier(
            criterion="entropy",
            random_state=RANDOM_STATE,
            min_samples_leaf=MIN_SAMPLES_LEAF,
            max_depth=6
        )
        dt.fit(Xt_res[:, sel], yt_res)
        rules = extract_rules_from_dt(dt, sel)
        top_rules = extract_top_rules(rules, TOP_N_RULES_PER_FOLD)
        init_rules.extend(top_rules)
        c0_count = sum(1 for r in top_rules if r["predicted_class"] == 0)
        c1_count = sum(1 for r in top_rules if r["predicted_class"] == 1)
        if top_rules:
            best = top_rules[0]
            print(
                f"  Fold {fi + 1}: {len(sel)} feats, {len(top_rules)} rules "
                f"(class0={c0_count}, class1={c1_count}) | "
                f"Best WOR={best['WOR']:.2f}"
            )

    print(f"\nInitial Transparent Ruleset: {len(init_rules)} rules")
    pre_acc = ruleset_accuracy(init_rules, X, y, default_class)
    matched = int(np.sum(np.any(np.column_stack([rule_match_mask(r, X) for r in init_rules]), axis=1))) if init_rules else 0
    print(f"  Pre-prune accuracy: {pre_acc * 100:.2f}%, Matched: {matched}/{len(X)}")

    final, acc = sequential_hill_climbing_prune(init_rules, X, y, default_class)
    print("\n=== After Pruning ===")
    c0_final = sum(1 for r in final if r["predicted_class"] == 0)
    c1_final = sum(1 for r in final if r["predicted_class"] == 1)
    print(f"  Rules: {len(final)} (class0={c0_final}, class1={c1_final}), Accuracy: {acc * 100:.2f}%")
    return final, default_class


def _format_condition(feature_name, op, threshold):
    if op in {"<=", ">"} and abs(threshold - 0.5) < 1e-9 and feature_name not in {
        "time_in_hospital",
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "number_diagnoses",
        "age_midpoint",
    }:
        return f"{feature_name} is {'False' if op == '<=' else 'True'}"
    return f"{feature_name} {op} {threshold:.2f}"


def display_rules(rules):
    print(f"\n=== Intelligible Insight Rules (All {len(rules)} Rules) ===")
    for i, r in enumerate(rules):
        c = " AND ".join(_format_condition(feature_names[f], op, t) for f, op, t in r["conditions"])
        cls_label = "No Readmission" if r["predicted_class"] == 0 else "Readmitted"
        print(
            f"  Rule {i + 1}: IF {c} THEN {cls_label} (Class={r['predicted_class']}, "
            f"WOR={r['WOR']:.2f}, CC={r['CC']:.0f}, IC={r['IC']:.0f})"
        )


if __name__ == "__main__":
    print("=" * 60)
    print("DiabeRules — UCI Hospital Dataset")
    print("=" * 60)

    final, dc = run_paper_style(X, y, feature_names)
    display_rules(final)

    preds = predict_with_rules(final, X, dc)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)

    print("\n" + "=" * 60)
    print("Final Model Metrics (UCI Implementation + SMOTE)")
    print("=" * 60)
    print(f"  Accuracy  : {acc * 100:.2f}%")
    print(f"  Recall    : {rec * 100:.2f}%")
    print(f"  Precision : {prec * 100:.2f}%")
    print(f"  F1 Score  : {f1 * 100:.2f}%")
