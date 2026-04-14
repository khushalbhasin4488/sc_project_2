"""
DiabeRules — UCI Hospital Dataset Corrected Implementation (Final v7 equivalent)
"""
from pathlib import Path
from collections import Counter
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

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_PATH = "/Users/khushalbhasin/Documents/code/sc_project/uci_dataset/diabetic_data.csv"
ID_MAP_PATH = "/Users/khushalbhasin/Documents/code/sc_project/uci_dataset/IDS_mapping.csv"

TOP_N_RULES_PER_FOLD   = 10
HIGH_MISSING_COLUMNS   = ["weight", "payer_code", "medical_specialty"]
ID_COLUMNS             = ["encounter_id", "patient_nbr"]
LOW_VARIANCE_THRESHOLD = 0.995

# ─── 1. Load & Preprocess ────────────────────────────────────────────────────

def load_id_mappings(path):
    mappings = {}
    current_table = None
    with open(path, newline="") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                current_table = None
                continue
            if line.endswith(",description"):
                current_table = line.split(",", 1)[0]
                mappings[current_table] = {}
                continue
            if current_table is None:
                continue
            code, description = line.split(",", 1)
            code = code.strip()
            if not code:
                continue
            mappings[current_table][code] = description.strip().strip('"')
    return mappings

def age_band_to_midpoint(age_band):
    if pd.isna(age_band):
        return np.nan
    cleaned = str(age_band).strip()[1:-1]
    lower, upper = cleaned.split("-")
    return (float(lower) + float(upper)) / 2.0

def categorize_diagnosis(code):
    if pd.isna(code):
        return "Missing"
    code = str(code).strip()
    if not code or code == "?":
        return "Missing"
    if code.startswith("V") or code.startswith("E"):
        return "Other"
    try:
        value = float(code)
    except ValueError:
        return "Other"
    if 390 <= value < 460 or value == 785:
        return "Circulatory"
    if 460 <= value < 520 or value == 786:
        return "Respiratory"
    if 520 <= value < 580 or value == 787:
        return "Digestive"
    if int(value) == 250:
        return "Diabetes"
    if 800 <= value < 1000:
        return "Injury"
    if 710 <= value < 740:
        return "Musculoskeletal"
    if 580 <= value < 630 or value == 788:
        return "Genitourinary"
    if 140 <= value < 240:
        return "Neoplasms"
    return "Other"

def drop_low_variance_columns(df, protected_columns):
    removable = []
    for col in df.columns:
        if col in protected_columns:
            continue
        counts = (
            df[col].astype("object")
            .fillna("__MISSING__")
            .value_counts(normalize=True, dropna=False)
        )
        if len(counts) <= 1 or counts.iloc[0] >= LOW_VARIANCE_THRESHOLD:
            removable.append(col)
    return df.drop(columns=removable), removable

def load_uci_data():
    df = pd.read_csv(DATA_PATH)
    df = df.replace("?", np.nan)
    df["gender"] = df["gender"].replace("Unknown/Invalid", np.nan)

    readmit_counts = df["readmitted"].value_counts().to_dict()
    df["Outcome"]  = (df["readmitted"] != "NO").astype(int)
    df = df.drop(columns=ID_COLUMNS + HIGH_MISSING_COLUMNS + ["readmitted"])

    id_maps = load_id_mappings(ID_MAP_PATH)
    df["admission_type"] = (
        df["admission_type_id"].astype(str)
        .map(id_maps.get("admission_type_id", {}))
        .fillna("Unknown")
    )
    df["discharge_disposition"] = (
        df["discharge_disposition_id"].astype(str)
        .map(id_maps.get("discharge_disposition_id", {}))
        .fillna("Unknown")
    )
    df["admission_source"] = (
        df["admission_source_id"].astype(str)
        .map(id_maps.get("admission_source_id", {}))
        .fillna("Unknown")
    )
    df = df.drop(
        columns=["admission_type_id", "discharge_disposition_id", "admission_source_id"]
    )

    df["age_midpoint"] = df["age"].apply(age_band_to_midpoint)
    df["diag_1_group"] = df["diag_1"].apply(categorize_diagnosis)
    df["diag_2_group"] = df["diag_2"].apply(categorize_diagnosis)
    df["diag_3_group"] = df["diag_3"].apply(categorize_diagnosis)
    df = df.drop(columns=["age", "diag_1", "diag_2", "diag_3"])

    df, dropped_lv = drop_low_variance_columns(df, protected_columns={"Outcome"})

    numeric_cols = [
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses", "age_midpoint",
    ]
    categorical_cols = [
        c for c in df.columns if c not in numeric_cols + ["Outcome"]
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())
    for col in categorical_cols:
        df[col] = df[col].fillna("Missing").astype(str)

    X_df = pd.get_dummies(df.drop("Outcome", axis=1), drop_first=False, dtype=np.uint8)
    X    = X_df.astype(np.float32).values
    y    = df["Outcome"].astype(int).values
    feature_names = list(X_df.columns)

    print(f"Dataset       : {len(df)} samples | {X.shape[1]} features after encoding")
    print(f"Readmit dist  : {readmit_counts}")
    print(f"Binary classes: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
    print(f"Dropped (high-missing) : {HIGH_MISSING_COLUMNS}")
    print(f"Dropped (low-variance) : {dropped_lv}")
    return X, y, feature_names

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
        if a: centroids[j] = np.mean(a)
    return np.array([np.abs(mv[i] - centroids[cl[i]]) for i in range(n)])

def shck(X_train, y_train, max_iter=50):
    n = X_train.shape[1]
    F = list(range(n)); K = len(np.unique(y_train))
    dt = DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE, min_samples_leaf=50)
    dt.fit(X_train[:, F], y_train)
    best_acc = accuracy_score(y_train, dt.predict(X_train[:, F]))
    S = F.copy()
    for _ in range(max_iter):
        if len(F) <= 3: break
        sv = compute_cluster_scores(X_train[:, F], y_train, K)
        t = np.sum(sv); p = sv/t if t > 0 else np.ones(len(F))/len(F)
        idx = np.random.choice(len(F), p=p)
        rm = F[idx]; F_new = [f for f in F if f != rm]
        dt2 = DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE, min_samples_leaf=50)
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
    """Return top N rules per class by WOR."""
    for r in rules:
        r['WOR'] = compute_wor(r['CC'], r['IC'], r['RL'])
    c0_rules = sorted([r for r in rules if r['predicted_class'] == 0], key=lambda x: x['WOR'], reverse=True)
    c1_rules = sorted([r for r in rules if r['predicted_class'] == 1], key=lambda x: x['WOR'], reverse=True)
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
    default_class = int(np.bincount(y).argmax())

    print(f"\nDefault class for unmatched samples: {default_class}")
    print("\n=== Phase 1 & 2: SHCK + Rule Generation (with SMOTE) ===")
    for fi, (tr, _) in enumerate(skf.split(X, y)):
        Xt, yt = X[tr], y[tr]
        
        # If the dataset is already well-balanced (minority/majority ratio >= 0.8), skip it.
        n0_tr = int(np.sum(yt == 0))
        n1_tr = int(np.sum(yt == 1))
        current_ratio = n1_tr / max(n0_tr, 1)
        
        if current_ratio < 0.8:
            try:
                smote = SMOTE(random_state=RANDOM_STATE)
                Xt_res, yt_res = smote.fit_resample(Xt, yt)
            except Exception as e:
                print(f"  [Fold {fi+1}] SMOTE failed ({e}), using raw data")
                Xt_res, yt_res = Xt, yt
        else:
            Xt_res, yt_res = Xt, yt
        
        # Run standard sequential SHCK (using max_iter=30 for UCI speeds)
        sel = shck(Xt_res, yt_res, max_iter=30)
        
        dt = DecisionTreeClassifier(criterion='entropy', random_state=RANDOM_STATE, min_samples_leaf=50)
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

def _format_condition(feature_name, op, threshold):
    _NUMERIC_FEATURE_NAMES = {
        "time_in_hospital", "num_lab_procedures", "num_procedures",
        "num_medications", "number_outpatient", "number_emergency",
        "number_inpatient", "number_diagnoses", "age_midpoint",
    }
    if (op in {"<=", ">"}
            and abs(threshold - 0.5) < 1e-9
            and feature_name not in _NUMERIC_FEATURE_NAMES):
        return f"{feature_name} is {'False' if op == '<=' else 'True'}"
    return f"{feature_name} {op} {threshold:.2f}"

def display_rules(rules):
    print(f"\n=== Intelligible Insight Rules (All {len(rules)} Rules) ===")
    for i, r in enumerate(rules):
        c = " AND ".join(_format_condition(feature_names[f], op, t) for f, op, t in r['conditions'])
        cls_label = "No Readmission" if r['predicted_class'] == 0 else "Readmitted"
        print(f"  Rule {i+1}: IF {c} THEN {cls_label} (Class={r['predicted_class']}, "
              f"WOR={r['WOR']:.2f}, CC={r['CC']:.0f}, IC={r['IC']:.0f})")

if __name__ == "__main__":
    print("=" * 60)
    print("DiabeRules — Corrected UCI Implementation (v7 style + SMOTE)")
    print("=" * 60)

    final, dc = run_paper_style(X, y, feature_names)
    display_rules(final)

    preds = predict_with_rules(final, X, dc)
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)

    print("\n" + "=" * 60)
    print("Final Model Metrics (Methodology synced with improved Pima)")
    print("=" * 60)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")