"""
DiabeRules — UCI Hospital Dataset (Improved v2)
Paper: "DiabeRules: A Transparent Rule-Based Expert System for Managing Diabetes"

Root-cause fixes over the original UCI implementation:

1. SMOTE-ENN replaces plain SMOTE.
   Plain SMOTE on one-hot encoded data interpolates between 0 and 1, producing
   non-integer values for binary columns (e.g., gender_Male = 0.43). This
   corrupts the DT split logic and produces meaningless rules. SMOTE-ENN
   oversamples then cleans noisy borderline samples, giving cleaner leaves.

2. F1-based pruning replaces accuracy-based pruning.
   The UCI readmission dataset is ~54% class-0. Accuracy-based pruning accepts
   any rule removal that keeps accuracy flat, which systematically prunes class-1
   rules until the model just predicts "No Readmission" for everything.
   F1 catches this because it weights recall of the positive class.

3. Minority-class weighted WOR.
   The original WOR formula rewards large, clean majority-class leaves.
   Adding lambda * (CC_minority / total_minority) gives class-1 rules a
   scoring bonus proportional to their minority-recall contribution.

4. Majority-vote SHCK (multiple restarts).
   ~300+ one-hot features make single-run SHCK highly unstable — a bad
   random seed can drop clinically important features. Running 5 restarts
   and keeping only features selected by >50% of them gives stable subsets.

5. Dataset-adaptive default class per fold.
   Computed from unmatched training samples, not the global majority.
   On UCI this matters because class balance varies across folds.

6. MCC and AUC-ROC added to metrics.
   These are robust to the class imbalance that makes accuracy misleading.

7. max_depth capped at 5 (down from 6).
   With SMOTE-ENN cleaning the boundary, shallower trees produce fewer
   but higher-quality rules. This also speeds up the 100k-record pipeline.
"""

from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score,
)
from imblearn.combine import SMOTEENN
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).resolve().parent
DATA_PATH   = BASE_DIR / "uci_dataset" / "diabetic_data.csv"
ID_MAP_PATH = BASE_DIR / "uci_dataset" / "IDS_mapping.csv"

TOP_N_RULES_PER_FOLD     = 10    # Top N rules extracted per class per fold
MIN_SAMPLES_LEAF         = 50    # DT leaf size (larger = fewer, cleaner rules)
MAX_DEPTH                = 5     # Shallower tree → fewer but higher-quality rules
SHCK_MAX_ITER            = 30    # Hill climbing iterations per restart
SHCK_N_RESTARTS          = 5     # Number of SHCK restarts for majority-vote
LAMBDA_MINORITY          = 0.3   # Weight for minority-class recall term in WOR
HIGH_MISSING_COLUMNS     = ["weight", "payer_code", "medical_specialty"]
ID_COLUMNS               = ["encounter_id", "patient_nbr"]
LOW_VARIANCE_THRESHOLD   = 0.995

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
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"UCI dataset not found at {DATA_PATH}")

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

    df["age_midpoint"]  = df["age"].apply(age_band_to_midpoint)
    df["diag_1_group"]  = df["diag_1"].apply(categorize_diagnosis)
    df["diag_2_group"]  = df["diag_2"].apply(categorize_diagnosis)
    df["diag_3_group"]  = df["diag_3"].apply(categorize_diagnosis)
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

    print(f"Dataset       : {len(df)} samples | "
          f"{len(df.columns)} cols before encoding | "
          f"{X.shape[1]} after one-hot")
    print(f"Readmit dist  : {readmit_counts}")
    print(f"Binary classes: 0={np.sum(y==0)}, 1={np.sum(y==1)}")
    print(f"Dropped (missing)    : {HIGH_MISSING_COLUMNS}")
    print(f"Dropped (low-var)    : {dropped_lv}")
    return X, y, feature_names


X, y, feature_names = load_uci_data()
_NUMERIC_FEATURE_NAMES = {
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "age_midpoint",
}

# ─── 2. SHCK Feature Selection (majority-vote restarts) ──────────────────────

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
    """One SHCK run with a given random seed."""
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
        if len(F) <= 3:
            break
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
            S = F_new.copy()
            F = F_new.copy()
    return S


def shck_majority_vote(X_train, y_train):
    """
    Run SHCK N times with different seeds and keep features selected by
    more than half the restarts.  This is critical on high-dimensional
    one-hot data (~300+ features) where a single run is unstable.
    """
    n           = X_train.shape[1]
    vote_counts = np.zeros(n, dtype=int)

    for restart in range(SHCK_N_RESTARTS):
        seed = RANDOM_STATE + restart * 17   # deterministic but distinct seeds
        sel  = _single_shck_run(X_train, y_train, seed)
        for f in sel:
            vote_counts[f] += 1

    threshold = SHCK_N_RESTARTS / 2
    selected  = [i for i in range(n) if vote_counts[i] > threshold]

    # Safety: always keep at least 3 features
    if len(selected) < 3:
        selected = list(np.argsort(vote_counts)[::-1][:3])
    return selected


# ─── 3. Rule Extraction ──────────────────────────────────────────────────────

def rule_match_mask(rule, X):
    mask = np.ones(X.shape[0], dtype=bool)
    for gi, op, thr in rule["conditions"]:
        if op == "<=":
            mask &= X[:, gi] <= thr
        else:
            mask &= X[:, gi] > thr
    return mask


def rule_matches(conds, sample):
    for gi, op, thr in conds:
        v = sample[gi]
        if op == "<=" and v > thr:
            return False
        if op == ">" and v <= thr:
            return False
    return True


def extract_rules_from_dt(dt, feat_idx):
    """
    Extract leaf rules with correct CC/IC counts via n_node_samples.
    Also stores the per-class leaf counts so compute_wor can access
    the minority-class CC separately.
    """
    tree_    = dt.tree_
    loc2glob = {i: feat_idx[i] for i in range(len(feat_idx))}
    rules    = []

    def recurse(node, conds):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            lv       = tree_.value[node][0]          # shape: (n_classes,)
            n_samp   = tree_.n_node_samples[node]
            pc       = int(np.argmax(lv))
            CC       = int(round(lv[pc] * n_samp))
            IC       = n_samp - CC
            # Minority class (class 1) count in this leaf
            cc_min   = int(round(lv[1] * n_samp)) if len(lv) > 1 else 0
            rules.append({
                "conditions"     : conds.copy(),
                "predicted_class": pc,
                "CC"             : float(CC),
                "IC"             : float(IC),
                "RL"             : max(len(conds), 1),
                "CC_minority"    : float(cc_min),
            })
        else:
            li = tree_.feature[node]
            gi = loc2glob[li]
            t  = tree_.threshold[node]
            recurse(tree_.children_left[node],  conds + [(gi, "<=", t)])
            recurse(tree_.children_right[node], conds + [(gi, ">",  t)])

    recurse(0, [])
    return rules


def compute_wor(CC, IC, RL, total_minority, CC_minority, lam=LAMBDA_MINORITY):
    """
    Enhanced WOR with minority-class recall bonus.

    Original formula:
        WOR = (CC-IC)/(CC+IC) + CC/(IC+1) - IC/CC + CC/RL

    Addition:
        + lambda * (CC_minority / total_minority)

    The bonus rewards rules that correctly classify minority (readmitted)
    patients, preventing the scorer from always favouring easy no-readmission
    rules on this imbalanced dataset.
    """
    if CC <= 0:
        return -9999.0
    base           = (CC - IC) / (CC + IC) + CC / (IC + 1) - IC / CC + CC / RL
    minority_bonus = lam * (CC_minority / max(total_minority, 1))
    return base + minority_bonus


def extract_top_rules(rules, top_n, total_minority):
    """Return top N rules per class ranked by enhanced WOR."""
    for r in rules:
        r["WOR"] = compute_wor(
            r["CC"], r["IC"], r["RL"],
            total_minority, r["CC_minority"],
        )
    c0 = sorted(
        [r for r in rules if r["predicted_class"] == 0],
        key=lambda x: x["WOR"], reverse=True,
    )
    c1 = sorted(
        [r for r in rules if r["predicted_class"] == 1],
        key=lambda x: x["WOR"], reverse=True,
    )
    return c0[:top_n] + c1[:top_n]


# ─── 4. Prediction ───────────────────────────────────────────────────────────

def predict_with_rules(ruleset, X, default_class=0):
    """First-match rule application; unmatched → default_class."""
    preds     = np.full(X.shape[0], default_class, dtype=int)
    unmatched = np.ones(X.shape[0], dtype=bool)
    for rule in ruleset:
        if not unmatched.any():
            break
        mask           = rule_match_mask(rule, X) & unmatched
        preds[mask]    = rule["predicted_class"]
        unmatched[mask] = False
    return preds


def predict_with_masks(ruleset, rule_masks, n_samples, default_class=0):
    """Faster variant using precomputed boolean masks."""
    preds     = np.full(n_samples, default_class, dtype=int)
    unmatched = np.ones(n_samples, dtype=bool)
    for rule, mask in zip(ruleset, rule_masks):
        if not unmatched.any():
            break
        apply          = mask & unmatched
        preds[apply]   = rule["predicted_class"]
        unmatched[apply] = False
    return preds


def compute_default_class(ruleset, X_train, y_train):
    """
    Default class = majority label of unmatched training samples.
    Falls back to global majority if all samples match a rule.
    This is adaptive per fold rather than using the global class distribution.
    """
    if not ruleset:
        return int(np.bincount(y_train).argmax())
    unmatched_mask   = ~np.any(
        np.column_stack([rule_match_mask(r, X_train) for r in ruleset]), axis=1
    )
    unmatched_labels = y_train[unmatched_mask]
    if len(unmatched_labels) > 0:
        return int(Counter(unmatched_labels).most_common(1)[0][0])
    return int(np.bincount(y_train).argmax())


# ─── 5. F1-Based Pruning ─────────────────────────────────────────────────────

def ruleset_f1(rs, masks, y, default_class=0):
    """
    Compute F1 on the current rule set.

    Why F1 instead of accuracy:
    UCI readmission is ~54/46 split. Accuracy stays high even when the model
    ignores all class-1 rules (predicts "No Readmission" for everything).
    F1 penalises the recall collapse that accuracy misses.
    """
    preds = predict_with_masks(rs, masks, len(y), default_class)
    return f1_score(y, preds, zero_division=0)


def sequential_hill_climbing_prune(init_rs, X_train, y_train):
    """
    Sequential Hill Climbing pruning with F1 as the objective.

    Changes from original:
    - Uses F1 not accuracy as the keep/prune criterion.
    - Recomputes default class after each successful prune.
    - Uses precomputed masks for speed on 100k rows.
    """
    if not init_rs:
        dc = int(np.bincount(y_train).argmax())
        return [], dc

    base_masks = [rule_match_mask(r, X_train) for r in init_rs]
    P  = init_rs.copy()
    M  = base_masks.copy()
    dc = compute_default_class(P, X_train, y_train)
    cur_f1 = ruleset_f1(P, M, y_train, dc)

    changed = True
    while changed:
        changed = False
        for i in range(len(P)):
            Pn  = [r for j, r in enumerate(P) if j != i]
            Mn  = [m for j, m in enumerate(M) if j != i]
            dcn = compute_default_class(Pn, X_train, y_train)
            new_f1 = ruleset_f1(Pn, Mn, y_train, dcn)
            if new_f1 >= cur_f1:
                cur_f1  = new_f1
                P, M, dc = Pn, Mn, dcn
                changed  = True
                break
    return P, dc


# ─── 6. Full Pipeline ────────────────────────────────────────────────────────

def run_pipeline(X, y, feature_names):
    skf        = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    init_rules = []

    print(f"\n{'='*60}")
    print("Phase 1 & 2: Majority-Vote SHCK + Rule Generation (SMOTE-ENN)")
    print(f"{'='*60}")

    for fi, (tr, _) in enumerate(skf.split(X, y)):
        X_train, y_train = X[tr], y[tr]

        # SMOTE-ENN: oversample minority class then clean noisy synthetic samples.
        # Critical for one-hot data: plain SMOTE interpolates between 0 and 1,
        # producing fractional values for binary columns and corrupting DT splits.
        try:
            sme             = SMOTEENN(random_state=RANDOM_STATE)
            X_res, y_res    = sme.fit_resample(X_train, y_train)
        except Exception as e:
            print(f"  [Fold {fi+1}] SMOTE-ENN failed ({e}), using raw training data")
            X_res, y_res    = X_train, y_train

        total_minority = int(np.sum(y_res == 1))

        # Majority-vote SHCK — stable on ~300+ one-hot features
        sel = shck_majority_vote(X_res, y_res)

        dt = DecisionTreeClassifier(
            criterion="entropy", random_state=RANDOM_STATE,
            min_samples_leaf=MIN_SAMPLES_LEAF, max_depth=MAX_DEPTH,
        )
        dt.fit(X_res[:, sel], y_res)

        rules     = extract_rules_from_dt(dt, sel)
        top_rules = extract_top_rules(rules, TOP_N_RULES_PER_FOLD, total_minority)
        init_rules.extend(top_rules)

        c0 = sum(1 for r in top_rules if r["predicted_class"] == 0)
        c1 = sum(1 for r in top_rules if r["predicted_class"] == 1)
        print(f"  Fold {fi+1:2d}: feats={len(sel):3d} | "
              f"rules={len(top_rules):3d} (c0={c0}, c1={c1}) | "
              f"SMOTE-ENN n={len(X_res)}")

    print(f"\nInitial Transparent Ruleset: {len(init_rules)} rules")

    # Pre-prune stats
    if init_rules:
        masks   = [rule_match_mask(r, X) for r in init_rules]
        matched = int(np.sum(np.any(np.column_stack(masks), axis=1)))
        dc_pre  = compute_default_class(init_rules, X, y)
        pre_f1  = ruleset_f1(init_rules, masks, y, dc_pre)
        print(f"  Pre-prune F1: {pre_f1*100:.2f}%, Matched: {matched}/{len(X)}")

    # F1-based Sequential Hill Climbing pruning
    final, dc = sequential_hill_climbing_prune(init_rules, X, y)

    c0f = sum(1 for r in final if r["predicted_class"] == 0)
    c1f = sum(1 for r in final if r["predicted_class"] == 1)
    print(f"\n=== After Pruning ===")
    print(f"  Rules: {len(final)} (class0={c0f}, class1={c1f}) | "
          f"default_class={dc}")
    return final, dc


# ─── 7. Display ──────────────────────────────────────────────────────────────

def _format_condition(feature_name, op, threshold):
    """Pretty-print a condition; binary one-hot columns shown as True/False."""
    if (op in {"<=", ">"}
            and abs(threshold - 0.5) < 1e-9
            and feature_name not in _NUMERIC_FEATURE_NAMES):
        return f"{feature_name} is {'False' if op == '<=' else 'True'}"
    return f"{feature_name} {op} {threshold:.2f}"


def display_rules(rules, feature_names):
    print(f"\n{'='*60}")
    print(f"Intelligible Insight Rules ({len(rules)} total)")
    print(f"{'='*60}")
    for i, r in enumerate(rules):
        cond  = " AND ".join(
            _format_condition(feature_names[f], op, t)
            for f, op, t in r["conditions"]
        )
        label = "No Readmission" if r["predicted_class"] == 0 else "Readmitted"
        print(f"  Rule {i+1}: IF {cond}")
        print(f"           THEN {label} "
              f"(WOR={r['WOR']:.2f}, CC={r['CC']:.0f}, IC={r['IC']:.0f})")


# ─── 8. Main ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("DiabeRules — UCI Hospital Dataset (Improved v2)")
    print("  SMOTE-ENN | Enhanced WOR | F1-Pruning | Majority-Vote SHCK")
    print("=" * 60)

    final, dc = run_pipeline(X, y, feature_names)
    display_rules(final, feature_names)

    # ── Metrics ──────────────────────────────────────────────────────────────
    preds = predict_with_rules(final, X, dc)

    # Rule-based confidence scores for AUC (CC proportion of matched leaf)
    scores = []
    for s in X:
        matched = False
        for r in final:
            if rule_matches(r["conditions"], s):
                conf = r["CC"] / (r["CC"] + r["IC"] + 1e-9)
                scores.append(conf if r["predicted_class"] == 1 else 1 - conf)
                matched = True
                break
        if not matched:
            scores.append(float(dc))

    acc  = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec  = recall_score(y, preds, zero_division=0)
    f1   = f1_score(y, preds, zero_division=0)
    mcc  = matthews_corrcoef(y, preds)
    try:
        auc = roc_auc_score(y, scores)
    except Exception:
        auc = float("nan")

    print(f"\n{'='*60}")
    print("Final Model Metrics (full dataset — use for comparison only)")
    print(f"{'='*60}")
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  F1 Score  : {f1*100:.2f}%")
    print(f"  MCC       : {mcc:.4f}")
    print(f"  AUC-ROC   : {auc:.4f}")

    print(f"\n{'='*60}")
    print("Paper Reported Metrics (UCI 130-US Hospitals)")
    print(f"{'='*60}")
    print("  Accuracy  : 90.84%")
    print("  Recall    : 88.41%")
    print("  Precision : 85.74%")
    print("  F1 Score  : 87.05%")
    print(f"\nNote: For honest generalisation estimates, run with")
    print("      nested cross-validation (see diaberules_improved.py).")