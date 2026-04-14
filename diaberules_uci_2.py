"""
DiabeRules — UCI Hospital Dataset (v3 — Balanced Fix)

Problem diagnosis from v2 vs original:
  Original v1:  Acc=62.76, Rec=45.72, Prec=63.29, F1=53.09  ← low recall, biased to class-0
  Improved v2:  Acc=53.87, Rec=89.97, Prec=49.98, F1=64.26  ← overcorrected, biased to class-1

Root causes of v2 overcorrection:
  1. SMOTE-ENN was too aggressive — it removed too many majority-class samples,
     making the resampled training data heavily skewed toward class-1. On a 100k
     dataset with many borderline samples, ENN's cleaning step removes a large
     fraction of class-0, flipping the imbalance direction.
  2. LAMBDA_MINORITY=0.3 in WOR stacked on top of already-oversampled data,
     double-penalising class-0 rules at scoring time.
  3. F1-based pruning (with binary F1 defaulting to class-1) retained class-1
     rules aggressively even when precision was collapsing.

Fixes in v3:
  1. Replace SMOTE-ENN with BorderlineSMOTE.
     BorderlineSMOTE only synthesises near the decision boundary — it doesn't
     clean majority samples, so the class ratio stays controlled. This avoids
     the ENN over-removal problem on large datasets.
  2. Tune sampling_strategy explicitly to a target ratio (default 0.6) instead
     of full 1:1 balance. This prevents the hard flip to class-1 dominance.
  3. Replace binary F1 pruning with macro-F1 pruning.
     Macro-F1 averages F1 across both classes equally, so the pruner must
     maintain performance on BOTH classes — it can't just maximise class-1 recall.
  4. Set LAMBDA_MINORITY=0.1 (down from 0.3).
     Since BorderlineSMOTE already handles imbalance at data level, the WOR
     bonus only needs a small nudge, not a large correction.
  5. Adaptive default class from unmatched training samples (kept from v2).
  6. Majority-vote SHCK (kept from v2).
  7. MCC + AUC-ROC reporting (kept from v2).
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
from imblearn.over_sampling import BorderlineSMOTE
import warnings

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─── Configuration ────────────────────────────────────────────────────────────

BASE_DIR  = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "uci_dataset" / "diabetic_data.csv"
ID_MAP_PATH = BASE_DIR / "uci_dataset" / "IDS_mapping.csv"

TOP_N_RULES_PER_FOLD   = 10    # Top N rules extracted per class per fold
MIN_SAMPLES_LEAF       = 50    # DT leaf size
MAX_DEPTH              = 5     # Shallower tree → fewer but cleaner rules
SHCK_MAX_ITER          = 30    # Hill climbing iterations per restart
SHCK_N_RESTARTS        = 5     # Number of SHCK restarts for majority-vote
LAMBDA_MINORITY        = 0.1   # Small WOR bonus for minority class (v2 was 0.3, too high)
SAMPLING_STRATEGY      = 0.6   # Target minority/majority ratio after resampling
                                # 0.6 means minority becomes 60% of majority count
                                # keeps data realistic without hard-flipping balance

HIGH_MISSING_COLUMNS   = ["weight", "payer_code", "medical_specialty"]
ID_COLUMNS             = ["encounter_id", "patient_nbr"]
LOW_VARIANCE_THRESHOLD = 0.995

_NUMERIC_FEATURE_NAMES = {
    "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "number_diagnoses", "age_midpoint",
}

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
        sv  = compute_cluster_scores(X_train[:, F], y_train, K)
        t   = np.sum(sv)
        p   = sv / t if t > 0 else np.ones(len(F)) / len(F)
        idx = rng.choice(len(F), p=p)
        rm  = F[idx]
        F_new = [f for f in F if f != rm]
        dt2 = DecisionTreeClassifier(
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
    n           = X_train.shape[1]
    vote_counts = np.zeros(n, dtype=int)
    for restart in range(SHCK_N_RESTARTS):
        seed = RANDOM_STATE + restart * 17
        sel  = _single_shck_run(X_train, y_train, seed)
        for f in sel:
            vote_counts[f] += 1
    threshold = SHCK_N_RESTARTS / 2
    selected  = [i for i in range(n) if vote_counts[i] > threshold]
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
    tree_    = dt.tree_
    loc2glob = {i: feat_idx[i] for i in range(len(feat_idx))}
    rules    = []

    def recurse(node, conds):
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            lv     = tree_.value[node][0]
            n_samp = tree_.n_node_samples[node]
            pc     = int(np.argmax(lv))
            CC     = int(round(lv[pc] * n_samp))
            IC     = n_samp - CC
            cc_min = int(round(lv[1] * n_samp)) if len(lv) > 1 else 0
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
    WOR with a small minority-class recall bonus (lambda=0.1).
    Kept small because BorderlineSMOTE already handles imbalance at data
    level — we don't want to double-penalise class-0 rules here.
    """
    if CC <= 0:
        return -9999.0
    base           = (CC - IC) / (CC + IC) + CC / (IC + 1) - IC / CC + CC / RL
    minority_bonus = lam * (CC_minority / max(total_minority, 1))
    return base + minority_bonus


def extract_top_rules(rules, top_n, total_minority):
    for r in rules:
        r["WOR"] = compute_wor(
            r["CC"], r["IC"], r["RL"], total_minority, r["CC_minority"],
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
    preds     = np.full(X.shape[0], default_class, dtype=int)
    unmatched = np.ones(X.shape[0], dtype=bool)
    for rule in ruleset:
        if not unmatched.any():
            break
        mask            = rule_match_mask(rule, X) & unmatched
        preds[mask]     = rule["predicted_class"]
        unmatched[mask] = False
    return preds


def predict_with_masks(ruleset, rule_masks, n_samples, default_class=0):
    preds     = np.full(n_samples, default_class, dtype=int)
    unmatched = np.ones(n_samples, dtype=bool)
    for rule, mask in zip(ruleset, rule_masks):
        if not unmatched.any():
            break
        apply           = mask & unmatched
        preds[apply]    = rule["predicted_class"]
        unmatched[apply] = False
    return preds


def compute_default_class(ruleset, X_train, y_train):
    """Adaptive default: majority label among unmatched training samples."""
    if not ruleset:
        return int(np.bincount(y_train).argmax())
    cols          = np.column_stack([rule_match_mask(r, X_train) for r in ruleset])
    unmatched_lbl = y_train[~np.any(cols, axis=1)]
    if len(unmatched_lbl) > 0:
        return int(Counter(unmatched_lbl).most_common(1)[0][0])
    return int(np.bincount(y_train).argmax())


# ─── 5. Macro-F1 Pruning ─────────────────────────────────────────────────────

def ruleset_macro_f1(rs, masks, y, default_class=0):
    """
    Macro-averaged F1 as the pruning objective.

    Why macro and not binary F1:
    - Binary F1 (class-1 only) caused v2 to retain all class-1 rules even
      when precision collapsed to 50% — it only cared about class-1 recall.
    - Macro-F1 averages F1 across both classes with equal weight, so the
      pruner must maintain performance on BOTH classes simultaneously.
    - This naturally finds a balance between the original's high-precision/
      low-recall and v2's high-recall/low-precision extremes.
    """
    preds = predict_with_masks(rs, masks, len(y), default_class)
    return f1_score(y, preds, average="macro", zero_division=0)


def sequential_hill_climbing_prune(init_rs, X_train, y_train):
    """Sequential Hill Climbing with macro-F1 objective and adaptive default."""
    if not init_rs:
        dc = int(np.bincount(y_train).argmax())
        return [], dc

    base_masks = [rule_match_mask(r, X_train) for r in init_rs]
    P   = init_rs.copy()
    M   = base_masks.copy()
    dc  = compute_default_class(P, X_train, y_train)
    cur = ruleset_macro_f1(P, M, y_train, dc)

    changed = True
    while changed:
        changed = False
        for i in range(len(P)):
            Pn  = [r for j, r in enumerate(P) if j != i]
            Mn  = [m for j, m in enumerate(M) if j != i]
            dcn = compute_default_class(Pn, X_train, y_train)
            new = ruleset_macro_f1(Pn, Mn, y_train, dcn)
            if new >= cur:
                cur, P, M, dc = new, Pn, Mn, dcn
                changed = True
                break
    return P, dc


# ─── 6. Full Pipeline ────────────────────────────────────────────────────────

def run_pipeline(X, y, feature_names):
    skf        = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    init_rules = []

    print(f"\n{'='*60}")
    print("Phase 1 & 2: Majority-Vote SHCK + BorderlineSMOTE + Rule Gen")
    print(f"  sampling_strategy={SAMPLING_STRATEGY}  lambda={LAMBDA_MINORITY}")
    print(f"{'='*60}")

    for fi, (tr, _) in enumerate(skf.split(X, y)):
        X_train, y_train = X[tr], y[tr]

        # BorderlineSMOTE: only synthesises near the decision boundary.
        # Guard: only apply if minority is genuinely under-represented.
        # If the dataset is already at or above the target ratio, skip resampling
        # (this is the cause of the "ratio required to remove samples" error —
        # BorderlineSMOTE cannot reduce the minority class, only grow it).
        n0_tr = int(np.sum(y_train == 0))
        n1_tr = int(np.sum(y_train == 1))
        current_ratio = n1_tr / max(n0_tr, 1)

        if current_ratio < SAMPLING_STRATEGY:
            # Minority is genuinely under-represented — safe to oversample
            try:
                bsmote       = BorderlineSMOTE(
                    random_state=RANDOM_STATE,
                    sampling_strategy=SAMPLING_STRATEGY,
                    kind="borderline-1",
                )
                X_res, y_res = bsmote.fit_resample(X_train, y_train)
            except Exception as e:
                print(f"  [Fold {fi+1}] BorderlineSMOTE failed ({e}), using raw data")
                X_res, y_res = X_train, y_train
        else:
            # Already balanced enough — use raw data, no resampling needed
            X_res, y_res = X_train, y_train

        total_minority = int(np.sum(y_res == 1))

        # Majority-vote SHCK
        sel = shck_majority_vote(X_res, y_res)

        dt = DecisionTreeClassifier(
            criterion="entropy", random_state=RANDOM_STATE,
            min_samples_leaf=MIN_SAMPLES_LEAF, max_depth=MAX_DEPTH,
        )
        dt.fit(X_res[:, sel], y_res)

        rules     = extract_rules_from_dt(dt, sel)
        top_rules = extract_top_rules(rules, TOP_N_RULES_PER_FOLD, total_minority)
        init_rules.extend(top_rules)

        c0     = sum(1 for r in top_rules if r["predicted_class"] == 0)
        c1     = sum(1 for r in top_rules if r["predicted_class"] == 1)
        n0_res = int(np.sum(y_res == 0))
        n1_res = int(np.sum(y_res == 1))
        tag    = "resampled" if len(y_res) != len(y_train) else "raw     "
        print(f"  Fold {fi+1:2d}: feats={len(sel):3d} | "
              f"rules={len(top_rules):2d} (c0={c0}, c1={c1}) | "
              f"{tag} 0={n0_res} 1={n1_res} ratio={n1_res/max(n0_res,1):.2f}")

    print(f"\nInitial Transparent Ruleset: {len(init_rules)} rules")

    # Pre-prune stats
    if init_rules:
        masks   = [rule_match_mask(r, X) for r in init_rules]
        matched = int(np.sum(np.any(np.column_stack(masks), axis=1)))
        dc_pre  = compute_default_class(init_rules, X, y)
        pre_mf1 = ruleset_macro_f1(init_rules, masks, y, dc_pre)
        preds_pre = predict_with_masks(init_rules, masks, len(y), dc_pre)
        pre_rec   = recall_score(y, preds_pre, zero_division=0)
        pre_prec  = precision_score(y, preds_pre, zero_division=0)
        print(f"  Pre-prune macro-F1={pre_mf1*100:.2f}% | "
              f"Rec={pre_rec*100:.2f}% | Prec={pre_prec*100:.2f}% | "
              f"Matched={matched}/{len(X)}")

    # Macro-F1 Sequential Hill Climbing pruning
    final, dc = sequential_hill_climbing_prune(init_rules, X, y)

    c0f = sum(1 for r in final if r["predicted_class"] == 0)
    c1f = sum(1 for r in final if r["predicted_class"] == 1)
    print(f"\n=== After Pruning ===")
    print(f"  Rules: {len(final)} (class0={c0f}, class1={c1f}) | default={dc}")
    return final, dc


# ─── 7. Display ──────────────────────────────────────────────────────────────

def _format_condition(feature_name, op, threshold):
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
    print("DiabeRules — UCI Hospital Dataset (v3)")
    print("  BorderlineSMOTE | Macro-F1 Pruning | Majority-Vote SHCK")
    print("=" * 60)

    final, dc = run_pipeline(X, y, feature_names)
    display_rules(final, feature_names)

    preds = predict_with_rules(final, X, dc)

    # Rule confidence scores for AUC
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

    acc   = accuracy_score(y, preds)
    prec  = precision_score(y, preds, zero_division=0)
    rec   = recall_score(y, preds, zero_division=0)
    f1_b  = f1_score(y, preds, zero_division=0)
    f1_m  = f1_score(y, preds, average="macro", zero_division=0)
    mcc   = matthews_corrcoef(y, preds)
    try:
        auc = roc_auc_score(y, scores)
    except Exception:
        auc = float("nan")

    print(f"\n{'='*60}")
    print("Final Model Metrics (v3)")
    print(f"{'='*60}")
    print(f"  Accuracy    : {acc*100:.2f}%")
    print(f"  Recall      : {rec*100:.2f}%")
    print(f"  Precision   : {prec*100:.2f}%")
    print(f"  F1 (binary) : {f1_b*100:.2f}%")
    print(f"  F1 (macro)  : {f1_m*100:.2f}%")
    print(f"  MCC         : {mcc:.4f}")
    print(f"  AUC-ROC     : {auc:.4f}")

    print(f"\n{'='*60}")
    print("Comparison")
    print(f"{'='*60}")
    print("            Acc    Rec    Prec   F1(bin)")
    print(f"  Original: 62.76  45.72  63.29  53.09   ← low recall")
    print(f"  v2 (bad): 53.87  89.97  49.98  64.26   ← low precision")
    print(f"  v3 (you): {acc*100:.2f}  {rec*100:.2f}  {prec*100:.2f}  {f1_b*100:.2f}   ← target balance")
    print(f"\n  Paper target (UCI): Acc=90.84, Rec=88.41, Prec=85.74, F1=87.05")
    print(f"\nNote: Paper metrics used nested CV + full preprocessing pipeline.")
    print(f"      Run with nested CV for honest generalisation estimates.")

    # ── Tuning hint ──────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Tuning hint: if recall is still too low, increase SAMPLING_STRATEGY")
    print("  (e.g. 0.7 or 0.8). If precision is too low, decrease it (e.g. 0.5).")
    print("LAMBDA_MINORITY can also be nudged: higher = more recall bias.")
    print(f"{'='*60}")