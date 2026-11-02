# ---------------------------
# Core
# ---------------------------
import os
import logging
import pickle
import warnings

warnings.filterwarnings("ignore")

# ---------------------------
# Data Manipulation
# ---------------------------
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# ---------------------------
# Machine Learning
# ---------------------------
from xgboost import XGBClassifier
import xgboost as xgb

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    precision_recall_curve
)
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer

from skopt import BayesSearchCV
from sklearn.neighbors import BallTree

# ---------------------------
# SHAP
# ---------------------------
import shap

# ---------------------------
# Statistics
# ---------------------------
from scipy.stats import chi2_contingency

# ---------------------------
# Visualization
# ---------------------------
import matplotlib.pyplot as plt

# ============================================================
#  CONFIG
# ============================================================

MIN_TRAIN_ROWS_PER_BASIN = 100_000
GLOBAL_TEST_SIZE = 0.30
RANDOM_STATE = 42

# early stopping
EARLY_STOPPING_ROUNDS = 50
EARLY_STOPPING_TEST_SIZE = 0.15

# SHAP safety (avoid blowing RAM)
MAX_SHAP_ROWS_PER_BASIN = 30_000  # set None to use full basin test

# ============================================================
#  LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


state_mapping = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas",
    "CA": "California", "CO": "Colorado", "CT": "Connecticut",
    "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas",
    "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine",
    "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan",
    "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico",
    "NY": "New York", "NC": "North Carolina", "ND": "North Dakota",
    "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island",
    "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah",
    "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia",
    "WI": "Wisconsin", "WY": "Wyoming",
}

EARTH_RADIUS_M = 6371000.0

# ============================================================
#  GENERAL UTILITIES
# ============================================================

def safe_filename(s: str) -> str:
    return str(s).replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")


def get_project_root():
    """
    Returns the root directory of the project.
    """
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython:
            notebook_dir = ipython.run_line_magic("pwd", "")
            os.chdir(notebook_dir)
            logger.debug(f"Changed working directory to {notebook_dir}")
    except Exception as e:
        logger.warning(f"Warning: Failed to change working directory {e}")
    return os.path.abspath(os.path.join(os.getcwd(), "."))


def fill_na_except_lat_lon(df, lat_col="Latitude", lon_col="Longitude"):
    df_copy = df.copy()
    for col in df_copy.columns:
        if col in [lat_col, lon_col]:
            continue
        elif df_copy[col].dtype.name in ["object", "category"]:
            df_copy[col] = df_copy[col].fillna("missing")
        else:
            df_copy[col] = df_copy[col].fillna(0)
    return df_copy


def standardize_column_values(val, mapping_dict, to_lower=False):
    if pd.isna(val) or val is None:
        return np.nan
    val_clean = str(val).strip()
    val_clean = val_clean.lower() if to_lower else val_clean.upper()
    if val_clean in mapping_dict:
        return mapping_dict[val_clean]
    return val


def engineer_date_features(df, date_column, prefix):
    df_copy = df.copy()
    if date_column not in df_copy.columns:
        return df_copy

    df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors="coerce")
    if df_copy[date_column].isnull().all():
        return df_copy

    df_copy[prefix + "_Year"] = df_copy[date_column].dt.year.fillna(0).astype(int)

    df_copy.drop(columns=[date_column], inplace=True)
    return df_copy


def best_threshold_max_f1(y_true, y_proba):
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if len(thresholds) == 0:
        return 0.5
    precision = precision[:-1]
    recall = recall[:-1]
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx])


def get_dynamic_n_iter(num_samples):
    if num_samples > 500_000:
        return 30
    elif num_samples > 250_000:
        return 25
    elif num_samples > 150_000:
        return 20
    else:
        return 15

# ============================================================
#  ENCODING UTILITIES (TRAIN ONLY)
# ============================================================

def fit_encoders_on_train_only(X_train_df, ohe_cols, multi_label_cols, global_mlb_classes):
    ohe_encoders = {}
    mlb_encoders = {}

    for col in ohe_cols:
        if col not in X_train_df.columns:
            continue
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        ohe.fit(X_train_df[col].fillna("Unknown").astype(str).to_frame())
        ohe_encoders[col] = ohe

    for col in multi_label_cols:
        if col not in X_train_df.columns:
            continue
        mlb = MultiLabelBinarizer(classes=global_mlb_classes.get(col, None))
        X_train_lists = X_train_df[col].fillna("").apply(
            lambda x: [s.strip() for s in x.split("|")] if x else []
        )
        mlb.fit(X_train_lists)
        mlb_encoders[col] = mlb

    return mlb_encoders, ohe_encoders


def transform_with_fitted_encoders(df, mlb_encoders, ohe_encoders, multi_cols, ohe_cols):
    df_out = df.copy()
    extra = []

    for col in ohe_cols:
        if col in df_out.columns and col in ohe_encoders:
            encoded = ohe_encoders[col].transform(
                df_out[col].fillna("Unknown").astype(str).to_frame()
            )
            names = ohe_encoders[col].get_feature_names_out([col])
            extra.append(pd.DataFrame(encoded, columns=names, index=df_out.index))
            df_out.drop(columns=[col], inplace=True)

    for col in multi_cols:
        if col in df_out.columns and col in mlb_encoders:
            lists = df_out[col].fillna("").apply(
                lambda x: [s.strip() for s in x.split("|")] if x else []
            )
            encoded = mlb_encoders[col].transform(lists)
            names = [f"{col}_{cls}" for cls in mlb_encoders[col].classes_]
            extra.append(pd.DataFrame(encoded, columns=names, index=df_out.index))
            df_out.drop(columns=[col], inplace=True)

    if extra:
        df_out = pd.concat([df_out] + extra, axis=1)

    return df_out

# ============================================================
#  NUMERIC MATRIX ENFORCEMENT
# ============================================================

def enforce_numeric_matrix(df):
    X = df.copy()

    bool_cols = X.select_dtypes(include=["bool"]).columns
    X[bool_cols] = X[bool_cols].astype(int)

    obj_cols = X.select_dtypes(include=["object", "category"]).columns
    for c in obj_cols:
        X[c] = X[c].astype("category").cat.codes

    for c in X.columns:
        if not np.issubdtype(X[c].dtype, np.number):
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

    return X

# ============================================================
#  TRAIN-ONLY DISTANCE TO NEAREST ORPHAN
# ============================================================

def add_distance_to_known_orphans(
    X_any: pd.DataFrame,
    known_orphans_df: pd.DataFrame,
    lat_col="Latitude",
    lon_col="Longitude",
    out_col="Distance_to_Known_Orphan_m",
    id_col=None,              # e.g., "NETL_DERIVED_API"
    known_id_col=None,        # e.g., "NETL_DERIVED_API" in known_orphans_df
    eps_m=1.0                 # treat <= 1m as "self / same point"
) -> pd.DataFrame:
    """
    Adds distance-to-nearest-known-orphan (meters) using a fixed external inventory.

    Self-exclusion:
      - If id_col/known_id_col are provided: if nearest orphan has same ID, use 2nd nearest.
      - Else: if nearest distance <= eps_m, use 2nd nearest (coords-based fallback).

    Notes:
      - Rows missing lat/lon remain NaN in out_col (you can add a missingness flag later).
      - Uses BallTree haversine on [lat, lon] in radians; converts to meters.
    """
    X = X_any.copy()

    # Guard: if no known orphans
    if known_orphans_df is None or known_orphans_df.empty:
        X[out_col] = np.nan
        return X

    # Build orphan inventory (ko) including id if available
    cols = [lat_col, lon_col]
    if known_id_col is not None and known_id_col in known_orphans_df.columns:
        cols = [known_id_col] + cols

    ko = known_orphans_df[cols].dropna(subset=[lat_col, lon_col]).copy()

    # Optional: dedupe by ID if provided (prevents weird "self" behavior with duplicates)
    if known_id_col is not None and known_id_col in ko.columns:
        ko = ko.drop_duplicates(subset=[known_id_col])

    # Always reset index so idx from BallTree aligns to ko.iloc
    ko = ko.reset_index(drop=True)

    if ko.empty:
        X[out_col] = np.nan
        return X

    # BallTree expects radians [lat, lon]
    ko_rad = np.deg2rad(ko[[lat_col, lon_col]].astype(float).values)
    tree = BallTree(ko_rad, metric="haversine")

    # Prepare X coords
    X_coords = X[[lat_col, lon_col]].copy()
    X_coords[lat_col] = pd.to_numeric(X_coords[lat_col], errors="coerce")
    X_coords[lon_col] = pd.to_numeric(X_coords[lon_col], errors="coerce")

    mask_valid = X_coords.notna().all(axis=1)
    X_rad = np.deg2rad(X_coords.loc[mask_valid, [lat_col, lon_col]].values.astype(float))

    # If no valid rows, just return all NaNs
    X[out_col] = np.nan
    if X_rad.shape[0] == 0:
        return X

    # Query 2 nearest to enable self-exclusion; fallback to 1 if only 1 orphan exists
    k = 2 if ko_rad.shape[0] >= 2 else 1
    dist_rad, idx = tree.query(X_rad, k=k)

    # Convert to meters
    dist_m = dist_rad * EARTH_RADIUS_M

    # Start with nearest distance
    chosen = dist_m[:, 0].copy()

    # Self-exclusion logic if we have 2 neighbors
    if k == 2:
        if (
            id_col is not None
            and known_id_col is not None
            and id_col in X.columns
            and known_id_col in ko.columns
        ):
            nearest_orphan_ids = ko.iloc[idx[:, 0]][known_id_col].astype(str).values
            row_ids = X.loc[mask_valid, id_col].astype(str).values

            is_self = (row_ids == nearest_orphan_ids)
            chosen[is_self] = dist_m[is_self, 1]
        else:
            near_zero = chosen <= eps_m
            chosen[near_zero] = dist_m[near_zero, 1]

    # Write back
    X.loc[mask_valid, out_col] = chosen
    return X

# ============================================================
#  SHAP EXPLAINER
# ============================================================

def explain_with_shap(model, X_test, max_display=20, return_df=False, plot_filename=None):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test)

    shap.summary_plot(shap_values, X_test, max_display=max_display, show=False)

    fig = plt.gcf()
    if plot_filename:
        output_dir = os.path.dirname(plot_filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        fig.set_size_inches(10, 8)
        plt.savefig(plot_filename, bbox_inches="tight", dpi=300)

    plt.close(fig)

    if return_df:
        shap_importance = np.abs(shap_values.values).mean(axis=0)
        return (
            pd.DataFrame({"Feature": X_test.columns, "Mean_Abs_SHAP": shap_importance})
              .sort_values(by="Mean_Abs_SHAP", ascending=False)
        )

    return None

# ============================================================
# Load & Prep DataFrame
# ============================================================

project_root = get_project_root()

File_path = os.path.join(project_root, "Raw CSV Files", "ML_Ready_df", "Prediction_DF_Final.csv")
merged_df = pd.read_csv(File_path, low_memory=False)

y = merged_df["Orphaned"]

X = merged_df.drop(
    columns=[
        "Orphaned",
        "Distance_to_Known_Orphan_m"
    ],
    axis=1,
    errors="ignore",
)

logger.info(f"Initial X DataFrame shape: {X.shape}")

# Feature engineering
columns_to_check = X.columns.drop("NETL_DERIVED_API") if "NETL_DERIVED_API" in X.columns else X.columns
X["nan_count_per_row"] = X[columns_to_check].isna().sum(axis=1)
X["nan_proportion_per_row"] = X["nan_count_per_row"] / max(len(columns_to_check), 1)
X = X.drop(columns=["nan_count_per_row"], errors="ignore")

X = fill_na_except_lat_lon(X)

X = engineer_date_features(X, "Status_Date", "Status")
X = engineer_date_features(X, "Completion_Date", "Completion")
X = engineer_date_features(X, "Spud_Date", "Spud")

# States
if "State" in X.columns:
    X["States_Updated"] = X["State"].apply(lambda x: standardize_column_values(x, state_mapping, to_lower=False))
    X["States_Updated"] = X["States_Updated"].replace("", "Unknown")
    X = X.rename(columns={"States_Updated": "States"})


# Drop columns
X = X.drop(
    columns=["State", "Township", "County", "Longitude_Bottom_Hole_Location", "Latitude_Bottom_Hole_Location"],
    errors="ignore"
)

# Fill Nan Basin_Name
X['Basin_Name'] = X['Basin_Name'].replace('missing', 'UNKNOWN BASIN NAME')

if "Basin_Name" not in X.columns:
    raise ValueError("'Basin_Name' column not found in X DataFrame. Required for basin-level analysis.")

y = y.reindex(X.index).astype(int)

X_no_api = X.drop(columns=["NETL_DERIVED_API", "Status_Category"], errors="ignore")
logger.info(f"X_no_api DataFrame shape after feature engineering and drops: {X_no_api.shape}")

# Fallback for calculating distance
X_raw_for_distance = X.copy()
logger.info("Developed dataframe to calculate distance to known orphans. This will be used for a leak-safe feature in the basin loop.")

logger.info("Completed data loading and initial preprocessing steps.")


# ============================================================
# GLOBAL TRAIN/TEST SPLIT
# ============================================================

X_global = X_no_api.copy()
y_global = y.reindex(X_global.index).astype(int)

X_train_global_raw, X_test_global_raw, y_train_global, y_test_global = train_test_split(
    X_global,
    y_global,
    test_size=GLOBAL_TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y_global,
)

# align distance-raw using the SAME indices
X_train_global_distance = X_raw_for_distance.loc[X_train_global_raw.index]

known_orphans_train = (
    X_train_global_distance.loc[y_train_global == 1, ["NETL_DERIVED_API", "Latitude", "Longitude"]]
    .dropna(subset = ['Latitude','Longitude'])
    .copy()
)

X_test_global_distance  = X_raw_for_distance.loc[X_test_global_raw.index]

logger.info(f"GLOBAL split: train={X_train_global_raw.shape}, test={X_test_global_raw.shape}")

# ============================================================
# Encoding setup (GLOBAL TRAIN ONLY)
# ============================================================

ohe_cols = ["States"]
multi_label_cols = ["Angle_Category", "Type_Category","land_cover"]

def to_lists(s):
    return s.fillna("").apply(lambda x: [t.strip() for t in x.split("|")] if x else [])

global_mlb_classes = {}
for col in multi_label_cols:
    if col not in X_train_global_raw.columns:
        global_mlb_classes[col] = []
        continue
    all_tokens = pd.Series([tok for lst in to_lists(X_train_global_raw[col]) for tok in lst])
    global_mlb_classes[col] = sorted(all_tokens[all_tokens != ""].unique().tolist())


unique_basins = X_train_global_raw["Basin_Name"].unique()
logger.info(f"Processing per-basin models for {len(unique_basins)} basins (global split).")

mlb_encoders_global, ohe_encoders_global = fit_encoders_on_train_only(
    X_train_global_raw,
    ohe_cols=ohe_cols,
    multi_label_cols=multi_label_cols,
    global_mlb_classes=global_mlb_classes
)

X_train_global_enc = transform_with_fitted_encoders(
    X_train_global_raw,
    mlb_encoders=mlb_encoders_global,
    ohe_encoders=ohe_encoders_global,
    multi_cols=multi_label_cols,
    ohe_cols=ohe_cols
)
X_test_global_enc = transform_with_fitted_encoders(
    X_test_global_raw,
    mlb_encoders=mlb_encoders_global,
    ohe_encoders=ohe_encoders_global,
    multi_cols=multi_label_cols,
    ohe_cols=ohe_cols
)

X_train_global_enc = enforce_numeric_matrix(X_train_global_enc)
X_test_global_enc = enforce_numeric_matrix(X_test_global_enc)

logger.info(f"Encoded GLOBAL train/test: {X_train_global_enc.shape} / {X_test_global_enc.shape}")

# ============================================================
# Outputs / dirs
# ============================================================

trained_models = {}
model_performance_metrics = {}
best_params_per_basin = {}
shap_dataset = {}

plot_dir = os.path.join(project_root, "Output Plots")
os.makedirs(plot_dir, exist_ok=True)

shap_output_dir = os.path.join(plot_dir, "shap_plots")
os.makedirs(shap_output_dir, exist_ok=True)

plot_output_dir = os.path.join(plot_dir, "threshold_plots")
os.makedirs(plot_output_dir, exist_ok=True)

classification_report_dir = os.path.join(plot_dir, "Classification Reports")
os.makedirs(classification_report_dir, exist_ok=True)

# Fixed XGB settings (applied everywhere)
XGB_FIXED = dict(
    objective="binary:logistic",
    eval_metric="aucpr",
    tree_method="hist",
    device="cuda",
    random_state=RANDOM_STATE,
    enable_categorical=False,
    base_score=0.5
)

# ============================================================
# Basin loop
# ============================================================

for basin_name in unique_basins:
    processed_name = basin_name
    safe_name = safe_filename(processed_name)

    try:
        # 1) Basin masks on RAW
        train_mask = X_train_global_raw["Basin_Name"] == processed_name
        test_mask = X_test_global_raw["Basin_Name"] == processed_name

        # Apply to distance frames (same index)
        X_train_raw_basin = X_train_global_distance.loc[X_train_global_raw.index[train_mask]].copy()
        X_test_raw_basin  = X_test_global_distance.loc[X_test_global_raw.index[test_mask]].copy()

        y_train_basin = y_train_global.loc[X_train_raw_basin.index].astype(int)
        y_test_basin = y_test_global.loc[X_test_raw_basin.index].astype(int)

        n_train = int(X_train_raw_basin.shape[0])
        n_test = int(X_test_raw_basin.shape[0])

        # 2) Skip logic
        if n_train <= MIN_TRAIN_ROWS_PER_BASIN:
            logger.info(f"Skipping {processed_name}: basin train={n_train} below {MIN_TRAIN_ROWS_PER_BASIN}.")
            model_performance_metrics[processed_name] = {
                "status": "skipped_below_threshold",
                "n_train": n_train,
                "n_test": n_test
            }
            best_params_per_basin[processed_name] = {"status": "skipped_below_threshold"}
            continue

        class_counts_train = y_train_basin.value_counts()
        if class_counts_train.nunique() < 2:
            logger.info(f"Skipping {processed_name}: single-class TRAIN. Counts={class_counts_train.to_dict()}")
            model_performance_metrics[processed_name] = {
                "status": "skipped_single_class_train",
                "train_class_counts": class_counts_train.to_dict(),
                "n_train": n_train,
                "n_test": n_test
            }
            best_params_per_basin[processed_name] = {"status": "skipped_single_class_train"}
            continue

        logger.info(f"Processing basin {processed_name}: train={n_train}, test={n_test}")

        # 3) CV setup (TRAIN only)
        min_class = int(class_counts_train.min())
        n_splits = 5 if min_class >= 5000 else 3
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        if min_class < cv.n_splits:
            logger.warning(
                f"Skipping {processed_name}: insufficient minority samples for CV. "
                f"Counts={class_counts_train.to_dict()}, n_splits={cv.n_splits}"
            )
            model_performance_metrics[processed_name] = {
                "status": "skipped_insufficient_minority_for_cv",
                "train_class_counts": class_counts_train.to_dict(),
                "n_train": n_train,
                "n_test": n_test
            }
            best_params_per_basin[processed_name] = {"status": "skipped_insufficient_minority_for_cv"}
            continue

        # 4) Pull basin MODEL matrices from GLOBAL-ENCODED frames
        X_train_model = X_train_global_enc.loc[X_train_raw_basin.index].copy()
        X_test_model = X_test_global_enc.loc[X_test_raw_basin.index].copy()

        X_train_model.drop(columns=["Basin_Name"], inplace=True, errors="ignore")
        X_test_model.drop(columns=["Basin_Name"], inplace=True, errors="ignore")

        # 5) Leak-safe distance (TRAIN orphans only) from RAW
        X_train_raw_basin_tmp = add_distance_to_known_orphans(
            X_train_raw_basin,
            known_orphans_df=known_orphans_train,
            lat_col="Latitude",
            lon_col="Longitude",
            out_col="Distance_to_Known_Orphan_m",
            id_col="NETL_DERIVED_API",
            known_id_col="NETL_DERIVED_API",
            eps_m=1.0
        )

        X_test_raw_basin_tmp = add_distance_to_known_orphans(
            X_test_raw_basin,
            known_orphans_df=known_orphans_train,
            lat_col="Latitude",
            lon_col="Longitude",
            out_col="Distance_to_Known_Orphan_m",
            id_col="NETL_DERIVED_API",
            known_id_col="NETL_DERIVED_API",
            eps_m=1.0
        )

        X_train_model["Distance_to_Known_Orphan_m"] = (
            X_train_raw_basin_tmp["Distance_to_Known_Orphan_m"].astype(float)
        )
        X_test_model["Distance_to_Known_Orphan_m"] = (
            X_test_raw_basin_tmp["Distance_to_Known_Orphan_m"].astype(float)
        )

        # add missingness flag
        X_train_model["Distance_to_Known_Orphan_missing"] = X_train_model["Distance_to_Known_Orphan_m"].isna().astype(int)
        X_test_model["Distance_to_Known_Orphan_missing"]  = X_test_model["Distance_to_Known_Orphan_m"].isna().astype(int)

        # fill NaNs with something conservative (e.g., max observed train distance)
        d = X_train_model["Distance_to_Known_Orphan_m"].to_numpy()

        fill_value = float(np.nanmax(d[np.isfinite(d)])) if np.any(np.isfinite(d)) else 50_000

        X_train_model["Distance_to_Known_Orphan_m"] = X_train_model["Distance_to_Known_Orphan_m"].fillna(fill_value)
        X_test_model["Distance_to_Known_Orphan_m"]  = X_test_model["Distance_to_Known_Orphan_m"].fillna(fill_value)

        X_train_model["Log_Distance_to_Known_Orphan"] = np.log1p(X_train_model["Distance_to_Known_Orphan_m"])
        X_test_model["Log_Distance_to_Known_Orphan"]  = np.log1p(X_test_model["Distance_to_Known_Orphan_m"])

        # drop lat/lon (optional)
        X_train_model.drop(columns=["Latitude", "Longitude"], inplace=True, errors="ignore")
        X_test_model.drop(columns=["Latitude", "Longitude"], inplace=True, errors="ignore")

        # Drop API (optional)
        X_train_model.drop(columns=["NETL_DERIVED_API"], inplace=True, errors="ignore")
        X_test_model.drop(columns=["NETL_DERIVED_API"], inplace=True, errors="ignore") 

        # Drop Distance to known orphan (optional, since we have log and missing flag)
        X_train_model.drop(columns=["Distance_to_Known_Orphan_m"], inplace=True, errors="ignore")
        X_test_model.drop(columns=["Distance_to_Known_Orphan_m"], inplace=True, errors="ignore")

        # Align columns (safety)
        X_train_model, X_test_model = X_train_model.align(X_test_model, join="left", axis=1, fill_value=0)
        X_test_model = X_test_model[X_train_model.columns]

        # 6) scale_pos_weight (TRAIN only)
        pos = int((y_train_basin == 1).sum())
        neg = int((y_train_basin == 0).sum())
        scale_pos_weight_value = (neg / pos) if pos > 0 else 1.0

        # 7) BayesSearchCV (TRAIN only)
        param_space_basin = {
            "n_estimators": (200, 1200),  # allow early stopping to work
            "max_depth": (3, 15),
            "learning_rate": (0.01, 0.3, "log-uniform"),
            "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0),
            "gamma": (0, 5),
            "reg_lambda": (0, 5),
            "reg_alpha": (0, 5),
            "scale_pos_weight": [scale_pos_weight_value],
        }

        clf_xgb_search = XGBClassifier(**XGB_FIXED)

        current_n_iter = get_dynamic_n_iter(int(X_train_model.shape[0]))
        logger.info(f"Running BayesSearchCV for {processed_name} with n_iter={current_n_iter}")

        search_basin = BayesSearchCV(
            estimator=clf_xgb_search,
            search_spaces=param_space_basin,
            scoring="average_precision",
            n_iter=current_n_iter,
            cv=cv,
            n_jobs=-1,
            verbose=0,
            random_state=RANDOM_STATE,
        )

        search_basin.fit(X_train_model, y_train_basin)
        best_params = search_basin.best_estimator_.get_params()

        # lock fixed params ALWAYS
        best_params.update(XGB_FIXED)
        best_params_per_basin[processed_name] = best_params

        # 8) OOF predictions for threshold tuning (TRAIN only)
        X_train_np = X_train_model.values
        y_train_np = y_train_basin.values
        oof_proba = np.zeros(X_train_np.shape[0], dtype=float)

        for tr_idx, va_idx in cv.split(X_train_np, y_train_np):
            X_tr_full = X_train_np[tr_idx]
            y_tr_full = y_train_np[tr_idx]

            # early stopping split ONLY within fold-train
            try:
                X_tr, X_es, y_tr, y_es = train_test_split(
                    X_tr_full,
                    y_tr_full,
                    test_size=EARLY_STOPPING_TEST_SIZE,
                    random_state=RANDOM_STATE,
                    stratify=y_tr_full
                )
            except ValueError:
                # fallback if stratify fails
                X_tr, X_es, y_tr, y_es = train_test_split(
                    X_tr_full,
                    y_tr_full,
                    test_size=EARLY_STOPPING_TEST_SIZE,
                    random_state=RANDOM_STATE,
                    stratify=None
                )

            fold_model = XGBClassifier(**best_params)
            fold_model.fit(
                X_tr, y_tr,
                eval_set=[(X_es, y_es)],
                verbose=False,
                callbacks=[xgb.callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS, save_best=True)]
            )

            oof_proba[va_idx] = fold_model.predict_proba(X_train_np[va_idx])[:, 1]

        best_threshold = best_threshold_max_f1(y_train_basin, oof_proba)
        logger.info(f"OOF-tuned threshold for {processed_name}: {best_threshold:.4f}")

        # 9) Threshold curve plot
        p_oof, r_oof, t_oof = precision_recall_curve(y_train_basin, oof_proba)

        if len(t_oof) > 0:
            p2, r2 = p_oof[:-1], r_oof[:-1]
            f1_curve = 2 * p2 * r2 / (p2 + r2 + 1e-12)

            plt.figure(figsize=(10, 6))
            plt.plot(t_oof, p2, label="Precision (OOF-TRAIN)", linewidth=2)
            plt.plot(t_oof, r2, label="Recall (OOF-TRAIN)", linewidth=2)
            plt.plot(t_oof, f1_curve, label="F1 (OOF-TRAIN)", linestyle="--", linewidth=2, alpha=0.7)
            plt.axvline(best_threshold, linestyle="--", label=f"OOF Best Thresh: {best_threshold:.4f}")

            plt.title(f"Precision/Recall/F1 vs Threshold (OOF TRAIN) | Basin: {processed_name}")
            plt.xlabel("Threshold")
            plt.ylabel("Score")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            oof_plot_filename = os.path.join(plot_output_dir, f"threshold_curve_OOF_TRAIN_{safe_name}_HC.jpeg")
            plt.savefig(oof_plot_filename, bbox_inches="tight", dpi=300)
            plt.close()

        # 10) Final model fit on basin TRAIN
        try:
            X_final_tr, X_final_es, y_final_tr, y_final_es = train_test_split(
                X_train_model,
                y_train_basin,
                test_size=EARLY_STOPPING_TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y_train_basin
            )
        except ValueError:
            X_final_tr, X_final_es, y_final_tr, y_final_es = train_test_split(
                X_train_model,
                y_train_basin,
                test_size=EARLY_STOPPING_TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=None
            )

        final_basin_model = XGBClassifier(**best_params)
        final_basin_model.fit(
            X_final_tr, y_final_tr,
            eval_set=[(X_final_es, y_final_es)],
            verbose=False,
            callbacks=[xgb.callback.EarlyStopping(rounds=EARLY_STOPPING_ROUNDS, save_best=True)]
        )

        trained_models[processed_name] = final_basin_model

        # 11) Evaluate ONCE on basin TEST
        basin_metrics = {
            "threshold": float(best_threshold),
            "n_train": int(X_train_model.shape[0]),
            "n_test": int(X_test_model.shape[0]),
        }

        if X_test_model.shape[0] == 0:
            basin_metrics["status"] = "no_test_rows_in_global_holdout"
        elif y_test_basin.nunique() < 2:
            basin_metrics["status"] = "single_class_in_test"
            basin_metrics["test_class_counts"] = y_test_basin.value_counts().to_dict()
        else:
            y_proba_test = final_basin_model.predict_proba(X_test_model)[:, 1]
            y_pred_test = (y_proba_test >= best_threshold).astype(int)

            basin_metrics.update({
                "precision": precision_score(y_test_basin, y_pred_test, zero_division=0),
                "recall": recall_score(y_test_basin, y_pred_test, zero_division=0),
                "f1_score": f1_score(y_test_basin, y_pred_test, zero_division=0),
                "roc_auc": roc_auc_score(y_test_basin, y_proba_test),
                "pr_auc": average_precision_score(y_test_basin, y_proba_test),
            })

            
            report = classification_report(y_test_basin, y_pred_test, zero_division=0, output_dict = True)
            
            df_metric = pd.DataFrame([basin_metrics]).transpose()
            df_report = pd.DataFrame(report).transpose()
                        
            # logger.info(f"Metrics for {processed_name}: {metric}")
            # logger.info(
            #     f"Classification report for {processed_name}:\n"
            #     f"{report}"
            # )
            
            logger.info("Saving classification report and basin metrics to CSV files...")
            
            # Convert basin metrics and classification report to CSV 
            df_report_path = os.path.join(classification_report_dir,f"{processed_name}_classification report_HC.csv")
            df_metric_path = os.path.join(classification_report_dir,f"{processed_name}_metrics_HC.csv")
            df_report.to_csv(df_report_path)
            df_metric.to_csv(df_metric_path)

            logger.info(f"Successfully saved {processed_name} metrics to CSV dir at: {classification_report_dir}")

        model_performance_metrics[processed_name] = basin_metrics

        # 12) Store for SHAP (with sampling safety)
        X_shap = X_test_model
        if MAX_SHAP_ROWS_PER_BASIN is not None and X_shap.shape[0] > MAX_SHAP_ROWS_PER_BASIN:
            X_shap = X_shap.sample(n=MAX_SHAP_ROWS_PER_BASIN, random_state=RANDOM_STATE)

        shap_dataset[processed_name] = {
            "model": final_basin_model,
            "X_test_data": X_shap,
            "best_params": best_params,
        }

    except Exception as e:
        logger.exception(f"Error in basin pipeline for {processed_name}: {e}")
        model_performance_metrics[processed_name] = {"status": f"error_{str(e)}"}
        best_params_per_basin[processed_name] = {"status": f"error_{str(e)}"}
        continue

# =====================
# Save outputs to pickle
# =====================
list_dict = [trained_models, model_performance_metrics, best_params_per_basin, shap_dataset]
list_dict_names = ["trained_models_HC", "model_performance_metrics_HC", "best_params_per_basin_HC", "shap_dataset_HC"]

for name, dict_object in zip(list_dict_names, list_dict):
    output_filepath = os.path.join(project_root, f"{name}.pkl")
    logger.info(f"Saving {name}.pkl to {output_filepath}")
    with open(output_filepath, "wb") as file:
        pickle.dump(dict_object, file)

# =====================
# SHAP per basin (holdout)
# =====================
for k, v in shap_dataset.items():
    logger.info(f"\nProcessing SHAP for basin: {k}")
    plot_filename = os.path.join(shap_output_dir, f"shap_summary_{safe_filename(k)}_HC.jpeg")
    _ = explain_with_shap(
        v["model"],
        v["X_test_data"],
        return_df=False,
        plot_filename=plot_filename,
    )
    logger.info(f"SHAP done for basin: {k}")