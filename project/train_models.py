"""
train_models.py — Salary Analysis System
=========================================
Run this ONCE before starting the Flask app.

Steps:
  1. Load both CSV files
  2. Clean & engineer features
  3. Train salary prediction models (Random Forest + Linear Regression)
  4. Save models + encoders + processed data to models/

Usage:
    python train_models.py
"""

import os, pickle, time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

BASE   = os.path.dirname(__file__)
DATA   = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

print("\n" + "="*60)
print("  SALARY ANALYSIS — MODEL TRAINING")
print("="*60)

# ── Step 1: Load data ─────────────────────────────────────────
print("\n[1/5] Loading CSV files...")
t0 = time.time()

df_priv = pd.read_csv(os.path.join(DATA, "Private_Sector_Salaries.csv"))
df_govt = pd.read_csv(os.path.join(DATA, "government__Sector_Salaries.csv"))

df_priv["Sector"] = "Private"
df_govt["Sector"] = "Government"

df = pd.concat([df_priv, df_govt], ignore_index=True)
print(f"  Private : {len(df_priv):,} rows")
print(f"  Govt    : {len(df_govt):,} rows")
print(f"  Combined: {len(df):,} rows  ({time.time()-t0:.1f}s)")

# ── Step 2: Clean ─────────────────────────────────────────────
print("\n[2/5] Cleaning & engineering features...")

# Fill Grade nulls with mode per sector
df["Grade"] = df["Grade"].fillna(df.groupby("Sector")["Grade"].transform(lambda x: x.mode()[0] if not x.mode().empty else "UNK"))

# Remove extreme outliers (salary < $10K or > $400K)
before = len(df)
df = df[(df["Base_Salary"] >= 10000) & (df["Base_Salary"] <= 400000)]
print(f"  Removed {before - len(df)} outlier rows")

# Feature engineering
df["Total_Pay"]       = df["Base_Salary"] + df["Overtime_Pay"] + df["Longevity_Pay"]
df["Overtime_Flag"]   = (df["Overtime_Pay"] > 0).astype(int)
df["Longevity_Flag"]  = (df["Longevity_Pay"] > 0).astype(int)
df["Overtime_Ratio"]  = (df["Overtime_Pay"] / df["Base_Salary"].replace(0, np.nan)).fillna(0).round(4)

# Salary band quartiles based on combined dataset
q25, q50, q75 = df["Base_Salary"].quantile([0.25, 0.50, 0.75])
def salary_band(s):
    if s < q25:  return "Low"
    if s < q50:  return "Mid-Low"
    if s < q75:  return "Mid-High"
    return "High"
df["Salary_Band"] = df["Base_Salary"].apply(salary_band)

# Dept avg salary
dept_avg = df.groupby("Department")["Base_Salary"].transform("mean").round(2)
df["Dept_Avg_Salary"] = dept_avg

# Dept+gender avg
df["Dept_Gender_Avg"] = df.groupby(["Department","Gender"])["Base_Salary"].transform("mean").round(2)

# Numeric grade extraction
def grade_to_num(g):
    try:
        return int(''.join(filter(str.isdigit, str(g))) or 0)
    except:
        return 0
df["Grade_Numeric"] = df["Grade"].apply(grade_to_num)

# Gender binary
df["Gender_Encoded"] = (df["Gender"] == "M").astype(int)

# Sector binary
df["Sector_Encoded"] = (df["Sector"] == "Private").astype(int)

print(f"  Engineered 10 derived features")
print(f"  Salary band cutoffs: Q25=${q25:,.0f}  Q50=${q50:,.0f}  Q75=${q75:,.0f}")

# ── Step 3: Encode categoricals ───────────────────────────────
print("\n[3/5] Encoding categorical features...")

encoders = {}
for col in ["Department", "Department_Name", "Division", "Grade", "Sector", "Salary_Band"]:
    le = LabelEncoder()
    df[col + "_enc"] = le.fit_transform(df[col].astype(str))
    encoders[col] = le
    print(f"  {col}: {len(le.classes_)} classes")

# ── Step 4: Train models ──────────────────────────────────────
print("\n[4/5] Training salary prediction models...")

FEATURES = [
    "Department_enc", "Department_Name_enc", "Gender_Encoded",
    "Grade_Numeric", "Sector_Encoded", "Overtime_Flag",
    "Longevity_Flag", "Dept_Avg_Salary", "Grade_enc"
]
TARGET = "Base_Salary"

X = df[FEATURES].values
y = df[TARGET].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}")

models_trained = {}

# Random Forest
print("  Training Random Forest...")
rf = RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_leaf=5,
                            random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_rmse  = np.sqrt(mean_squared_error(y_test, rf_preds))
rf_mae   = mean_absolute_error(y_test, rf_preds)
rf_r2    = r2_score(y_test, rf_preds)
models_trained["random_forest"] = {"model": rf, "rmse": rf_rmse, "mae": rf_mae, "r2": rf_r2}
print(f"    RMSE=${rf_rmse:,.0f}  MAE=${rf_mae:,.0f}  R²={rf_r2:.4f}")

# Ridge Regression
print("  Training Ridge Regression...")
scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)
ridge = Ridge(alpha=10.0)
ridge.fit(X_train_s, y_train)
ridge_preds = ridge.predict(X_test_s)
ridge_rmse  = np.sqrt(mean_squared_error(y_test, ridge_preds))
ridge_mae   = mean_absolute_error(y_test, ridge_preds)
ridge_r2    = r2_score(y_test, ridge_preds)
models_trained["ridge"] = {"model": ridge, "scaler": scaler, "rmse": ridge_rmse, "mae": ridge_mae, "r2": ridge_r2}
print(f"    RMSE=${ridge_rmse:,.0f}  MAE=${ridge_mae:,.0f}  R²={ridge_r2:.4f}")

# Best model
best = max(models_trained, key=lambda k: models_trained[k]["r2"])
print(f"\n  Best model: {best} (R²={models_trained[best]['r2']:.4f})")

# Feature importances (RF)
feat_imp = dict(zip(FEATURES, rf.feature_importances_.round(4)))
feat_imp_sorted = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))

# ── Step 5: Precompute analytics ──────────────────────────────
print("\n[5/5] Precomputing analytics for dashboard...")

# Dashboard KPIs
kpis = {
    "total_employees":   int(len(df)),
    "private_count":     int(len(df[df["Sector"]=="Private"])),
    "govt_count":        int(len(df[df["Sector"]=="Government"])),
    "private_avg_base":  round(df[df["Sector"]=="Private"]["Base_Salary"].mean(), 2),
    "govt_avg_base":     round(df[df["Sector"]=="Government"]["Base_Salary"].mean(), 2),
    "private_avg_total": round(df[df["Sector"]=="Private"]["Total_Pay"].mean(), 2),
    "govt_avg_total":    round(df[df["Sector"]=="Government"]["Total_Pay"].mean(), 2),
    "private_avg_ot":    round(df[df["Sector"]=="Private"]["Overtime_Pay"].mean(), 2),
    "govt_avg_ot":       round(df[df["Sector"]=="Government"]["Overtime_Pay"].mean(), 2),
    "overall_avg":       round(df["Base_Salary"].mean(), 2),
    "q25": round(q25, 2), "q50": round(q50, 2), "q75": round(q75, 2),
}

# Gender pay gap
gender_gap = {}
for sector in ["Private", "Government"]:
    sub = df[df["Sector"]==sector]
    m_avg = sub[sub["Gender"]=="M"]["Base_Salary"].mean()
    f_avg = sub[sub["Gender"]=="F"]["Base_Salary"].mean()
    gender_gap[sector] = {
        "M_avg": round(m_avg, 2),
        "F_avg": round(f_avg, 2),
        "gap":   round(m_avg - f_avg, 2),
        "gap_pct": round(((m_avg - f_avg) / f_avg) * 100, 2)
    }

# Dept salary averages
dept_salary = df.groupby(["Sector","Department","Department_Name"]).agg(
    avg_base=("Base_Salary","mean"),
    avg_total=("Total_Pay","mean"),
    count=("Base_Salary","count"),
    avg_overtime=("Overtime_Pay","mean"),
).reset_index().round(2)

# Grade salary mapping
grade_salary = df.groupby(["Sector","Grade"]).agg(
    avg_base=("Base_Salary","mean"),
    count=("Base_Salary","count"),
).reset_index().sort_values("avg_base", ascending=False).round(2)

# Salary distribution for histogram (bins of 10K)
bins = list(range(0, 310000, 10000))
for sector in ["Private", "Government"]:
    sub = df[df["Sector"]==sector]["Base_Salary"]
    hist, edges = np.histogram(sub, bins=bins)

# Overtime analysis by dept
ot_analysis = df[df["Overtime_Pay"]>0].groupby(["Sector","Department"]).agg(
    avg_overtime=("Overtime_Pay","mean"),
    pct_with_ot=("Overtime_Flag","mean"),
    count=("Overtime_Pay","count"),
).reset_index().round(2)

# Longevity analysis
lon_analysis = df[df["Longevity_Pay"]>0].groupby(["Sector","Department"]).agg(
    avg_longevity=("Longevity_Pay","mean"),
    pct_with_lon=("Longevity_Flag","mean"),
).reset_index().round(2)

# Top/bottom departments
top_depts = df.groupby(["Sector","Department_Name"])["Base_Salary"].mean().reset_index()
top_depts.columns = ["Sector","Department_Name","avg_salary"]
top_depts = top_depts.sort_values("avg_salary", ascending=False).round(2)

# Full cleaned dataframe (for table/export)
export_cols = ["Sector","Department","Department_Name","Division","Gender",
               "Base_Salary","Overtime_Pay","Longevity_Pay","Total_Pay",
               "Grade","Salary_Band","Overtime_Flag","Longevity_Flag"]

# ── Save everything ───────────────────────────────────────────
print("\n  Saving artefacts to models/...")

with open(os.path.join(MODELS, "models.pkl"), "wb") as f:
    pickle.dump(models_trained, f)

with open(os.path.join(MODELS, "encoders.pkl"), "wb") as f:
    pickle.dump(encoders, f)

with open(os.path.join(MODELS, "analytics.pkl"), "wb") as f:
    pickle.dump({
        "kpis":         kpis,
        "gender_gap":   gender_gap,
        "dept_salary":  dept_salary.to_dict("records"),
        "grade_salary": grade_salary.to_dict("records"),
        "ot_analysis":  ot_analysis.to_dict("records"),
        "lon_analysis": lon_analysis.to_dict("records"),
        "top_depts":    top_depts.to_dict("records"),
        "feat_imp":     feat_imp_sorted,
        "features":     FEATURES,
        "best_model":   best,
        "model_metrics":{ k: {kk:vv for kk,vv in v.items() if kk != "model" and kk != "scaler"}
                          for k,v in models_trained.items() },
        "salary_bands": {"Q25": round(q25,2), "Q50": round(q50,2), "Q75": round(q75,2)},
    }, f)

df[export_cols].to_pickle(os.path.join(MODELS, "processed_data.pkl"))

print("  Saved: models.pkl, encoders.pkl, analytics.pkl, processed_data.pkl")

print("\n" + "="*60)
print("  TRAINING COMPLETE")
print("="*60)
print(f"  Random Forest  — R²={rf_r2:.4f}  RMSE=${rf_rmse:,.0f}  MAE=${rf_mae:,.0f}")
print(f"  Ridge Regress  — R²={ridge_r2:.4f}  RMSE=${ridge_rmse:,.0f}  MAE=${ridge_mae:,.0f}")
print(f"  Best model     : {best}")
print(f"  Total records  : {len(df):,}")
print("="*60)
print("\nRun: python app.py\n")