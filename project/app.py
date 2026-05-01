"""
Salary Analysis System — Flask App
====================================
10 features:
  1. Dashboard & KPI overview
  2. Cross-sector salary comparison
  3. Gender pay gap analysis
  4. Department salary explorer
  5. Pay grade analysis
  6. Salary distribution charts
  7. Overtime analysis
  8. Longevity pay tracker
  9. Salary prediction model
 10. Total compensation calculator
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import os, pickle

app = Flask(__name__)

BASE   = os.path.dirname(__file__)
DATA   = os.path.join(BASE, "data")
MODELS = os.path.join(BASE, "models")

# ── Load artefacts ─────────────────────────────────────────────
print("Loading models and analytics...")

with open(os.path.join(MODELS, "models.pkl"),   "rb") as f:
    trained_models = pickle.load(f)
with open(os.path.join(MODELS, "encoders.pkl"), "rb") as f:
    encoders = pickle.load(f)
with open(os.path.join(MODELS, "analytics.pkl"),"rb") as f:
    analytics = pickle.load(f)

df = pd.read_pickle(os.path.join(MODELS, "processed_data.pkl"))
print(f"Loaded {len(df):,} records. Ready.")

# ── Helper ─────────────────────────────────────────────────────
def safe_float(v):
    try: return round(float(v), 2)
    except: return 0.0

# ── Pages ──────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/comparison")
def comparison():
    return render_template("comparison.html")

@app.route("/gender")
def gender():
    return render_template("gender.html")

@app.route("/departments")
def departments():
    return render_template("departments.html")

@app.route("/grades")
def grades():
    return render_template("grades.html")

@app.route("/distribution")
def distribution():
    return render_template("distribution.html")

@app.route("/overtime")
def overtime():
    return render_template("overtime.html")

@app.route("/longevity")
def longevity():
    return render_template("longevity.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/calculator")
def calculator():
    return render_template("calculator.html")

# ── API: Dashboard KPIs ────────────────────────────────────────
@app.route("/api/kpis")
def api_kpis():
    return jsonify(analytics["kpis"])

# ── API: Cross-sector comparison ───────────────────────────────
@app.route("/api/comparison")
def api_comparison():
    result = {}
    for sector in ["Private", "Government"]:
        sub = df[df["Sector"] == sector]
        result[sector] = {
            "avg_base":      round(sub["Base_Salary"].mean(), 2),
            "avg_total":     round(sub["Total_Pay"].mean(), 2),
            "avg_overtime":  round(sub["Overtime_Pay"].mean(), 2),
            "avg_longevity": round(sub["Longevity_Pay"].mean(), 2),
            "median_base":   round(sub["Base_Salary"].median(), 2),
            "max_base":      round(sub["Base_Salary"].max(), 2),
            "min_base":      round(sub["Base_Salary"].min(), 2),
            "std_base":      round(sub["Base_Salary"].std(), 2),
            "count":         int(len(sub)),
            "q25": round(sub["Base_Salary"].quantile(0.25), 2),
            "q75": round(sub["Base_Salary"].quantile(0.75), 2),
        }

    # Monthly breakdown (simulate from salary distributions)
    monthly = []
    for sector in ["Private", "Government"]:
        sub = df[df["Sector"]==sector]
        monthly.append({
            "sector": sector,
            "avg_monthly": round(sub["Base_Salary"].mean() / 12, 2)
        })

    result["monthly"] = monthly
    return jsonify(result)

# ── API: Gender pay gap ────────────────────────────────────────
@app.route("/api/gender")
def api_gender():
    result = {}
    for sector in ["Private", "Government", "All"]:
        sub = df if sector == "All" else df[df["Sector"]==sector]
        m = sub[sub["Gender"]=="M"]["Base_Salary"]
        f = sub[sub["Gender"]=="F"]["Base_Salary"]
        result[sector] = {
            "M_avg":   round(m.mean(), 2),
            "F_avg":   round(f.mean(), 2),
            "M_med":   round(m.median(), 2),
            "F_med":   round(f.median(), 2),
            "M_count": int(len(m)),
            "F_count": int(len(f)),
            "gap":     round(m.mean() - f.mean(), 2),
            "gap_pct": round(((m.mean() - f.mean()) / f.mean()) * 100, 2) if f.mean() > 0 else 0,
        }

    # Gender gap by department (top 10 biggest gaps)
    dept_gap = df.groupby(["Sector","Department","Gender"])["Base_Salary"].mean().unstack("Gender").reset_index()
    dept_gap.columns = ["Sector","Department","F_avg","M_avg"]
    dept_gap["gap"] = (dept_gap["M_avg"] - dept_gap["F_avg"]).round(2)
    dept_gap = dept_gap.dropna().sort_values("gap", ascending=False).head(15)

    result["dept_gap"] = dept_gap.to_dict("records")
    return jsonify(result)

# ── API: Department explorer ───────────────────────────────────
@app.route("/api/departments")
def api_departments():
    sector = request.args.get("sector", "All")
    sub = df if sector == "All" else df[df["Sector"]==sector]

    dept = sub.groupby(["Sector","Department","Department_Name"]).agg(
        avg_base=("Base_Salary","mean"),
        avg_total=("Total_Pay","mean"),
        avg_overtime=("Overtime_Pay","mean"),
        avg_longevity=("Longevity_Pay","mean"),
        count=("Base_Salary","count"),
        max_salary=("Base_Salary","max"),
        min_salary=("Base_Salary","min"),
    ).reset_index().round(2).sort_values("avg_base", ascending=False)

    return jsonify(dept.to_dict("records"))

@app.route("/api/department/detail")
def api_dept_detail():
    dept_name = request.args.get("dept", "")
    sub = df[df["Department_Name"].str.contains(dept_name, case=False, na=False)]

    divisions = sub.groupby("Division").agg(
        avg_base=("Base_Salary","mean"),
        count=("Base_Salary","count"),
    ).reset_index().sort_values("avg_base", ascending=False).round(2)

    gender = sub.groupby("Gender").agg(
        avg_base=("Base_Salary","mean"),
        count=("Base_Salary","count"),
    ).reset_index().round(2)

    return jsonify({
        "department": dept_name,
        "total_count": int(len(sub)),
        "avg_base": round(sub["Base_Salary"].mean(), 2),
        "divisions": divisions.to_dict("records"),
        "gender": gender.to_dict("records"),
    })

# ── API: Grade analysis ────────────────────────────────────────
@app.route("/api/grades")
def api_grades():
    sector = request.args.get("sector", "All")
    sub = df if sector == "All" else df[df["Sector"]==sector]

    grade = sub.groupby(["Sector","Grade","Grade_Numeric"]).agg(
        avg_base=("Base_Salary","mean"),
        avg_total=("Total_Pay","mean"),
        count=("Base_Salary","count"),
        min_salary=("Base_Salary","min"),
        max_salary=("Base_Salary","max"),
    ).reset_index().sort_values("Grade_Numeric").round(2)

    return jsonify(grade.to_dict("records"))

# ── API: Salary distribution ───────────────────────────────────
@app.route("/api/distribution")
def api_distribution():
    bins = list(range(0, 310000, 15000))
    labels = [f"${b//1000}K" for b in bins[:-1]]
    result = {}
    for sector in ["Private", "Government"]:
        sub = df[df["Sector"]==sector]["Base_Salary"]
        counts, _ = np.histogram(sub, bins=bins)
        result[sector] = {
            "labels": labels,
            "counts": counts.tolist(),
            "percentiles": {
                "p10": round(sub.quantile(0.10), 2),
                "p25": round(sub.quantile(0.25), 2),
                "p50": round(sub.quantile(0.50), 2),
                "p75": round(sub.quantile(0.75), 2),
                "p90": round(sub.quantile(0.90), 2),
            }
        }
    return jsonify(result)

# ── API: Overtime analysis ─────────────────────────────────────
@app.route("/api/overtime")
def api_overtime():
    sector = request.args.get("sector", "All")
    sub = df if sector == "All" else df[df["Sector"]==sector]

    summary = {}
    for s in ["Private","Government"]:
        d = df[df["Sector"]==s]
        summary[s] = {
            "pct_with_ot":  round((d["Overtime_Flag"].mean())*100, 2),
            "avg_ot":       round(d["Overtime_Pay"].mean(), 2),
            "avg_ot_earners": round(d[d["Overtime_Pay"]>0]["Overtime_Pay"].mean(), 2),
            "max_ot":       round(d["Overtime_Pay"].max(), 2),
        }

    dept_ot = sub.groupby(["Sector","Department"]).agg(
        avg_ot=("Overtime_Pay","mean"),
        pct_ot=("Overtime_Flag","mean"),
        count=("Overtime_Pay","count"),
    ).reset_index().round(2).sort_values("avg_ot", ascending=False)

    gender_ot = sub.groupby("Gender").agg(
        avg_ot=("Overtime_Pay","mean"),
        pct_ot=("Overtime_Flag","mean"),
    ).reset_index().round(2)

    return jsonify({
        "summary": summary,
        "dept_ot": dept_ot.to_dict("records"),
        "gender_ot": gender_ot.to_dict("records"),
    })

# ── API: Longevity tracker ─────────────────────────────────────
@app.route("/api/longevity")
def api_longevity():
    sector = request.args.get("sector", "All")
    sub = df if sector == "All" else df[df["Sector"]==sector]

    summary = {}
    for s in ["Private","Government"]:
        d = df[df["Sector"]==s]
        summary[s] = {
            "pct_with_lon":    round(d["Longevity_Flag"].mean()*100, 2),
            "avg_lon":         round(d["Longevity_Pay"].mean(), 2),
            "avg_lon_earners": round(d[d["Longevity_Pay"]>0]["Longevity_Pay"].mean(), 2),
            "max_lon":         round(d["Longevity_Pay"].max(), 2),
        }

    dept_lon = sub.groupby(["Sector","Department"]).agg(
        avg_lon=("Longevity_Pay","mean"),
        pct_lon=("Longevity_Flag","mean"),
    ).reset_index().round(2).sort_values("avg_lon", ascending=False)

    return jsonify({
        "summary": summary,
        "dept_lon": dept_lon.to_dict("records"),
    })

# ── API: Salary prediction ─────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()

    try:
        sector     = data.get("sector", "Private")
        department = data.get("department", "IT")
        gender     = data.get("gender", "M")
        grade      = data.get("grade", "10")
        has_ot     = int(data.get("has_overtime", 0))
        has_lon    = int(data.get("has_longevity", 0))

        # Encode inputs
        dept_enc = encoders["Department"].transform([department])[0] \
            if department in encoders["Department"].classes_ else 0
        grade_enc = encoders["Grade"].transform([grade])[0] \
            if grade in encoders["Grade"].classes_ else 0
        sector_enc = 1 if sector == "Private" else 0
        gender_enc = 1 if gender == "M" else 0

        # Grade numeric
        grade_num = int(''.join(filter(str.isdigit, str(grade))) or 0)

        # Dept avg salary from analytics
        dept_rows = [r for r in analytics["dept_salary"]
                     if r["Department"] == department and r["Sector"] == sector]
        dept_avg = dept_rows[0]["avg_base"] if dept_rows else analytics["kpis"]["overall_avg"]

        # Dept_Name_enc (use first matching)
        dept_name_rows = [r for r in analytics["dept_salary"]
                          if r["Department"] == department and r["Sector"] == sector]
        dept_name = dept_name_rows[0]["Department_Name"] if dept_name_rows else department
        dept_name_enc = encoders["Department_Name"].transform([dept_name])[0] \
            if dept_name in encoders["Department_Name"].classes_ else 0

        features = np.array([[
            dept_enc, dept_name_enc, gender_enc,
            grade_num, sector_enc, has_ot, has_lon,
            dept_avg, grade_enc
        ]])

        # RF prediction
        rf_model = trained_models["random_forest"]["model"]
        rf_pred  = float(rf_model.predict(features)[0])

        # Ridge prediction
        ridge_model  = trained_models["ridge"]["model"]
        ridge_scaler = trained_models["ridge"]["scaler"]
        ridge_pred   = float(ridge_model.predict(ridge_scaler.transform(features))[0])

        # Ensemble average
        ensemble = (rf_pred * 0.65 + ridge_pred * 0.35)

        metrics = analytics["model_metrics"]

        return jsonify({
            "rf_prediction":     round(rf_pred, 2),
            "ridge_prediction":  round(ridge_pred, 2),
            "ensemble":          round(ensemble, 2),
            "dept_avg":          round(dept_avg, 2),
            "rf_r2":             round(metrics["random_forest"]["r2"], 4),
            "ridge_r2":          round(metrics["ridge"]["r2"], 4),
            "rf_mae":            round(metrics["random_forest"]["mae"], 2),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

# ── API: Calculator ────────────────────────────────────────────
@app.route("/api/calculator", methods=["POST"])
def api_calculator():
    data = request.get_json()
    base      = safe_float(data.get("base_salary", 0))
    overtime  = safe_float(data.get("overtime_pay", 0))
    longevity = safe_float(data.get("longevity_pay", 0))
    sector    = data.get("sector", "Private")
    department= data.get("department", "IT")

    total = base + overtime + longevity
    ot_ratio = round((overtime / base * 100), 2) if base > 0 else 0

    # Compare against sector avg
    kpis = analytics["kpis"]
    sector_avg = kpis["private_avg_base"] if sector == "Private" else kpis["govt_avg_base"]
    diff_from_avg = round(base - sector_avg, 2)
    pct_from_avg  = round((diff_from_avg / sector_avg) * 100, 2) if sector_avg > 0 else 0

    # Band
    q25, q50, q75 = kpis["q25"], kpis["q50"], kpis["q75"]
    if base < q25:   band = "Low (bottom 25%)"
    elif base < q50: band = "Mid-Low (25–50%)"
    elif base < q75: band = "Mid-High (50–75%)"
    else:            band = "High (top 25%)"

    # Dept avg
    dept_rows = [r for r in analytics["dept_salary"]
                 if r["Department"] == department and r["Sector"] == sector]
    dept_avg = dept_rows[0]["avg_base"] if dept_rows else sector_avg

    return jsonify({
        "total_compensation": round(total, 2),
        "base_salary":        base,
        "overtime_pay":       overtime,
        "longevity_pay":      longevity,
        "overtime_ratio_pct": ot_ratio,
        "sector_avg":         sector_avg,
        "dept_avg":           round(dept_avg, 2),
        "diff_from_sector_avg": diff_from_avg,
        "pct_from_avg":       pct_from_avg,
        "salary_band":        band,
        "annual_monthly":     round(total / 12, 2),
        "annual_weekly":      round(total / 52, 2),
    })

# ── API: Filter options ────────────────────────────────────────
@app.route("/api/options")
def api_options():
    return jsonify({
        "private_depts":  sorted(df[df["Sector"]=="Private"]["Department"].unique().tolist()),
        "govt_depts":     sorted(df[df["Sector"]=="Government"]["Department"].unique().tolist()),
        "private_grades": sorted(df[df["Sector"]=="Private"]["Grade"].unique().tolist()),
        "govt_grades":    sorted(df[df["Sector"]=="Government"]["Grade"].dropna().unique().tolist()),
        "private_dept_names": sorted(df[df["Sector"]=="Private"]["Department_Name"].unique().tolist()),
        "govt_dept_names":    sorted(df[df["Sector"]=="Government"]["Department_Name"].unique().tolist()),
    })

# ── API: Table data ────────────────────────────────────────────
@app.route("/api/table")
def api_table():
    sector = request.args.get("sector", "All")
    dept   = request.args.get("dept", "")
    gender = request.args.get("gender", "")
    page   = int(request.args.get("page", 1))
    per_page = 50

    sub = df.copy()
    if sector != "All":
        sub = sub[sub["Sector"] == sector]
    if dept:
        sub = sub[sub["Department"] == dept]
    if gender:
        sub = sub[sub["Gender"] == gender]

    total = len(sub)
    sub   = sub.sort_values("Base_Salary", ascending=False)
    sub   = sub.iloc[(page-1)*per_page : page*per_page]

    cols = ["Sector","Department","Department_Name","Gender",
            "Base_Salary","Overtime_Pay","Longevity_Pay","Total_Pay","Grade","Salary_Band"]
    return jsonify({
        "data":  sub[cols].round(2).to_dict("records"),
        "total": total,
        "page":  page,
        "pages": (total // per_page) + 1
    })

if __name__ == "__main__":
    app.run(debug=True, port=5002)