"""
app.py
──────
Flask application — Placement Skill-Gap & Readiness Timeline Predictor

Routes
------
GET  /              → index.html   (input form)
GET  /get_roles     → JSON         (AJAX: roles for a company)
POST /results       → results.html (ML prediction + charts)
GET  /history       → history.html (past predictions)
POST /clear_history → redirect /history
"""

import sqlite3
import json
import logging
from datetime import datetime

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    g,
    flash,
)

import model as ml


# ─────────────────────────────────────────────────────────────────────────────
# Logging Setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────────────────────────────────────

app = Flask(__name__)
app.secret_key = "super_secret_skillgap_key_for_flash_messages"  # In production, use os.environ
DATABASE = "database.db"


# ─────────────────────────────────────────────────────────────────────────────
# Database helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_db() -> sqlite3.Connection:
    """Return a per-request SQLite connection stored on Flask's `g`."""
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    """Create the predictions table if it doesn't already exist."""
    with app.app_context():
        try:
            db = get_db()
            db.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp       TEXT    NOT NULL,
                    student_name    TEXT,
                    company         TEXT    NOT NULL,
                    role            TEXT    NOT NULL,
                    student_skills  TEXT    NOT NULL,
                    required_skills TEXT    NOT NULL,
                    match_pct       REAL    NOT NULL,
                    missing_skills  TEXT    NOT NULL,
                    total_hours     REAL    NOT NULL,
                    weeks_needed    INTEGER NOT NULL,
                    study_hours     INTEGER NOT NULL
                )
            """)
            db.commit()
            logger.info("Database initialized successfully.")
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    """Landing page — input form."""
    companies = ml.get_companies()
    return render_template("index.html", companies=companies)


@app.route("/get_roles")
def get_roles():
    """
    AJAX endpoint.
    Query param: ?company=<name>
    Returns: JSON list of role strings.
    """
    company = request.args.get("company", "").strip()
    roles   = ml.get_roles(company) if company else []
    return jsonify(roles)


@app.route("/results", methods=["POST"])
def results():
    """
    Receive form data, run ML prediction, save to DB, render results.
    """
    # ── Collect & Validate form inputs ────────────────────────────────────
    student_name = request.form.get("student_name", "Student")
    company      = request.form.get("company", "").strip()
    role         = request.form.get("role", "").strip()
    student_skills_raw = request.form.get("student_skills", "").strip()
    study_hours_raw = request.form.get("study_hours", "10")
    
    # 1. Validate required strings
    if not company or not role:
        flash("Company and Role are required fields.", "danger")
        logger.warning("Prediction attempt with missing company or role.")
        return redirect(url_for("index"))
        
    if not student_skills_raw:
        flash("Please enter at least one skill to predict readiness.", "warning")
        logger.warning(f"Prediction attempt with empty skills for {company} - {role}.")
        return redirect(url_for("index"))

    # 2. Validate Study Hours
    try:
        study_hours = int(study_hours_raw)
        if study_hours <= 0:
            flash("Study hours per week must be greater than 0.", "danger")
            return redirect(url_for("index"))
        if study_hours > 168:
            flash("Study hours cannot exceed 168 hours in a week.", "danger")
            return redirect(url_for("index"))
    except ValueError:
        flash("Invalid input for study hours. Please enter a valid number.", "danger")
        logger.error("ValueError parsing study hours.")
        return redirect(url_for("index"))

    logger.info(f"Processing prediction request: Student='{student_name}', Company='{company}', Role='{role}', Hours={study_hours}")

    # ── Get required skills for the role ─────────────────────────────────
    try:
        required_skills_str = ml.get_required_skills(company, role)
        if not required_skills_str:
            flash(f"No skills found for {role} at {company}. Please try another role.", "warning")
            return redirect(url_for("index"))
    except Exception as e:
        logger.error(f"Error fetching required skills: {e}")
        flash("An error occurred while fetching role requirements.", "danger")
        return redirect(url_for("index"))

    # ── ML computations ───────────────────────────────────────────────────
    try:
        match_pct = ml.compute_match(student_skills_raw, required_skills_str)
        timeline  = ml.predict_timeline(student_skills_raw, required_skills_str, study_hours)
        all_companies = ml.compute_all_company_matches(student_skills_raw)
        charts        = ml.build_charts(match_pct, timeline, all_companies)
    except Exception as e:
        logger.error(f"ML Pipeline Error: {e}")
        flash("An internal error occurred during the ML prediction phase. Please try again.", "danger")
        return redirect(url_for("index"))

    # ── Readiness level label ─────────────────────────────────────────────
    # match_pct is Skill Coverage % (set intersection / required count * 100).
    # Thresholds are calibrated to coverage, not TF-IDF similarity.
    if match_pct >= 70:
        readiness_label, readiness_class = "Placement Ready 🎯", "success"
    elif match_pct >= 40:
        readiness_label, readiness_class = "Almost There 📈",   "warning"
    else:
        readiness_label, readiness_class = "Needs Work 🛠️",     "danger"

    # ── Persist to SQLite ─────────────────────────────────────────────────
    try:
        db = get_db()
        db.execute(
            """
            INSERT INTO predictions
                (timestamp, student_name, company, role,
                 student_skills, required_skills, match_pct,
                 missing_skills, total_hours, weeks_needed, study_hours)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                student_name or "Anonymous",
                company,
                role,
                student_skills_raw,
                required_skills_str,
                round(match_pct, 2),
                json.dumps(timeline["missing_skills"]),
                timeline["total_learning_hours"],
                timeline["weeks_needed"],
                study_hours,
            ),
        )
        db.commit()
        logger.info("Prediction successfully saved to database.")
    except Exception as e:
        logger.error(f"Database insertion failed: {e}")
        # Do not block rendering, just warn
        flash("Results generated, but could not save to history.", "warning")

    return render_template(
        "results.html",
        student_name    = student_name or "Student",
        company         = company,
        role            = role,
        match_pct       = round(match_pct, 1),
        readiness_label = readiness_label,
        readiness_class = readiness_class,
        timeline        = timeline,
        study_hours     = study_hours,
        charts          = charts,
        all_companies   = all_companies,
        required_skills = [s.strip() for s in required_skills_str.split(",") if s.strip()],
        student_skills  = [s.strip() for s in student_skills_raw.split(",") if s.strip()],
    )


@app.route("/history")
def history():
    """Show all past predictions and generate a trend chart."""
    try:
        db   = get_db()
        rows = db.execute(
            "SELECT * FROM predictions ORDER BY id DESC"
        ).fetchall()

        predictions = []
        timestamps = []
        match_scores = []
        
        # We need chronological order for the chart (oldest first)
        for row in reversed(rows):
            d = dict(row)
            d["missing_skills"] = json.loads(d["missing_skills"])
            # Format timestamp for chart label
            ts = datetime.strptime(d["timestamp"], "%Y-%m-%d %H:%M:%S").strftime("%b %d, %H:%M")
            timestamps.append(ts)
            match_scores.append(d["match_pct"])
            
            # Rebuild list for table (newest first)
            predictions.insert(0, d)
            
        trend_fig = None
        if timestamps:
            trend_fig = {
                "data": [{
                    "x": timestamps,
                    "y": match_scores,
                    "type": "scatter",
                    "mode": "lines+markers",
                    "marker": {"color": "#6366f1", "size": 8},
                    "line": {"color": "#6366f1", "width": 3, "shape": "spline"}
                }],
                "layout": {
                    "title": {"text": "Readiness Trend Over Time", "font": {"color": "#e2e8f0"}},
                    "paper_bgcolor": "rgba(0,0,0,0)",
                    "plot_bgcolor": "rgba(0,0,0,0)",
                    "font": {"color": "#e2e8f0"},
                    "yaxis": {"title": "Match %", "range": [0, 100], "gridcolor": "rgba(255,255,255,0.08)"},
                    "xaxis": {"gridcolor": "rgba(255,255,255,0.08)"},
                    "margin": {"t": 40, "b": 40, "l": 40, "r": 20},
                    "height": 300
                }
            }

        return render_template("history.html", predictions=predictions, trend_json=json.dumps(trend_fig) if trend_fig else "{}")
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        flash("An error occurred while loading your history.", "danger")
        return redirect(url_for("index"))


@app.route("/clear_history", methods=["POST"])
def clear_history():
    """Delete all prediction records."""
    try:
        db = get_db()
        db.execute("DELETE FROM predictions")
        db.commit()
        logger.info("Prediction history cleared.")
        flash("Prediction history successfully cleared.", "success")
    except Exception as e:
        logger.error(f"Error clearing history: {e}")
        flash("Could not clear history due to a database error.", "danger")
        
    return redirect(url_for("history"))


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()
    app.run(debug=True)
