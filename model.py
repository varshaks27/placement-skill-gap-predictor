"""
model.py
────────
ML engine for the Placement Skill-Gap & Readiness Timeline Predictor.

Responsibilities:
  - Load and cache job postings + skill difficulty data
  - Expose helper functions for dropdown population
  - Compute skill-match % via TF-IDF cosine similarity
  - Predict readiness timeline using configurable learning hours from CSV
  - Build Plotly chart JSON for the results page
"""

import os
import json
import math
import re

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DATA_DIR         = os.path.join(BASE_DIR, "data")
JOBS_CSV         = os.path.join(DATA_DIR, "job_postings.csv")
DIFFICULTY_CSV   = os.path.join(DATA_DIR, "skill_difficulty.csv")

# ── Target Placement Companies (canonical display names, fixed order) ────────────
PLACEMENT_COMPANIES: list[str] = [
    "TCS",
    "Infosys",
    "Wipro",
    "Accenture",
    "Capgemini",
    "Cognizant",
    "IBM",
]

# ── Default fallback (hours) when a skill is absent from skill_difficulty.csv ─
DEFAULT_LEARNING_HOURS = 10

# ── Radar Chart Categories ────────────────────────────────────────────────────
SKILL_CATEGORIES = {
    "Programming": ["python", "java", "c++", "c#", "c", "javascript", "typescript", "ruby", "go", "php"],
    "Frontend": ["html", "css", "react", "angular", "vue", "bootstrap", "tailwind", "jquery", "sass"],
    "Backend": ["node.js", "django", "flask", "spring boot", "express", "laravel", ".net", "rest api", "graphql"],
    "Databases": ["sql", "mysql", "postgresql", "mongodb", "oracle", "redis", "nosql", "dynamodb"],
    "Cloud": ["aws", "azure", "gcp", "google cloud", "heroku", "docker", "kubernetes"],
    "DevOps": ["git", "github", "gitlab", "ci/cd", "jenkins", "linux", "bash", "agile", "jira"],
    "Soft Skills": ["communication", "leadership", "teamwork", "problem solving", "analytical", "presentation"]
}

# ── Skill Resources Lookup (Documentation, Courses, Projects) ──────────────────
# Fulfills Phase 4 requirement to map missing skills to resources.
SKILL_RESOURCES = {
    "python": {
        "docs": "https://docs.python.org/3/",
        "course": "Python for Everybody (Coursera)",
        "project": "Build an Automated File Organizer script"
    },
    "sql": {
        "docs": "https://www.w3schools.com/sql/",
        "course": "SQL for Data Science (UC Davis)",
        "project": "Create a Library Management System database schema"
    },
    "react": {
        "docs": "https://react.dev/",
        "course": "React - The Complete Guide (Udemy)",
        "project": "Build a personal Task Management Kanban board"
    },
    "node.js": {
        "docs": "https://nodejs.org/en/docs",
        "course": "Learn Node.js (Codecademy)",
        "project": "Build a real-time Chat application with WebSockets"
    },
    "docker": {
        "docs": "https://docs.docker.com/",
        "course": "Docker Mastery (Udemy)",
        "project": "Containerize a Flask application with Nginx"
    },
    "aws": {
        "docs": "https://docs.aws.amazon.com/",
        "course": "AWS Certified Cloud Practitioner (Stephane Maarek)",
        "project": "Deploy a static web application on S3 & CloudFront"
    },
    "git": {
        "docs": "https://git-scm.com/doc",
        "course": "Git & GitHub Crash Course (FreeCodeCamp)",
        "project": "Setup a collaborative GitHub repo with branch protections"
    },
    "c++": {
        "docs": "https://en.cppreference.com/",
        "course": "C++ Tutorial for Beginners (Programming with Mosh)",
        "project": "Build a text-based RPG Command Line game"
    },
    "machine learning": {
        "docs": "https://scikit-learn.org/stable/documentation.html",
        "course": "Machine Learning Specialization (Andrew Ng)",
        "project": "Train a House Price Prediction model on Kaggle"
    }
}

# ── Module-level cache ────────────────────────────────────────────────────────
_jobs_df: pd.DataFrame | None = None
_diff_df: pd.DataFrame | None = None


# ─────────────────────────────────────────────────────────────────────────────
# Data loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_jobs() -> pd.DataFrame:
    """Load (and cache) the cleaned job postings CSV."""
    global _jobs_df
    if _jobs_df is None:
        _jobs_df = pd.read_csv(JOBS_CSV)
        _jobs_df["Company"] = _jobs_df["Company"].str.strip()
        _jobs_df["Role"]    = _jobs_df["Role"].str.strip()
        _jobs_df["Skills"]  = _jobs_df["Skills"].str.lower().str.strip()
    return _jobs_df


def load_difficulty() -> pd.DataFrame:
    """
    Load (and cache) the skill difficulty CSV.
    Expected columns: skill, difficulty_weight, estimated_learning_hours
    """
    global _diff_df
    if _diff_df is None:
        _diff_df = pd.read_csv(DIFFICULTY_CSV)
        _diff_df["skill"] = _diff_df["skill"].str.strip().str.lower()
        _diff_df.set_index("skill", inplace=True)
    return _diff_df


# ─────────────────────────────────────────────────────────────────────────────
# Dropdown helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_companies() -> list[str]:
    """
    Return the fixed list of target placement companies.
    Order is intentional (most-recruited first).
    Only companies present in the cleaned CSV are returned.
    """
    df = load_jobs()
    available = set(df["Company"].dropna().unique())
    return [c for c in PLACEMENT_COMPANIES if c in available]


def get_roles(company: str) -> list[str]:
    """Return a sorted list of roles available at the given company.
    Only allowed if the company is in PLACEMENT_COMPANIES.
    """
    if company not in PLACEMENT_COMPANIES:
        return []
    df = load_jobs()
    roles = df.loc[df["Company"] == company, "Role"].dropna().unique().tolist()
    return sorted(roles)


def get_required_skills(company: str, role: str) -> str:
    """
    Return the comma-separated skills string for a given company + role.
    If multiple rows match, skills are merged and deduplicated.
    """
    df = load_jobs()
    mask = (df["Company"] == company) & (df["Role"] == role)
    rows = df.loc[mask, "Skills"]
    if rows.empty:
        return ""
    # Merge all matching skill sets (different postings for same role)
    all_skills: set[str] = set()
    for cell in rows:
        for s in str(cell).split(","):
            cleaned = s.strip().lower()
            if cleaned:
                all_skills.add(cleaned)
    return ", ".join(sorted(all_skills))


# ─────────────────────────────────────────────────────────────────────────────
# ML — Helpers
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_skills(skills_str: str) -> str:
    """
    Converts a comma-separated string of skills into a normalized space-separated string,
    stripping purely special characters to prevent vectorizer errors.
    """
    if not isinstance(skills_str, str) or not skills_str.strip():
        return ""
    
    skills = []
    for s in skills_str.split(","):
        cleaned = s.strip().lower()
        # Keep letters, numbers, spaces, +, #, . (e.g. C++, C#, .NET, Node.js)
        cleaned = re.sub(r'[^a-z0-9\s\+#\.]', '', cleaned).strip()
        if cleaned:
            skills.append(cleaned)
            
    return " ".join(skills)


# ─────────────────────────────────────────────────────────────────────────────
# ML — Canonical Skill-Match Score  (ONE definition used everywhere)
# ─────────────────────────────────────────────────────────────────────────────

def compute_match(student_skills_str: str, required_skills_str: str) -> float:
    """
    CANONICAL match score = Skill Coverage %.

    Definition
    ----------
    skill_coverage = |student_skills ∩ required_skills| / |required_skills| * 100

    This is the ONE number shown on the Gauge, stat card, Placement Report,
    Company Comparison, History, and PDF Export.

    Using set-intersection (not TF-IDF) means the Gauge and the Pie Chart
    are mathematically consistent: both derive from the same sets.
    TF-IDF is kept as a private helper for company-wide ranking only.

    Parameters
    ----------
    student_skills_str  : comma-separated skills entered by the student
    required_skills_str : comma-separated skills required for the job

    Returns
    -------
    float : coverage percentage 0.0 – 100.0, rounded to 1 decimal place
    """
    student_skills  = {s.strip().lower() for s in student_skills_str.split(",") if s.strip()}
    required_skills = {s.strip().lower() for s in required_skills_str.split(",") if s.strip()}

    if not required_skills:
        return 0.0

    matched = student_skills & required_skills
    coverage = len(matched) / len(required_skills) * 100
    return round(coverage, 1)


def _tfidf_similarity(student_skills_str: str, required_skills_str: str) -> float:
    """
    INTERNAL ONLY — TF-IDF cosine similarity between two skill strings.
    Used exclusively inside compute_all_company_matches to rank companies.
    Never shown directly to users.

    Returns float 0.0 – 100.0.
    """
    s_str = preprocess_skills(student_skills_str)
    r_str = preprocess_skills(required_skills_str)
    if not s_str or not r_str:
        return 0.0
    vectorizer = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizer.fit_transform([r_str, s_str])
        sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return round(float(sim) * 100, 2)
    except ValueError:
        return 0.0



# ─────────────────────────────────────────────────────────────────────────────
# ML — Readiness Timeline
# ─────────────────────────────────────────────────────────────────────────────

def predict_timeline(
    student_skills_str: str,
    required_skills_str: str,
    study_hours_per_week: float,
) -> dict:
    """
    Predict placement readiness timeline.

    Formula
    -------
    Total Estimated Learning Hours = Σ estimated_learning_hours (for each missing skill)
    Predicted Weeks = ceil(Total Estimated Learning Hours ÷ study_hours_per_week)

    Parameters
    ----------
    student_skills_str    : comma-separated skills the student already has
    required_skills_str   : comma-separated skills required for the target role
    study_hours_per_week  : hours the student can dedicate per week

    Returns
    -------
    dict with keys:
        missing_skills          : list[str]
        total_learning_hours    : float
        weeks_needed            : float
        breakdown               : list[dict]  (per-skill detail)
        already_known           : list[str]
        radar_data              : dict (scores per category)
        confidence_score        : float
    """
    diff_df = load_difficulty()
    # Ensure required columns exist; provide defaults if missing
    if 'estimated_learning_hours' not in diff_df.columns:
        diff_df['estimated_learning_hours'] = DEFAULT_LEARNING_HOURS
    if 'difficulty_weight' not in diff_df.columns:
        diff_df['difficulty_weight'] = 2  # default medium difficulty

    student_skills  = {s.strip().lower() for s in student_skills_str.split(",") if s.strip()}
    required_skills = {s.strip().lower() for s in required_skills_str.split(",") if s.strip()}

    missing_skills = sorted(required_skills - student_skills)
    already_known  = sorted(required_skills & student_skills)

    breakdown: list[dict] = []
    total_learning_hours  = 0.0

    for skill in missing_skills:
        if skill in diff_df.index:
            hours  = float(diff_df.loc[skill, "estimated_learning_hours"])
            weight = int(diff_df.loc[skill, "difficulty_weight"])
        else:
            hours  = DEFAULT_LEARNING_HOURS
            weight = 2  # treat unknowns as medium

        total_learning_hours += hours
        breakdown.append({
            "skill":            skill,
            "difficulty_weight": weight,
            "learning_hours":   hours,
        })

    study_hours_per_week = max(study_hours_per_week, 1)  # guard divide-by-zero
    weeks_needed = math.ceil(total_learning_hours / study_hours_per_week)

    # ── Confidence Score ──────────────────────────────────────────────────
    # Confidence scales with the volume of required skills (more data = more confidence).
    base_confidence = 60.0
    data_bonus = min(len(required_skills) * 3, 30.0) # Up to 30% from data volume
    confidence = min(round(base_confidence + data_bonus, 1), 99.0)
    if len(required_skills) == 0:
        confidence = 0.0

    # ── Radar Chart Metrics ───────────────────────────────────────────────
    radar_data = {cat: 0 for cat in SKILL_CATEGORIES.keys()}
    
    # A simple scoring: if they know a skill in a category that is REQUIRED, they get points.
    for skill in required_skills:
        for cat, items in SKILL_CATEGORIES.items():
            if any(item in skill for item in items):
                if skill in already_known:
                    radar_data[cat] += 2 # Mastered
                else:
                    radar_data[cat] += 0 # Missing

    # Scale radar scores out of 5 for visually pleasing charts
    max_cat_score = max(radar_data.values()) if radar_data.values() else 1
    if max_cat_score == 0: max_cat_score = 1
    
    for cat in radar_data:
        radar_data[cat] = round((radar_data[cat] / max_cat_score) * 5, 1)

    # ── Roadmap Generation ────────────────────────────────────────────────
    # Sequentially schedules missing skills week-by-week based on weekly hours
    roadmap = []
    current_week = 1
    accumulated_hours = 0
    current_week_skills = []
    
    # Sort breakdown items (prioritize skills with higher weights/hours)
    sorted_breakdown = sorted(breakdown, key=lambda x: x.get("learning_hours", x.get("estimated_learning_hours", 0)), reverse=True)
    
    for item in sorted_breakdown:
        skill_name = item["skill"]
        skill_hours = item.get("learning_hours", item.get("estimated_learning_hours", 0))
        
        while skill_hours > 0:
            remaining_capacity = study_hours_per_week - accumulated_hours
            if skill_hours <= remaining_capacity:
                current_week_skills.append({
                    "skill": skill_name,
                    "hours": round(skill_hours, 1)
                })
                accumulated_hours += skill_hours
                skill_hours = 0
            else:
                current_week_skills.append({
                    "skill": skill_name,
                    "hours": round(remaining_capacity, 1)
                })
                skill_hours -= remaining_capacity
                # Week capacity filled! Push and transition
                roadmap.append({
                    "week": current_week,
                    "skills": current_week_skills
                })
                current_week += 1
                current_week_skills = []
                accumulated_hours = 0
                
    if current_week_skills:
        roadmap.append({
            "week": current_week,
            "skills": current_week_skills
        })

    # Attach recommendations / resources to breakdown items
    for item in breakdown:
        skill_clean = item["skill"].lower()
        # Find match in SKILL_RESOURCES or set generic fallbacks
        matched_resource = SKILL_RESOURCES.get(skill_clean, {
            "docs": f"https://www.google.com/search?q={skill_clean}+official+documentation",
            "course": f"Introductory {item['skill']} Course (FreeCodeCamp / YouTube)",
            "project": f"Build a prototype project incorporating {item['skill']}"
        })
        item["resources"] = matched_resource

    return {
        "missing_skills":       missing_skills,
        "already_known":        already_known,
        "total_learning_hours": round(total_learning_hours, 1),
        "weeks_needed":         weeks_needed,
        "breakdown":            breakdown,
        "radar_data":           radar_data,
        "confidence_score":     confidence,
        "roadmap":              roadmap
    }


# ─────────────────────────────────────────────────────────────────────────────
# Chart builders (Plotly JSON)
# ─────────────────────────────────────────────────────────────────────────────

def build_charts(match_pct: float, timeline: dict, all_companies: list[dict] = None) -> dict:
    """
    Build Plotly figure JSON for the results page.

    Returns
    -------
    dict with keys:
        gauge_json      : skill-match gauge chart
        bar_json        : missing-skills bar chart (hours per skill)
        donut_json      : known vs missing pie chart
        compare_json    : company comparison bar chart
        radar_json      : radar chart mapping categories
    """

    # ── Colour palette (dark theme) ──────────────────────────────────────
    COLOR_EASY   = "#4ade80"   # green
    COLOR_MEDIUM = "#facc15"   # yellow
    COLOR_HARD   = "#f87171"   # red
    BG           = "rgba(0,0,0,0)"
    FONT_COLOR   = "#e2e8f0"

    def difficulty_color(w: int) -> str:
        return {1: COLOR_EASY, 2: COLOR_MEDIUM, 3: COLOR_HARD}.get(w, COLOR_MEDIUM)

    # ── 1. Gauge — Skill Coverage % ──────────────────────────────────────
    # Shows the canonical Skill Coverage score: |known ∩ required| / |required| * 100
    # This matches the Pie Chart, stat card, and Company Comparison exactly.
    gauge_fig = {
        "data": [{
            "type": "indicator",
            "mode": "gauge+number+delta",
            "value": match_pct,
            "delta": {"reference": 70, "increasing": {"color": COLOR_EASY}},
            "number": {"suffix": "%", "font": {"color": FONT_COLOR, "size": 48}},
            "gauge": {
                "axis": {"range": [0, 100], "tickcolor": FONT_COLOR},
                "bar":  {"color": "#6366f1"},
                "steps": [
                    {"range": [0,  40], "color": "rgba(248,113,113,0.25)"},
                    {"range": [40, 70], "color": "rgba(250,204,21,0.25)"},
                    {"range": [70,100], "color": "rgba(74,222,128,0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#a78bfa", "width": 4},
                    "thickness": 0.75,
                    "value": 70,
                },
            },
            "title": {"text": "Skill Coverage", "font": {"color": FONT_COLOR, "size": 16}},
        }],
        "layout": {
            "paper_bgcolor": BG,
            "font": {"color": FONT_COLOR, "family": "Inter, sans-serif"},
            "margin": {"t": 60, "b": 10, "l": 20, "r": 20},
            "height": 300,
        },
    }

    # ── 2. Bar — Missing Skills by Learning Hours ────────────────────────
    breakdown = timeline.get("breakdown", [])

    if breakdown:
        skill_labels = [b["skill"].title() for b in breakdown]
        skill_hours  = [b["learning_hours"] for b in breakdown]
        skill_colors = [difficulty_color(b["difficulty_weight"]) for b in breakdown]
        skill_text   = [
            f"{b['learning_hours']}h | {'Easy' if b['difficulty_weight']==1 else 'Medium' if b['difficulty_weight']==2 else 'Hard'}"
            for b in breakdown
        ]

        bar_fig = {
            "data": [{
                "type": "bar",
                "x": skill_hours,
                "y": skill_labels,
                "orientation": "h",
                "marker": {"color": skill_colors, "opacity": 0.9},
                "text": skill_text,
                "textposition": "outside",
                "textfont": {"color": FONT_COLOR, "size": 11},
                "hovertemplate": "<b>%{y}</b><br>%{x} hours<extra></extra>",
            }],
            "layout": {
                "title": {
                    "text": "Missing Skills — Learning Hours Required",
                    "font": {"color": FONT_COLOR, "size": 15},
                },
                "paper_bgcolor": BG,
                "plot_bgcolor":  BG,
                "font": {"color": FONT_COLOR, "family": "Inter, sans-serif"},
                "xaxis": {
                    "title": "Estimated Learning Hours",
                    "gridcolor": "rgba(255,255,255,0.08)",
                    "color": FONT_COLOR,
                },
                "yaxis": {
                    "automargin": True,
                    "color": FONT_COLOR,
                    "categoryorder": "total ascending",
                },
                "margin": {"t": 60, "b": 40, "l": 10, "r": 80},
                "height": max(280, len(breakdown) * 42 + 80),
            },
        }
    else:
        bar_fig = {
            "data": [{"type": "bar", "x": [], "y": []}],
            "layout": {
                "title": {"text": "No Missing Skills 🎉", "font": {"color": COLOR_EASY, "size": 18}},
                "paper_bgcolor": BG,
                "plot_bgcolor": BG,
                "height": 200,
            },
        }

    # ── 3. Donut — Known vs Missing ──────────────────────────────────────
    n_known   = len(timeline.get("already_known", []))
    n_missing = len(timeline.get("missing_skills", []))

    donut_fig = {
        "data": [{
            "type": "pie",
            "labels": ["Already Known", "Need to Learn"],
            "values": [max(n_known, 0), max(n_missing, 0)],
            "hole": 0,
            "marker": {
                "colors": [COLOR_EASY, "#f87171"],
                "line": {"color": "rgba(0,0,0,0.3)", "width": 2},
            },
            "textinfo": "label+percent",
            "textfont": {"color": FONT_COLOR, "size": 12},
            "hovertemplate": "<b>%{label}</b>: %{value} skills<extra></extra>",
        }],
        "layout": {
            "paper_bgcolor": BG,
            "font": {"color": FONT_COLOR, "family": "Inter, sans-serif"},
            "showlegend": False,
            "margin": {"t": 20, "b": 10, "l": 10, "r": 10},
            "height": 260,
        },
    }

    # ── 4. Bar — Company Comparison ──────────────────────────────────────
    compare_fig = None
    if all_companies:
        comp_names = [c["company"] for c in all_companies]
        comp_scores = [c["match_percentage"] for c in all_companies]
        
        # Determine colors (highlight the selected one? Actually let's use gradient or just brand colors)
        # We can just use primary accent for all
        bar_colors = ["#6366f1"] * len(comp_names)
        
        compare_fig = {
            "data": [{
                "type": "bar",
                "x": comp_names,
                "y": comp_scores,
                "marker": {"color": bar_colors, "opacity": 0.8},
                "text": [f"{s}%" for s in comp_scores],
                "textposition": "auto",
                "textfont": {"color": FONT_COLOR},
                "hovertemplate": "<b>%{x}</b><br>Match: %{y}%<extra></extra>",
            }],
            "layout": {
                "title": {
                    "text": "Company Match Comparison",
                    "font": {"color": FONT_COLOR, "size": 15},
                },
                "paper_bgcolor": BG,
                "plot_bgcolor": BG,
                "font": {"color": FONT_COLOR, "family": "Inter, sans-serif"},
                "yaxis": {
                    "title": "Match %",
                    "range": [0, 100],
                    "gridcolor": "rgba(255,255,255,0.08)",
                    "color": FONT_COLOR,
                },
                "xaxis": {
                    "color": FONT_COLOR,
                    "tickangle": -45,
                },
                "margin": {"t": 50, "b": 60, "l": 40, "r": 10},
                "height": 300,
            },
        }

    # ── 5. Radar — Skill Categories ──────────────────────────────────────
    radar_data = timeline.get("radar_data", {})
    radar_fig = None
    if radar_data and sum(radar_data.values()) > 0:
        categories = list(radar_data.keys())
        scores = list(radar_data.values())
        # Close the loop for radar chart
        categories.append(categories[0])
        scores.append(scores[0])

        radar_fig = {
            "data": [{
                "type": "scatterpolar",
                "r": scores,
                "theta": categories,
                "fill": "toself",
                "fillcolor": "rgba(99,102,241,0.3)",
                "line": {"color": "#6366f1", "width": 2},
                "marker": {"color": "#a78bfa", "size": 6}
            }],
            "layout": {
                "polar": {
                    "radialaxis": {
                        "visible": True,
                        "range": [0, 5],
                        "color": "rgba(255,255,255,0.2)",
                        "tickfont": {"color": "rgba(255,255,255,0.5)"},
                        "gridcolor": "rgba(255,255,255,0.1)",
                        "linecolor": "rgba(255,255,255,0.1)"
                    },
                    "angularaxis": {
                        "tickfont": {"color": FONT_COLOR, "size": 11},
                        "linecolor": "rgba(255,255,255,0.1)"
                    },
                    "bgcolor": BG
                },
                "showlegend": False,
                "paper_bgcolor": BG,
                "margin": {"t": 30, "b": 20, "l": 40, "r": 40},
                "height": 280,
            }
        }
    else:
        # Empty state radar
        radar_fig = {
            "data": [],
            "layout": {
                "title": {"text": "Not enough data for Radar mapping", "font": {"color": FONT_COLOR, "size": 12}},
                "paper_bgcolor": BG,
                "plot_bgcolor": BG,
                "height": 280,
            }
        }

    return {
        "gauge_json": json.dumps(gauge_fig),
        "bar_json":   json.dumps(bar_fig),
        "donut_json": json.dumps(donut_fig),
        "compare_json": json.dumps(compare_fig) if compare_fig else "{}",
        "radar_json": json.dumps(radar_fig)
    }

# ─────────────────────────────────────────────────────────────────────────────
# ML — Company Comparison
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_company_matches(student_skills_str: str) -> list[dict]:
    """
    Compute Skill Coverage % for every company and return a ranked list.

    Uses the SAME formula as compute_match():
        coverage = |student ∩ required| / |required| * 100

    For each company, the score is the MAX coverage across all its roles
    (a student only needs to fit ONE role well, not every role).

    This ensures that the numbers in the Company Comparison chart and
    the Placement Report Gauge are derived from identical arithmetic.

    Returns
    -------
    list of dicts: [{"company": "TCS", "match_percentage": 80.0}, ...]
    sorted highest → lowest.
    """
    df = load_jobs()
    student_skills = {s.strip().lower() for s in student_skills_str.split(",") if s.strip()}

    results = []
    for company in PLACEMENT_COMPANIES:
        company_rows = df[df["Company"] == company]
        if company_rows.empty:
            continue

        best_coverage = 0.0
        for _, row in company_rows.iterrows():
            required_skills = {
                s.strip().lower()
                for s in str(row["Skills"]).split(",")
                if s.strip()
            }
            if not required_skills:
                continue
            matched  = student_skills & required_skills
            coverage = len(matched) / len(required_skills) * 100
            if coverage > best_coverage:
                best_coverage = coverage

        results.append({
            "company": company,
            "match_percentage": round(best_coverage, 1)
        })

    # Sort highest → lowest
    results.sort(key=lambda x: x["match_percentage"], reverse=True)
    return results
