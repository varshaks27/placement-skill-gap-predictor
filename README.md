# Placement Skill-Gap & Readiness Timeline Predictor

**A machine learning web application that quantifies placement readiness by mapping student skills to real-world job requirements.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-purple)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5-7952B3)

---

## Overview

During campus placements, students often struggle to answer a deceptively simple question: *"How far am I from being industry-ready?"* They may know basic Python or SQL, but have no structured way to map that knowledge against the specific stacks demanded by recruiters at companies like TCS, Infosys, or Accenture.

This project solves that problem with a data-driven approach. It compares a student's self-reported skill set against aggregated job requirements using natural language processing, then returns a quantified match score, a ranked list of missing skills, and a realistic timeline to close the gap.

### Objectives

| Goal | Approach |
|---|---|
| Identify the skill gap | Set-difference analysis between required and possessed skills |
| Quantify readiness | TF-IDF vectorization + cosine similarity match score |
| Predict a timeline | Learning-hour estimates scaled to the student's weekly study capacity |
| Compare options | Side-by-side ranking across 7 major IT recruiters |

---

## Key Features

- **Company-Specific Targeting** — Analyzes roles across seven major recruiters: TCS, Infosys, Wipro, Accenture, Capgemini, Cognizant, and IBM.
- **ML-Powered Skill Matching** — Uses TF-IDF vectorization and cosine similarity to mathematically compare a candidate's skill profile against aggregated job postings.
- **Dynamic Timeline Prediction** — Estimates readiness timelines using `weeks_needed = ceil(total_learning_hours / study_hours_per_week)`.
- **Interactive Visualizations** — Renders responsive Plotly gauge, bar, and donut charts directly in the browser.
- **Prediction History** — Logs every prediction to a local SQLite database for longitudinal tracking.
- **Polished UI** — A dark-mode, glassmorphism interface built on Bootstrap 5.

---

## Machine Learning Pipeline

**1. Data Preprocessing**
Raw skill strings are sanitized with targeted regular expressions that strip non-essential characters while preserving meaningful technical symbols (e.g., `C++`, `C#`, `.NET`), then normalized into space-separated tokens.

**2. TF-IDF Vectorization**
Skill text is converted into numerical vectors using `TfidfVectorizer`:
- **Term Frequency (TF)** — how often a skill appears.
- **Inverse Document Frequency (IDF)** — down-weights common terms (e.g., "communication") and up-weights distinctive technical skills.

**3. Cosine Similarity**
The model computes the cosine similarity between the student's skill vector and each job role's skill vector:
This produces a value between 0.0 and 1.0, scaled to a 0–100% match score.

**4. Readiness Prediction**
Missing skills are extracted via set subtraction (`Required_Skills − Student_Skills`), mapped to estimated learning hours from a reference dataset, and converted into a timeline:

```python
weeks_needed = math.ceil(total_learning_hours / study_hours_per_week)
```

---

## Architecture

The application follows an MVC-inspired separation of concerns:

| Layer | File | Responsibility |
|---|---|---|
| Model | `model.py` | All ML logic — data loading, vectorization, similarity scoring, chart JSON generation. No Flask code. |
| Controller | `app.py` | HTTP routing, form validation, error handling, flash messaging, database operations. |
| View | `templates/` | Jinja2 templates styled with Bootstrap 5. |

### Database Schema (`database.db`)

| Column | Type | Description |
|---|---|---|
| `id` | INTEGER | Primary key |
| `timestamp` | TEXT | Time of prediction |
| `company` | TEXT | Target company |
| `role` | TEXT | Target job role |
| `match_pct` | REAL | Cosine similarity match score |
| `missing_skills` | TEXT | JSON-serialized list of missing skills |
| `weeks_needed` | INTEGER | Predicted readiness timeline (weeks) |

---

## Getting Started

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/placement-readiness-predictor.git
cd placement-readiness-predictor

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Then open **http://127.0.0.1:5000/** in your browser.

---

## Roadmap

- **Resume Parsing (NLP)** — Integrate PyMuPDF and spaCy to auto-extract skills from uploaded resumes.
- **Live Job Data** — Replace static CSVs with a connected job-postings API for real-time requirement updates.
- **AI-Generated Learning Paths** — Use an LLM to produce a personalized, week-by-week syllabus for each missing skill.
