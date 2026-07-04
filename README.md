# Placement Skill-Gap & Readiness Timeline Predictor

**A machine learning web application that quantifies placement readiness by mapping student skills to real-world job requirements.**

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Flask](https://img.shields.io/badge/Flask-Backend-black)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)
![Plotly](https://img.shields.io/badge/Plotly-Visualization-purple)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5-7952B3)

---

## Overview

During campus placements, students often struggle to answer a deceptively simple question: "How far am I from being industry-ready?" They may know basic Python or SQL, but have no structured way to map that knowledge against the specific stacks demanded by recruiters at companies like TCS, Infosys, or Accenture.

This project solves that problem with a data-driven approach. It compares a student's self-reported skill set against aggregated job requirements using natural language processing, then returns a quantified match score, a ranked list of missing skills, and a realistic timeline to close the gap. Missing skills are extracted via set subtraction between required and possessed skills, mapped to estimated learning hours, and converted into a timeline using `weeks_needed = ceil(total_learning_hours / study_hours_per_week)`.

---

## Objectives

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

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Machine Learning | scikit-learn (TF-IDF, Cosine Similarity) |
| Database | SQLite |
| Visualization | Plotly |
| Frontend | HTML, CSS, Bootstrap 5 |

---

## Project Workflow

1. Student enters their current skill set and target company/role.
2. Skills are cleaned and normalized using regex-based preprocessing.
3. Both student and job-role skills are vectorized using TF-IDF.
4. Cosine similarity is computed to generate a match percentage.
5. Missing skills are identified via set subtraction against the required skill set.
6. A readiness timeline is calculated based on the student's weekly study capacity.
7. Results are visualized through Plotly charts and logged to the SQLite database.

---

## Learning Outcomes

- Gained practical experience applying TF-IDF vectorization and cosine similarity to a real-world NLP problem.
- Learned to design an MVC-inspired Flask architecture with a clear separation between ML logic and web routing.
- Strengthened skills in data preprocessing, including regex-based text sanitization.
- Built experience integrating Plotly visualizations into a Flask front end.
- Improved understanding of database design for tracking historical predictions in SQLite.

---

## Future Enhancements

- **Resume Parsing (NLP)** — Integrate PyMuPDF and spaCy to auto-extract skills from uploaded resumes.
- **Live Job Data** — Replace static CSVs with a connected job-postings API for real-time requirement updates.
- **AI-Generated Learning Paths** — Use an LLM to produce a personalized, week-by-week syllabus for each missing skill.

---

## Author

**Varsha KS**
B.Tech Computer Science Engineering
Mangalam College of Engineering
