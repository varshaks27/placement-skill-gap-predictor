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
