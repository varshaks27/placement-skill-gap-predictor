# 🎯 Placement Skill-Gap & Readiness Timeline Predictor

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.x-black?style=for-the-badge&logo=flask&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Charts-purple?style=for-the-badge&logo=plotly&logoColor=white)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5-7952B3?style=for-the-badge&logo=bootstrap&logoColor=white)

A production-grade **Machine Learning web application** designed to help students predict their placement readiness. By analyzing a student's current technical skills against real-world job requirements across top IT companies, the system calculates a personalized skill-match percentage and estimates the exact number of weeks required to become placement-ready.

---

## 📖 Project Overview

### Problem Statement
During campus placements, students often struggle to understand exactly how far they are from being "industry-ready." They might know basic Python or SQL, but they lack a clear roadmap mapping their current skills to the specific tech stacks demanded by companies like TCS, Infosys, and Accenture.

### Objectives
1. **Identify the Skill Gap:** Instantly highlight missing technical skills required for a specific role at a target company.
2. **Quantify Readiness:** Use ML to calculate a precise "Match Percentage".
3. **Predict Timeline:** Estimate the weeks required to become placement-ready based on the student's available weekly study hours.
4. **Compare Options:** Rank 7 major IT companies based on the student's current skill profile.

---

## ✨ Features

- **Company-Specific Targeting:** Filters and analyzes roles strictly across 7 top placement companies (TCS, Infosys, Wipro, Accenture, Capgemini, Cognizant, IBM).
- **ML-Powered Skill Matching:** Utilizes **TF-IDF Vectorization** and **Cosine Similarity** to mathematically compare a student's skills against aggregated job postings.
- **Dynamic Timeline Prediction:** Calculates learning timelines dynamically: `ceil(Total Learning Hours ÷ Study Hours per Week)`.
- **Interactive Visualizations:** Renders beautiful, responsive Plotly charts (Gauges, Bar charts, Donut charts) natively on the frontend.
- **SQLite History Tracking:** Automatically logs predictions to a local database for historical tracking.
- **Premium Glassmorphism UI:** Built with Bootstrap 5, featuring a sleek, dark-mode glass aesthetic.

---

## 🧠 Machine Learning Pipeline

### 1. Data Preprocessing
Skills are sanitized using robust Regular Expressions to strip purely special characters while retaining essential tech symbols (e.g., `C++`, `C#`, `.NET`). The strings are then normalized and space-separated.

### 2. TF-IDF Vectorization
The `TfidfVectorizer` converts the text data (skills) into numerical vectors. 
- **TF (Term Frequency):** Measures how often a skill appears.
- **IDF (Inverse Document Frequency):** Weighs down extremely common skills (like "communication") while highlighting niche, critical technical skills.

### 3. Cosine Similarity
The system computes the Cosine Similarity between the student's skill vector and the job role's skill vector. 
`Similarity = (A · B) / (||A|| × ||B||)`
This returns a value between `0.0` and `1.0`, which is scaled to a `0-100%` Match Score.

### 4. Readiness Prediction Algorithm
Missing skills are extracted using Python Set logic (`Required_Skills - Student_Skills`). Each missing skill is mapped to its `estimated_learning_hours` via a lookup CSV. The timeline is predicted as:
```python
weeks_needed = math.ceil(total_learning_hours / study_hours_per_week)
```

---

## 🏗️ Architecture & Database Design

The application enforces a strict separation of concerns (MVC architecture):
- **`model.py`:** Pure Machine Learning. Handles all Pandas data loading, vectorization, TF-IDF math, and JSON chart generation. Zero Flask logic.
- **`app.py`:** The Controller. Handles HTTP routes, form validation, error handling, flash messages, and SQLite operations.
- **`templates/`:** The Views. Jinja2 templates styled with Bootstrap 5.

### SQLite Database (`database.db`)
| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Primary Key |
| `timestamp` | TEXT | Time of prediction |
| `company` | TEXT | Target company |
| `role` | TEXT | Target job role |
| `match_pct` | REAL | ML Cosine Similarity Score |
| `missing_skills` | TEXT | JSON serialized list |
| `weeks_needed` | INTEGER | Calculated timeline |

---

## 🚀 Installation & Usage

### Prerequisites
- Python 3.10+
- pip

### Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/placement-readiness-predictor.git
   cd placement-readiness-predictor
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the application:**
   ```bash
   python app.py
   ```
4. **Access the Web App:**
   Open your browser and navigate to `http://127.0.0.1:5000/`.

---

## 🔮 Future Enhancements
- **Resume Parsing (NLP):** Integrate PyMuPDF and SpaCy to automatically extract skills from uploaded student resumes.
- **Expanded Dataset:** Connect to an external Job API to dynamically fetch real-time postings rather than relying on static CSVs.
- **Learning Path Generation:** Integrate OpenAI's API to generate a week-by-week syllabus for the missing skills.

---

## 💼 For Recruiters & Interviewers

### Resume Bullet Points
If you are adding this project to your resume, consider using these impactful bullet points:
- *Developed a Flask-based Machine Learning web application utilizing TF-IDF and Cosine Similarity to predict placement readiness with automated skill-gap analysis.*
- *Engineered a dynamic timeline prediction algorithm that calculates learning trajectories based on individualized study paces, reducing uncertainty for placement candidates.*
- *Designed a responsive, production-ready frontend using Bootstrap 5 and Plotly.js, backed by a robust SQLite database handling user histories and analytics.*

### Interview Questions & Answers

**Q: Why did you use TF-IDF instead of simple word matching?**
> A: Simple word matching treats all skills equally. TF-IDF understands the *importance* of a word within a corpus. If a job requires "Python" and "Communication", TF-IDF assigns a higher mathematical weight to "Python" because it is a more distinct, differentiating technical term across all job postings compared to a generic word.

**Q: How did you handle edge cases, such as a student entering symbols or empty strings?**
> A: I implemented robust input validation on both the frontend (HTML5 required tags, max/min limits) and backend. In `model.py`, I use Regular Expressions `re.sub(r'[^a-z0-9\s\+#\.]', '', cleaned)` to strip harmful or useless special characters while explicitly preserving valid tech symbols like `C++` or `C#`. In `app.py`, Flask Flash messages handle graceful error rendering.

**Q: How is the application modular?**
> A: I strictly separated the Machine Learning logic from the web server. `model.py` acts purely as an analytical engine—it takes inputs and returns structured dictionaries or JSON strings. `app.py` acts strictly as the HTTP controller. This allows the ML model to be easily swapped, tested via CLI, or migrated to an API without touching the Flask routes.

---
*Developed with ❤️ for the Campus Placement Community.*
