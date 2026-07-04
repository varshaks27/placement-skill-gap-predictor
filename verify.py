"""Verification script — run once to check all components work end-to-end."""
import os
os.environ["PYTHONUTF8"] = "1"

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Test 1: DB init ───────────────────────────────────────────────────────────
from app import app, init_db
init_db()
print("[CHECK] DB init: OK")

# ── Test 2: Model loading ─────────────────────────────────────────────────────
import model as ml
companies = ml.get_companies()
print(f"[CHECK] Companies loaded: {len(companies)} total")
print(f"[CHECK] First 5 companies: {companies[:5]}")

# ── Test 3: Role lookup ───────────────────────────────────────────────────────
test_company = "Infosys"
roles = ml.get_roles(test_company)
if not roles:
    test_company = companies[0]
    roles = ml.get_roles(test_company)
print(f"[CHECK] Roles for {test_company}: {roles[:3]}")

# ── Test 4: Required skills ───────────────────────────────────────────────────
target_role = roles[0]
required = ml.get_required_skills(test_company, target_role)
print(f"[CHECK] Required skills for {target_role}: {required}")

# ── Test 5: ML match ──────────────────────────────────────────────────────────
student = "python, sql, django"
match = ml.compute_match(student, required)
print(f"[CHECK] Match score: {match}%")

# ── Test 6: Timeline prediction ───────────────────────────────────────────────
timeline = ml.predict_timeline(student, required, 10)
print(f"[CHECK] Missing skills: {timeline['missing_skills']}")
print(f"[CHECK] Total learning hours: {timeline['total_learning_hours']}h")
print(f"[CHECK] Predicted weeks @ 10 hrs/wk: {timeline['weeks_needed']}")

# ── Test 7: Chart building ────────────────────────────────────────────────────
import json
charts = ml.build_charts(match, timeline)
for k in ("gauge_json", "bar_json", "donut_json"):
    data = json.loads(charts[k])
    assert "data" in data and "layout" in data, f"Bad chart JSON for {k}"
print("[CHECK] All 3 Plotly charts built: OK")

print()
print("=" * 50)
print("  ALL CHECKS PASSED — Flask app is ready to run")
print("  Run:  python app.py")
print("=" * 50)
