"""
clean_data.py
─────────────
One-time data cleaning script.
Reads  : data/job_postings_raw.csv  (india_job_market_2024_2026.csv copy)
Writes : data/job_postings.csv      (cleaned)

Run once before starting the Flask app:
    python data/clean_data.py
"""

import os
import sys
import pandas as pd

# Force UTF-8 output so Japanese characters in the Windows path don't crash print()
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

RAW_PATH     = os.path.join(os.path.dirname(__file__), "job_postings_raw.csv")
CLEANED_PATH = os.path.join(os.path.dirname(__file__), "job_postings.csv")

# ── Target placement companies ────────────────────────────────────────────────
# Maps every variant (post-title-case) → canonical display name
CANONICAL_COMPANY_MAP = {
    "Tcs":        "TCS",
    "Ibm":        "IBM",
    "Infosys":    "Infosys",
    "Wipro":      "Wipro",
    "Accenture":  "Accenture",
    "Capgemini":  "Capgemini",
    "Cognizant":  "Cognizant",
}
PLACEMENT_COMPANIES = set(CANONICAL_COMPANY_MAP.values())


def clean():
    print("[clean_data] Reading raw data from: data/job_postings_raw.csv")
    df = pd.read_csv(RAW_PATH)
    print(f"[clean_data] Raw shape: {df.shape}")

    # ── 1. Keep only the three columns we need ────────────────────────────
    df = df[["Company", "Job_Title", "Skills_Required"]].copy()
    df.rename(columns={"Job_Title": "Role", "Skills_Required": "Skills"}, inplace=True)

    # ── 2. Handle missing values ──────────────────────────────────────────
    df.dropna(subset=["Company", "Role", "Skills"], inplace=True)

    # ── 3. Standardise Company names ──────────────────────────────────────
    #   - strip whitespace → title-case → apply canonical acronym map
    #   - then filter to placement companies only
    df["Company"] = (
        df["Company"]
        .str.strip()
        .str.title()
        .str.replace(r"\s+", " ", regex=True)  # collapse internal spaces
    )
    # Fix acronyms (Tcs → TCS, Ibm → IBM) and drop rows not in target list
    df["Company"] = df["Company"].map(CANONICAL_COMPANY_MAP)
    df.dropna(subset=["Company"], inplace=True)   # NaN = not a target company
    print(f"[clean_data] After company filter ({len(PLACEMENT_COMPANIES)} companies): {df.shape[0]} rows")

    # ── 4. Standardise Role names ─────────────────────────────────────────
    df["Role"] = (
        df["Role"]
        .str.strip()
        .str.title()
        .str.replace(r"\s+", " ", regex=True)
    )

    # ── 5. Normalise Skills column ────────────────────────────────────────
    #   - lowercase each individual skill
    #   - strip spaces around commas
    #   - sort skills alphabetically per row (canonical form)
    def normalise_skills(raw: str) -> str:
        skills = [s.strip().lower() for s in raw.split(",") if s.strip()]
        skills = sorted(set(skills))          # deduplicate + sort
        return ", ".join(skills)

    df["Skills"] = df["Skills"].apply(normalise_skills)

    # ── 6. Drop duplicates ────────────────────────────────────────────────
    df.drop_duplicates(subset=["Company", "Role", "Skills"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # ── 7. Save ───────────────────────────────────────────────────────────
    df.to_csv(CLEANED_PATH, index=False)
    print(f"[clean_data] Final shape: {df.shape}")
    print(f"[clean_data] Companies in output: {sorted(df['Company'].unique())}")
    print(f"[clean_data] Rows per company:\n{df['Company'].value_counts().to_string()}")
    print("[clean_data] Saved to: data/job_postings.csv")
    print(f"[clean_data] Sample:\n{df.head(5).to_string(index=False)}")


if __name__ == "__main__":
    clean()
