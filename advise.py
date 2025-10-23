#!/usr/bin/env python3

"""
Academic Advising Tool

Generates a Markdown report summarizing a student's academic progress
based on course data in a CSV file.

Usage:
    python advise.py path/to/student_courses.csv --out-dir reports/

Options:
    --prefix: Regex prefix to filter Classics & Letters courses (default: '^(CLC|GRK|LTRS|LAT)')
    --out-dir: Directory to save the generated Markdown report (default: current directory)

Author: ChatGPT & Samuel J. Huskey
Date: 2025-10-17
"""

from __future__ import annotations
import argparse
import csv
import pandas as pd
from pathlib import Path
import re
import textwrap
from typing import Dict, List

from markdown import markdown

# -----------------------
# Constants / configuration
# -----------------------
GENED_CODES = {
    "NWC",
    "NSL",
    "NS",
    "WC",
    "EN1",
    "EN2",
    "AF",
    "FL",
    "HIST",
    "FYE",
    "PSC",
    "SS",
    "MATH",
    "notgened",
}
FLAGS = {"tr", "SR", "old", "JC", "lib", "OSS"}  # add more if you encounter them
LEVELS = {"ld", "ud"}
TERM_NAMES = {"Fall", "Spring", "Summer", "Winter"}  # adjust if needed

YEAR_SEM_PATTERN = re.compile(r"^\d{4}[A-Za-z]+$")  # e.g., 2014F, 2016Sp
GRADE_PATTERN = re.compile(r"^[A-F][+-]?$")  # A, B+, C-, etc.

REQUIREMENT_LABELS = {
    "EN1": "English I",
    "EN2": "English II",
    "FL": "Foreign Language",
    "MATH": "Math",
    "NS": "Science",
    "NSL": "Science Lab",
    "PSC": "Political Science",
    "SS": "Social Science",
    "AF": "Understanding Artistic Forms",
    "HIST": "History",
    "NWC": "Non-Western C&C",
    "WC": "Western C&C",
    "FYE": "First Year Experience",
}

CLASSICS_LETTERS_PREFIXES = "CLC|GRK|LTRS|LAT|HIST|MLLL|PHIL|RELS|ENGL"

# -----------------------
# Utilities
# -----------------------
def split_parts(s: str) -> list[str]:
    s = str(s).strip().strip("()")
    return [p.strip() for p in s.split(",") if p.strip()]


# -----------------------
# Data preparation
# -----------------------
def prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize expected columns
    df = df.rename(
        columns={"Unnamed: 0": "Number", "Unnamed: 1": "Course", "Unnamed: 2": "Data"}
    )
    df = df.dropna(subset=["Data"]).copy()

    # Clean the 'Course' field (quotes/spaces)
    df["Course"] = (
        df["Course"]
        .astype(str)
        .str.strip()
        .str.strip('"')
        .str.replace(r"\s+", " ", regex=True)
    )

    # --- NEW: extract the comma-separated contents inside parentheses and make a list
    def extract_list(text):
        """
        Take a string like "( 3, 2016Sp, B, Spring, SR, ld, AF, Understanding The Theatre )"
        and return a Python list of items ['3','2016Sp','B','Spring','SR','ld','AF','Understanding The Theatre'].
        """
        if pd.isna(text):
            return []
        # Grab what's inside the outermost parentheses
        match = re.search(r"\((.*)\)", str(text))
        if not match:
            return []
        inner = match.group(1)
        # Split on commas, strip whitespace
        parts = [p.strip() for p in inner.split(",") if p.strip()]
        return parts

    df["Items"] = df["Data"].apply(extract_list)

    # Now build derived columns directly from that list
    df["Hours"] = (
        df["Items"].str[0].pipe(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    )
    df["Division"] = df["Items"].apply(
        lambda xs: "ud" if "ud" in xs else ("ld" if "ld" in xs else None)
    )
    df["GenEd"] = df["Items"].apply(
        lambda xs: next((g for g in xs if g in GENED_CODES), None)
    )
    df["Title"] = df["Items"].apply(
        lambda xs: xs[-2] if "tr" in xs and len(xs) >= 2 else (xs[-1] if xs else None)
    )

    # Reorder columns for clarity
    cols = ["Number", "Course", "Title", "Hours", "Division", "GenEd", "Items"]
    return df[cols].copy()


# -----------------------
# Summaries (pure functions that RETURN data)
# -----------------------
def gened_summary(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Map each Gen-Ed code to a de-duplicated, order-preserving list of courses.
    Assumes df['GenEd'] holds a single code (or None) per row.
    """
    tmp = df.loc[df["GenEd"].notna(), ["GenEd", "Course"]]
    courses_by_code = (
        tmp.groupby("GenEd")["Course"].agg(lambda s: list(dict.fromkeys(s))).to_dict()
    )
    # Ensure all codes are present in the result (even if empty)
    all_codes = (
        REQUIREMENT_LABELS.keys() if "REQUIREMENT_LABELS" in globals() else GENED_CODES
    )
    return {code: courses_by_code.get(code, []) for code in all_codes}


def hours_summary(
    df: pd.DataFrame, upper_required: int = 48, total_required: int = 120
) -> dict:
    lower = int(df.loc[df["Division"].eq("ld"), "Hours"].sum())
    upper = int(df.loc[df["Division"].eq("ud"), "Hours"].sum())
    total = int(df["Hours"].sum())
    return {
        "total": total,
        "lower": lower,
        "upper": upper,
        "upper_remaining": max(upper_required - upper, 0),
        "total_remaining": max(total_required - total, 0),
    }


def candl(prefix, frame):
    no_engl_or_hist = frame[~frame.Course.str.contains("ENGL1113|ENGL1213|EXPO1213|HIST1483|HIST1493")]
    classicsandletters = no_engl_or_hist[no_engl_or_hist.Course.str.contains(CLASSICS_LETTERS_PREFIXES)]

    return classicsandletters

def other_courses(prefix, frame):
    no_engl_or_hist = frame[~frame.Course.str.contains("ENGL1113|ENGL1213|EXPO1213|HIST1483|HIST1493")]
    other_courses = no_engl_or_hist[~no_engl_or_hist.Course.str.contains(CLASSICS_LETTERS_PREFIXES)]

    return other_courses


# -----------------------
# Report generation
# -----------------------
def make_markdown_report(
    df: pd.DataFrame,
    gened_data: Dict[str, List[str]],
    hours_data: Dict[str, int],
    candl_df: pd.DataFrame,
    other_courses_df: pd.DataFrame,
    student_file: str,
    out_dir: str = ".",
) -> Path:
    """
    Write a Markdown report summarizing advising data.
    Expects:
      - df has columns: Course, Title, Hours, Division, GenEd, Items (list)
      - gened_data is a dict: {GenEdCode -> [courses]}
      - hours_data is a dict with keys: total, total_remaining, lower, upper, upper_remaining
      - candl_df is a filtered subset of df for Classics/Letters
      - REQUIREMENT_LABELS is a dict mapping GenEdCode -> human-readable label
    Returns:
      Path to the generated .md file
    """
    out_path = Path(out_dir) / f"{Path(student_file).stem}_report.md"
    md_lines: List[str] = []

    # Header
    md_lines.append(f"# Academic Advising Report for `{Path(student_file).stem}`\n")

    # General Education
    md_lines.append("## General Education Requirements\n")
    md_lines.append("| Requirement | Courses | Status |")
    md_lines.append("|-------------|---------|--------|")
    for code, label in REQUIREMENT_LABELS.items():
        courses = gened_data.get(code, [])
        status = "✅ Satisfied" if courses else "❌ Not satisfied"
        course_text = ", ".join(courses) if courses else "-"
        md_lines.append(f"| {label} | {course_text} | {status} |")
    md_lines.append("")

    # Hours Summary
    hs = hours_data  # just a local alias for readability
    md_lines.append("## Credit Hours Summary")
    md_lines.append("| Category | Hours | Remaining |")
    md_lines.append("|----------|-------|-----------|")
    md_lines.append(f"| Total | {hs['total']} | {hs['total_remaining']} |")
    md_lines.append(f"| Lower Division | {hs['lower']} | - |")
    md_lines.append(f"| Upper Division | {hs['upper']} | {hs['upper_remaining']} |")
    md_lines.append("")

    # Classics and Letters
    md_lines.append("## Classics and Letters Courses")
    md_lines.append("*These are courses that count toward emphases in Classics or Letters*.\n")
    if not candl_df.empty:
        md_lines.append("| Course | Title | Hours | Division |")
        md_lines.append("|--------|-------|-------|----------|")
        for _, r in candl_df.iterrows():
            hours_val = int(r["Hours"]) if pd.notna(r["Hours"]) else 0
            division_val = r.get("Division") or "-"
            title_val = r.get("Title") or "-"
            md_lines.append(
                f"| {r['Course']} | {title_val} | {hours_val} | {division_val} |"
            )
    else:
        md_lines.append("_No Classics or Letters courses found._")
    if not other_courses_df.empty:
        md_lines.append("\n## Other Courses\n")
        md_lines.append("| Course | Title | Hours | Division |")
        md_lines.append("|--------|-------|-------|----------|")
        for _, r in other_courses_df.iterrows():
            hours_val = int(r["Hours"]) if pd.notna(r["Hours"]) else 0
            division_val = r.get("Division") or "-"
            title_val = r.get("Title") or "-"
            md_lines.append(
                f"| {r['Course']} | {title_val} | {hours_val} | {division_val} |"
            )

    md_lines.append("\n---\n")
    md_lines.append("_Generated automatically by Huskey's Academic Advising Tool 🤠._")

    out_path.write_text("\n".join(md_lines), encoding="utf-8")
    return out_path


def write_html_from_markdown(md_path: Path, out_dir: str = ".") -> Path:
    """
    Convert a Markdown file to a styled standalone HTML file.
    Requires: pip install markdown
    """

    html_out = Path(out_dir) / (md_path.stem + ".html")
    md_text = md_path.read_text(encoding="utf-8")

    # Lightweight, print-friendly CSS
    css = """
    <style>
      :root { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
      body { max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.5; }
      h1, h2, h3 { margin-top: 1.6rem; }
      table { border-collapse: collapse; width: 100%; margin: 1rem 0; }
      th, td { border: 1px solid #ddd; padding: 0.5rem; vertical-align: top; }
      th { background: #f5f7fb; text-align: left; }
      code { background: #f6f8fa; padding: 0.1rem 0.3rem; border-radius: 4px; }
      .footer { color: #666; font-size: 0.9rem; margin-top: 2rem; }
    </style>
    """

    body = markdown(md_text, extensions=["tables", "fenced_code"])
    html = f"<!doctype html><html><head><meta charset='utf-8'>{css}</head><body>{body}</body></html>"
    html_out.write_text(html, encoding="utf-8")
    return html_out


# -----------------------
# CLI / Presentation
# -----------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Academic Advising Tool")
    parser.add_argument("file", help="Path to the TSV file containing course data")
    parser.add_argument(
        "--prefix",
        default=rf"^({CLASSICS_LETTERS_PREFIXES})",
        help="Regex for Classics & Letters (department prefixes)",
    )
    parser.add_argument(
        "--out-dir", default=".", help="Directory to save Markdown/HTML reports"
    )
    parser.add_argument(
        "--format", default="md", choices=["md", "html", "both"], help="Report format"
    )
    args = parser.parse_args()

    df_raw = pd.read_csv(
        args.file,
        sep="\t",
        header=None,
        quoting=csv.QUOTE_NONE,
        engine="python",
        dtype=str,
        on_bad_lines="error",
    )

    # If your file has exactly 3 columns (as your examples suggest), name them:
    df_raw.columns = ["Number", "Course", "Data"]

    df_raw.columns = ["Number", "Course", "Data"]

    df = prepare_df(df_raw)
    ge = gened_summary(df)
    hs = hours_summary(df)
    candl_df = candl(args.prefix, df)
    other_courses_df = other_courses(args.prefix, df)

    md_path = make_markdown_report(df, ge, hs, candl_df, other_courses_df, args.file, args.out_dir)
    if args.format in ("html", "both"):
        html_path = write_html_from_markdown(md_path, args.out_dir)
        print(f"🌐 HTML report saved to: {html_path}")
        print(f"✅ Markdown report saved to: {md_path}")

if __name__ == "__main__":
    main()
