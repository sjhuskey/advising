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

# Gen-Ed science subgroups
BIO_SCI_PREFIXES = ("BIOL", "HES", "MBIO", "PBIO")
PHYS_SCI_PREFIXES = ("AGSC", "ASTR", "CHEM", "GEOG", "GEOL", "GPHY", "METR", "PHYS")

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


def ns_science_groups(df: pd.DataFrame) -> Dict[str, object]:
    """
    For rows tagged GenEd == 'NS', split into Biological vs Physical science groups by Dept prefixes.
    Returns:
      {
        'bio_courses':  [...],
        'phys_courses': [...],
        'bio_ok': bool,
        'phys_ok': bool,
        'overall_ok': bool,  # both groups satisfied
      }
    """
    if "GenEd" not in df or "Dept" not in df:
        return {
            "bio_courses": [],
            "phys_courses": [],
            "bio_ok": False,
            "phys_ok": False,
            "overall_ok": False,
        }

    ns = df.loc[df["GenEd"] == "NS", ["Course", "Dept"]].copy()
    if ns.empty:
        return {
            "bio_courses": [],
            "phys_courses": [],
            "bio_ok": False,
            "phys_ok": False,
            "overall_ok": False,
        }

    bio = ns[ns["Dept"].isin(BIO_SCI_PREFIXES)]
    phys = ns[ns["Dept"].isin(PHYS_SCI_PREFIXES)]

    bio_courses = list(dict.fromkeys(bio["Course"].tolist()))
    phys_courses = list(dict.fromkeys(phys["Course"].tolist()))

    bio_ok = len(bio_courses) > 0
    phys_ok = len(phys_courses) > 0

    return {
        "Biological Sciences": bio_courses,
        "Physical Sciences": phys_courses,
        "bio_ok": bio_ok,
        "phys_ok": phys_ok,
        "overall_ok": bio_ok and phys_ok,
    }


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

    df["Dept"] = (
        df["Course"].str.extract(r"^([A-Za-z]+)", expand=False).str.upper().fillna("")
    )

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
    cols = [
        "Number",
        "Course",
        "Dept",
        "Title",
        "Hours",
        "Division",
        "GenEd",
        "Items",
    ]
    return df[cols].copy()


# -----------------------
# Summaries (pure functions that RETURN data)
# -----------------------
def gened_summary(df: pd.DataFrame, detailed: bool = False) -> Dict[str, Any]:
    """
    Map each Gen-Ed code to a de-duplicated, order-preserving list of courses.
    If detailed=True, also include NS subgroup coverage:
      {
        'by_code': { 'EN1': [...], 'NS': [...], ... },
        'ns_groups': {
            'bio_courses': [...],
            'phys_courses': [...],
            'bio_ok': bool,
            'phys_ok': bool,
            'overall_ok': bool
        }
      }
    Otherwise (detailed=False), returns {code: [courses]} for backward compatibility.
    """
    tmp = df.loc[df["GenEd"].notna(), ["GenEd", "Course"]]
    courses_by_code = (
        tmp.groupby("GenEd")["Course"].agg(lambda s: list(dict.fromkeys(s))).to_dict()
    )

    all_codes = (
        REQUIREMENT_LABELS.keys() if "REQUIREMENT_LABELS" in globals() else GENED_CODES
    )
    by_code = {code: courses_by_code.get(code, []) for code in all_codes}

    if not detailed:
        return by_code

    # Attach NS subgroup breakdown
    ns = ns_science_groups(df)  # uses Dept prefixes you already defined
    return {"by_code": by_code, "ns_groups": ns}


def ns_science_groups_including_lab(df: pd.DataFrame) -> Dict[str, object]:
    """
    Science requirement validator:
      - Eligible courses are GenEd == 'NS' or 'NSL'
      - Must have at least one Biological science (by dept prefix)
      - Must have at least one Physical science (by dept prefix)
      - Must have at least one lab (NSL) among those sciences (either side)
    Returns a dict with course lists and boolean flags.
    """

    def dedup(seq: List[str]) -> List[str]:
        return list(dict.fromkeys(seq))

    # Make sure we have the columns we need
    required_cols = {"Course", "GenEd"}
    if not required_cols.issubset(df.columns):
        return {
            "all_ns_courses": [],
            "bio_courses": [],
            "phys_courses": [],
            "lab_courses": [],
            "bio_ok": False,
            "phys_ok": False,
            "lab_ok": False,
            "overall_ok": False,
        }

    # Select eligible science rows
    eligible_cols = ["Course", "GenEd"] + (["Dept"] if "Dept" in df.columns else [])
    eligible = df.loc[df["GenEd"].isin(["NS", "NSL"]), eligible_cols].copy()

    if eligible.empty:
        return {
            "all_ns_courses": [],
            "bio_courses": [],
            "phys_courses": [],
            "lab_courses": [],
            "bio_ok": False,
            "phys_ok": False,
            "lab_ok": False,
            "overall_ok": False,
        }

    # Ensure Dept is present
    if "Dept" not in eligible.columns or eligible["Dept"].isna().any():
        eligible["Dept"] = (
            eligible["Course"]
            .astype(str)
            .str.extract(r"^([A-Za-z]+)", expand=False)
            .str.upper()
        )

    # Build groups
    bio_courses = dedup(
        eligible.loc[eligible["Dept"].isin(BIO_SCI_PREFIXES), "Course"].tolist()
    )
    phys_courses = dedup(
        eligible.loc[eligible["Dept"].isin(PHYS_SCI_PREFIXES), "Course"].tolist()
    )
    lab_courses = dedup(eligible.loc[eligible["GenEd"].eq("NSL"), "Course"].tolist())
    all_ns_courses = dedup(eligible["Course"].tolist())

    bio_ok = len(bio_courses) > 0
    phys_ok = len(phys_courses) > 0
    lab_ok = len(lab_courses) > 0
    overall_ok = bio_ok and phys_ok and lab_ok

    return {
        "all_ns_courses": all_ns_courses,
        "bio_courses": bio_courses,
        "phys_courses": phys_courses,
        "lab_courses": lab_courses,
        "bio_ok": bio_ok,
        "phys_ok": phys_ok,
        "lab_ok": lab_ok,
        "overall_ok": overall_ok,
    }


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
    ns_groups: Dict[str, object] | None = None,
) -> Path:
    """
    Write a Markdown report summarizing advising data.
    Expects:
      - df has columns: Course, Title, Hours, Division, GenEd, Items (list)
      - gened_data is a dict: {GenEdCode -> [courses]}
      - hours_data is a dict with keys: total, total_remaining, lower, upper, upper_remaining
      - candl_df is a filtered subset of df for Classics/Letters
      - REQUIREMENT_LABELS is a dict mapping GenEdCode -> human-readable label
    """
    out_path = Path(out_dir) / f"{Path(student_file).stem}_report.md"
    md_lines: List[str] = []

    if ns_groups is None:
        ns_groups = ns_science_groups(df)

    # Header
    md_lines.append(f"# Academic Advising Report for `{Path(student_file).stem}`\n")

    # =========================
    # General Education section
    # =========================
    md_lines.append("## General Education Requirements\n")
    md_lines.append("| Requirement | Courses | Status |")
    md_lines.append("|-------------|---------|--------|")

    for code, label in REQUIREMENT_LABELS.items():
        if code == "NS":
            ns = ns_science_groups_including_lab(df)  # <-- CAPTURE the return

            ns_status = "✅ Satisfied" if ns["overall_ok"] else "❌ Not satisfied"
            course_text = ", ".join(ns["all_ns_courses"]) if ns["all_ns_courses"] else "-"

            md_lines.append(f"| {label} | {course_text} | {ns_status} |")
            md_lines.append(
                f"| ↳ Biological Science (BIOL/HES/MBIO/PBIO) | "
                f"{', '.join(ns['bio_courses']) if ns['bio_courses'] else '-'} | "
                f"{'✅' if ns['bio_ok'] else '❌'} |"
            )
            md_lines.append(
                f"| ↳ Physical Science (AGSC/ASTR/CHEM/GEOG/GEOL/GPHY/METR/PHYS) | "
                f"{', '.join(ns['phys_courses']) if ns['phys_courses'] else '-'} | "
                f"{'✅' if ns['phys_ok'] else '❌'} |"
            )
            md_lines.append(
                f"| ↳ Lab Requirement (NSL) | "
                f"{', '.join(ns['lab_courses']) if ns['lab_courses'] else '-'} | "
                f"{'✅' if ns['lab_ok'] else '❌'} |"
            )
            continue
        # Skip NSL as it's handled above
        if code == "NSL":
            continue


    md_lines.append("")  # <-- blank line to end the table in Markdown

    # =====================
    # Hours Summary section
    # =====================
    hs = hours_data
    md_lines.append("## Credit Hours Summary")
    md_lines.append("| Category | Hours | Remaining |")
    md_lines.append("|----------|-------|-----------|")
    md_lines.append(f"| Total | {hs['total']} | {hs['total_remaining']} |")
    md_lines.append(f"| Lower Division | {hs['lower']} | - |")
    md_lines.append(f"| Upper Division | {hs['upper']} | {hs['upper_remaining']} |")
    md_lines.append("")

    # ================================
    # Classics and Letters section
    # ================================
    md_lines.append("## Classics and Letters Courses")
    md_lines.append(
        "*These are courses that count toward emphases in Classics or Letters.*\n"
    )
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

    # =====================
    # Other courses section
    # =====================
    if other_courses_df is not None and not other_courses_df.empty:
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

    # Footer
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
    ns_science = ns_science_groups(df)
    hs = hours_summary(df)
    candl_df = candl(args.prefix, df)
    other_courses_df = other_courses(args.prefix, df)

    md_path = make_markdown_report(df, ge, ns_science, hs, candl_df, other_courses_df, args.file, args.out_dir)
    if args.format in ("html", "both"):
        html_path = write_html_from_markdown(md_path, args.out_dir)
        print(f"🌐 HTML report saved to: {html_path}")
        print(f"✅ Markdown report saved to: {md_path}")

if __name__ == "__main__":
    main()
