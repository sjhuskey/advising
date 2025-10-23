# streamlit run advising_app.py
import streamlit as st
import pandas as pd
import io, csv, re
from pathlib import Path

# --- import from your module or paste the functions directly ---
from advise import (
    other_courses,
    prepare_df,
    gened_summary,
    hours_summary,
    write_html_from_markdown,
    make_markdown_report,
    CLASSICS_LETTERS_PREFIXES,
    REQUIREMENT_LABELS,
    candl,
)

DEFAULT_PREFIX = CLASSICS_LETTERS_PREFIXES


st.set_page_config(page_title="Academic Advising Report", layout="wide")
st.title("Academic Advising Report Generator")

uploaded = st.file_uploader(
    "Upload student TSV (tab-separated)", type=["tsv", "txt", "csv"]
)
# prefix = st.text_input("Dept prefix regex for Classics & Letters", value=DEFAULT_PREFIX)
out_name = st.text_input("Output base name (no extension)", value="student_report")
make_html = st.checkbox("Also generate HTML", value=True)

if uploaded:
    # Read TSV robustly
    df_raw = pd.read_csv(
        uploaded,
        sep="\t",
        header=None,
        quoting=csv.QUOTE_NONE,
        engine="python",
        dtype=str,
        on_bad_lines="error",
    )
    df_raw.columns = ["Number", "Course", "Data"]

    df = prepare_df(df_raw)
    ge = gened_summary(df)
    hs = hours_summary(df)
    candl_df = candl(CLASSICS_LETTERS_PREFIXES, df)
    other_courses_df = other_courses(CLASSICS_LETTERS_PREFIXES, df)

    st.subheader("Credit Hours Summary")

    hours_df = pd.DataFrame(
        {
            "Category": ["Total", "Lower Division", "Upper Division"],
            "Hours": [hs["total"], hs["lower"], hs["upper"]],
            "Remaining": [str(hs["total_remaining"]), "-", str(hs["upper_remaining"])],
        }
    )

    st.table(hours_df)


    st.subheader("General Education Coverage")
    ge_rows = []
    for code, label in REQUIREMENT_LABELS.items():
        courses = ", ".join(ge.get(code, [])) if ge.get(code) else "-"
        status = "✅" if ge.get(code) else "❌"
        ge_rows.append({"Requirement": label, "Courses": courses, "Status": status})
    st.table(pd.DataFrame(ge_rows))

    st.subheader("Classics & Letters Courses")
    st.dataframe(
        candl_df[["Course", "Title", "Hours", "Division"]], width="stretch"
    )

    st.subheader("All Other Courses")
    st.dataframe(
        other_courses_df[["Course", "Title", "Hours", "Division"]], width="stretch"
    )

    if st.button("Generate report"):
        # Create Markdown in-memory
        md_path = Path(f"{out_name}.md")
        # Reuse your writer
        tmp = make_markdown_report(
            df, ge, hs, candl_df, other_courses_df, student_file=out_name, out_dir="."
        )
        # Return as download
        with open(tmp, "rb") as f:
            st.download_button(
                "Download Markdown", f, file_name=md_path.name, mime="text/markdown"
            )

        if make_html:
            # Convert and offer an HTML download
            html_path = write_html_from_markdown(tmp, ".")
            with open(html_path, "rb") as f:
                st.download_button(
                    "Download HTML", f, file_name=html_path.name, mime="text/html"
                )
