
import io
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.quality.pipeline import run_quality_pipeline


st.set_page_config(
    page_title="DataSentinel | Data Quality",
    page_icon="🛡️",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def analyze_csv(file_bytes: bytes, filename: str):
    """Analyze a CSV and cache its report for repeated uploads."""
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".csv",
        delete=False,
    ) as temp_file:
        temp_file.write(file_bytes)
        temp_path = Path(temp_file.name)

    try:
        report = run_quality_pipeline(str(temp_path), rules={})
        report["file_name"] = filename
        return report
    finally:
        temp_path.unlink(missing_ok=True)


def build_column_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Build a readable profile for every column."""
    total_rows = len(df)
    profile = []

    for column in df.columns:
        missing = int(df[column].isna().sum())
        missing_pct = missing / total_rows * 100 if total_rows else 0

        profile.append({
            "Column": str(column),
            "Data type": str(df[column].dtype),
            "Missing values": missing,
            "Missing (%)": round(missing_pct, 2),
            "Unique values": int(df[column].nunique()),
            "Completeness (%)": round(100 - missing_pct, 2),
        })

    return pd.DataFrame(profile)


def get_issue_count(report, issue_type):
    """Get the count for a particular issue type."""
    return sum(
        int(issue.get("count", 0))
        for issue in report.get("issues", [])
        if issue.get("type") == issue_type
    )


# Sidebar
with st.sidebar:
    st.title("🛡️ DataSentinel")
    st.caption("Data quality intelligence")
    st.divider()

    st.markdown("### How it works")
    st.markdown(
        "1. Upload a CSV dataset\n"
        "2. Inspect its quality\n"
        "3. Identify unusual rows\n"
        "4. Export your report"
    )

    st.divider()
    st.caption("Local analysis · No external AI API required")


# Main header
st.title("Data Quality Dashboard")
st.write(
    "Understand your dataset, identify quality problems, "
    "and investigate unusual numeric records."
)

uploaded_file = st.file_uploader(
    "Choose a CSV dataset",
    type=["csv"],
    help="Upload a CSV file to begin your analysis.",
)

if uploaded_file is None:
    st.info("Upload a CSV file to generate your data quality dashboard.")

    st.markdown("### What you'll get")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Dataset overview**")
        st.caption("Rows, columns, missing values, and completeness.")

    with col2:
        st.markdown("**Quality findings**")
        st.caption("Missing data, duplicate rows, and unusual records.")

    with col3:
        st.markdown("**Exportable reports**")
        st.caption("Download your findings as a JSON report.")

else:
    file_bytes = uploaded_file.getvalue()

    try:
        df = pd.read_csv(io.BytesIO(file_bytes))

        if df.empty:
            st.warning("This CSV contains no data rows.")
            st.stop()

        if len(df.columns) == 0:
            st.warning("No columns were found in this CSV.")
            st.stop()

        profile_df = build_column_profile(df)

        missing_cells = int(df.isna().sum().sum())
        total_cells = len(df) * len(df.columns)
        missing_pct = (
            missing_cells / total_cells * 100
            if total_cells else 0
        )

        # Run button
        st.caption(
            f"Dataset: **{uploaded_file.name}** · "
            f"{len(df):,} rows · {len(df.columns):,} columns"
        )

        run_analysis = st.button(
            "Run data quality analysis",
            type="primary",
            use_container_width=True,
        )

        if run_analysis:
            with st.spinner(
                "Analyzing data quality and unusual numeric records..."
            ):
                report = analyze_csv(file_bytes, uploaded_file.name)

            st.session_state["ds_report"] = report
            st.session_state["ds_file_bytes"] = file_bytes
            st.session_state["ds_filename"] = uploaded_file.name

        # Only show a report for the currently selected dataset.
        same_dataset = (
            st.session_state.get("ds_file_bytes") == file_bytes
            and st.session_state.get("ds_filename") == uploaded_file.name
        )

        if same_dataset and "ds_report" in st.session_state:
            report = st.session_state["ds_report"]
            issues = report.get("issues", [])

            duplicate_rows = get_issue_count(report, "duplicate_rows")
            anomalous_rows = int(report.get("anomalous_rows", 0))

            duplicate_pct = (
                duplicate_rows / len(df) * 100
                if len(df) else 0
            )

            # This score reflects completeness and duplicate rows only.
            quality_score = max(
                0,
                min(100, 100 - missing_pct - duplicate_pct),
            )

            st.divider()
            st.subheader("Dataset health")

            metric1, metric2, metric3, metric4 = st.columns(4)

            metric1.metric(
                "Quality score",
                f"{quality_score:.1f}/100",
                help=(
                    "Based on missing-cell percentage and duplicate-row "
                    "percentage. It does not measure every aspect of quality."
                ),
            )
            metric2.metric("Rows", f"{len(df):,}")
            metric3.metric("Missing cells", f"{missing_cells:,}")
            metric4.metric("Unusual rows", f"{anomalous_rows:,}")

            st.progress(float(quality_score / 100))

            st.caption(
                "The quality score measures completeness and duplicates. "
                "Unusual rows are reported separately because an outlier "
                "is not automatically a data error."
            )

            # Tabs keep the dashboard organized.
            overview_tab, issues_tab, columns_tab, preview_tab = st.tabs([
                "Overview",
                "Issues",
                "Column profile",
                "Data preview",
            ])

            with overview_tab:
                st.subheader("Quality summary")

                left, right = st.columns(2)

                with left:
                    st.markdown("**Completeness**")
                    st.metric(
                        "Cells with values",
                        f"{100 - missing_pct:.2f}%",
                    )
                    st.caption(f"{missing_cells:,} missing cells found.")

                with right:
                    st.markdown("**Duplicate rows**")
                    st.metric(
                        "Duplicate rows detected",
                        f"{duplicate_rows:,}",
                    )
                    st.caption(f"{duplicate_pct:.2f}% of all rows.")

                st.markdown("**Analysis findings**")

                if issues:
                    issue_summary = (
                        pd.DataFrame(issues)
                        .groupby("type", dropna=False)
                        .size()
                        .reset_index(name="Findings")
                        .rename(columns={"type": "Issue type"})
                    )
                    st.dataframe(
                        issue_summary,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.success("No issues were reported by the pipeline.")

                st.warning(
                    "Anomaly detection uses Isolation Forest with its current "
                    "default contamination of 5%. This is a model setting, "
                    "not the confirmed error or fraud rate."
                )

            with issues_tab:
                st.subheader("Detected issues")

                if issues:
                    issue_df = pd.DataFrame(issues)

                    # Put the most useful information first.
                    preferred_columns = [
                        column
                        for column in [
                            "type", "column", "count", "severity"
                        ]
                        if column in issue_df.columns
                    ]
                    other_columns = [
                        column
                        for column in issue_df.columns
                        if column not in preferred_columns
                    ]

                    issue_df = issue_df[
                        preferred_columns + other_columns
                    ]

                    st.dataframe(
                        issue_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.caption(
                        "Review findings before changing or removing any data."
                    )
                else:
                    st.success("No issues were reported.")

            with columns_tab:
                st.subheader("Column-level profile")
                st.dataframe(
                    profile_df,
                    use_container_width=True,
                    hide_index=True,
                )

                columns_with_missing = profile_df[
                    profile_df["Missing values"] > 0
                ]

                if not columns_with_missing.empty:
                    st.markdown("**Columns with missing values**")
                    chart_data = columns_with_missing.set_index("Column")[
                        "Missing (%)"
                    ]
                    st.bar_chart(chart_data)
                else:
                    st.success("All columns are complete.")

            with preview_tab:
                st.subheader("Dataset preview")
                st.caption("Showing the first 100 rows.")
                st.dataframe(
                    df.head(100),
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Download original CSV",
                    data=file_bytes,
                    file_name=uploaded_file.name,
                    mime="text/csv",
                )

            # Downloads
            st.divider()
            st.subheader("Export report")

            download_col1, download_col2 = st.columns(2)

            with download_col1:
                st.download_button(
                    "Download JSON report",
                    data=json.dumps(report, indent=2, default=str),
                    file_name="datasentinel_report.json",
                    mime="application/json",
                    use_container_width=True,
                )

            with download_col2:
                st.download_button(
                    "Download column profile",
                    data=profile_df.to_csv(index=False),
                    file_name="datasentinel_column_profile.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

        else:
            st.info("Click **Run data quality analysis** to generate your report.")

    except Exception as exc:
        st.error(f"Could not read or analyze this CSV: {exc}")
