
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from src.quality.pipeline import run_quality_pipeline


st.set_page_config(
    page_title="DataSentinel",
    page_icon="🛡️",
    layout="wide",
)

st.title("🛡️ DataSentinel")
st.caption("CSV data quality and anomaly analysis")

uploaded_file = st.file_uploader(
    "Upload a CSV dataset",
    type=["csv"],
)

st.info(
    "Upload a CSV to inspect its quality, validate basic rules, "
    "and detect unusual numeric rows."
)

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.subheader("Dataset overview")
        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", len(df))
        col2.metric("Columns", len(df.columns))
        col3.metric(
            "Missing cells",
            int(df.isna().sum().sum()),
        )

        st.subheader("Quality analysis")

        if st.button("Run analysis", type="primary"):
            with tempfile.NamedTemporaryFile(
                mode="wb",
                suffix=".csv",
                delete=False,
            ) as temp_file:
                temp_file.write(uploaded_file.getvalue())
                temp_path = Path(temp_file.name)

            try:
                report = run_quality_pipeline(
    str(temp_path),
    rules={},
)

                report["file_name"] = uploaded_file.name

                st.metric("Total issues", report["total_issues"])
                st.metric(
                    "Anomalous rows",
                    report["anomalous_rows"],
                )

                st.subheader("Issue details")
                issues = report.get("issues", [])

                if issues:
                    st.dataframe(
                        pd.DataFrame(issues),
                        use_container_width=True,
                    )
                else:
                    st.success("No issues detected.")

                with st.expander("Full JSON report"):
                    st.json(report)

                st.download_button(
                    "Download quality report",
                    data=json.dumps(report, indent=2),
                    file_name="quality_report.json",
                    mime="application/json",
                )
            finally:
                temp_path.unlink(missing_ok=True)

    except Exception as exc:
        st.error(f"Could not analyze this CSV: {exc}")
