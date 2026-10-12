# DataSentinel

DataSentinel is a local Streamlit dashboard for inspecting CSV data quality
and reviewing unusual numeric records.

## Features

- Dataset health metrics: row count, column count, missing cells, and a
  completeness/duplicate-based quality score.
- Issue summaries for missing values, duplicate rows, rule violations, and
  anomaly candidates.
- Column-level profiling and a 100-row data preview.
- A flagged-records table and CSV download when anomaly detection runs.
- JSON report and column-profile downloads.
- No external AI API or remote data upload is required.

## Installation

Requires Python 3.10 or newer.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the dashboard

```powershell
streamlit run app.py
```

Upload a CSV, click **Run data quality analysis**, and inspect the **Overview**
and **Issues** tabs. The **Issues** tab contains the actual rows flagged by
Isolation Forest and provides a CSV download.

## Architecture

```text
app.py
  -> src/quality/pipeline.py
       -> src/quality/profiler.py
       -> src/quality/rules.py
       -> src/anomaly/detector.py
```

The app reads the uploaded bytes into a temporary CSV. The quality pipeline
profiles the file, applies configured quality rules, and runs anomaly
detection only when the dataset has at least five rows and one numeric column.
The temporary file is deleted after analysis.

## Model evaluation and limitations

Isolation Forest is an unsupervised candidate-finding model. It identifies
rows that are isolated in the available numeric feature space; a flagged row
is not automatically invalid, fraudulent, or anomalous in a business sense.
The default contamination is 5%, and the detector uses a fixed random seed
for repeatable results.

The dashboard does not claim accuracy, precision, recall, or fraud-detection
performance because this repository does not provide labeled ground truth.
Evaluate it on representative, labeled data before using it for decisions.
Categorical-only data and datasets with fewer than five rows are reported as
skipped rather than treated as model failures.

## Testing

Run the automated tests from the repository root:

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

The tests cover flagged-record output, categorical-only data, and small
datasets. The dashboard also reports friendly errors for empty, malformed, or
unreadable CSV input.
