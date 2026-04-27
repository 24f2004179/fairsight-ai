# FairSight — AI Bias Detection Platform

A comprehensive, accessible tool to inspect datasets and ML models for hidden unfairness and discrimination.

## Features

### Fairness Metrics
- **Disparate Impact (80% Rule)** — Legal standard used by the EEOC
- **Statistical Parity Difference** — Raw outcome gap between groups
- **Equalized Odds** — TPR & FPR parity (when predictions are provided)
- **Intersectional Bias** — Cross-attribute bias (e.g., gender × race)
- **Feature Correlation** — Proxy discrimination detection

### Mitigation Strategies
- Pre-processing: Reweighing, Oversampling, Disparate Impact Remover
- In-processing: Adversarial Debiasing, Fairness Constraints
- Post-processing: Equalized Odds, Reject Option Classifier

### Other Features
- Beautiful dark-mode UI
- Interactive charts (bar + doughnut)
- Drag-and-drop file upload
- Sample datasets included
- Export reports to JSON
- Print-friendly layout
- 100% private — data stays in your session

## Setup

### Requirements
- Python 3.8+
- pip

### Install & Run

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
python app.py
```

Then open: http://localhost:5000

## Usage

1. **Upload** your CSV or JSON dataset (or use sample data)
2. **Select** the sensitive attribute column (gender, race, etc.)
3. **Select** the target/outcome column (hired, approved, etc.)
4. Optionally add prediction column and extra sensitive attributes
5. Click **Run Fairness Analysis**
6. Review flags, charts, metrics, and mitigation strategies
7. **Export** the report as JSON

## File Format

Your CSV should have:
- One column for the sensitive attribute (categorical values)
- One column for the outcome/decision (binary: 0 or 1)
- Optional: a model prediction column (binary: 0 or 1)

### Example:
```csv
gender,race,loan_approved,model_prediction
Male,White,1,1
Female,Black,0,0
Female,Hispanic,0,1
```

## Tech Stack

- **Backend**: Python Flask
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Charts**: Chart.js
- **Fonts**: Inter, JetBrains Mono (Google Fonts)
- **Data**: pandas, numpy

## Notes on Fairness

No single fairness metric captures all aspects of discrimination. FairSight intentionally reports **multiple metrics** because they can conflict:
- Satisfying statistical parity may violate equalized odds
- The "right" metric depends on your domain and legal context

Always consult domain experts, ethicists, and affected communities alongside technical audits.
