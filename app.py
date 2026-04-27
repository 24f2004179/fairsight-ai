from flask import Flask, request, jsonify, render_template, send_from_directory
import pandas as pd
import numpy as np
import os
import json
import uuid
import traceback
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max

os.makedirs('uploads', exist_ok=True)

ALLOWED_EXTENSIONS = {'csv', 'json'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ─── Analysis Engine ────────────────────────────────────────────────────────

def compute_disparate_impact(df, sensitive_col, target_col):
    """80% Rule / Disparate Impact Ratio"""
    rates = df.groupby(sensitive_col)[target_col].mean()
    if rates.max() == 0:
        return None, rates
    ratio = rates.min() / rates.max()
    return round(ratio * 100, 2), rates

def compute_statistical_parity(df, sensitive_col, target_col):
    """Statistical parity difference between groups"""
    rates = df.groupby(sensitive_col)[target_col].mean()
    spd = float(rates.max() - rates.min())
    return round(spd, 4), rates

def compute_group_sizes(df, sensitive_col):
    """Count and proportion of each group"""
    counts = df[sensitive_col].value_counts()
    proportions = df[sensitive_col].value_counts(normalize=True).round(4)
    return counts.to_dict(), proportions.to_dict()

def compute_equalized_odds(df, sensitive_col, target_col, predicted_col=None):
    """If a prediction column exists, compute TPR & FPR parity"""
    if predicted_col is None or predicted_col not in df.columns:
        return None
    results = {}
    for group in df[sensitive_col].unique():
        grp = df[df[sensitive_col] == group]
        tp = ((grp[predicted_col] == 1) & (grp[target_col] == 1)).sum()
        fp = ((grp[predicted_col] == 1) & (grp[target_col] == 0)).sum()
        tn = ((grp[predicted_col] == 0) & (grp[target_col] == 0)).sum()
        fn = ((grp[predicted_col] == 0) & (grp[target_col] == 1)).sum()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        results[str(group)] = {
            'tpr': round(float(tpr), 4),
            'fpr': round(float(fpr), 4),
            'precision': round(float(precision), 4),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
        }
    tpr_values = [v['tpr'] for v in results.values()]
    fpr_values = [v['fpr'] for v in results.values()]
    tpr_diff = round(max(tpr_values) - min(tpr_values), 4)
    fpr_diff = round(max(fpr_values) - min(fpr_values), 4)
    return {'groups': results, 'tpr_diff': tpr_diff, 'fpr_diff': fpr_diff}

def compute_intersectional(df, sensitive_cols, target_col):
    """Analyze bias across intersections of multiple sensitive attributes"""
    if len(sensitive_cols) < 2:
        return None
    df_copy = df.copy()
    df_copy['_intersect'] = df_copy[sensitive_cols].astype(str).agg(' × '.join, axis=1)
    rates = df_copy.groupby('_intersect')[target_col].mean().round(4)
    counts = df_copy.groupby('_intersect')[target_col].count()
    result = []
    for group in rates.index:
        result.append({
            'group': group,
            'rate': float(rates[group]),
            'count': int(counts[group])
        })
    return sorted(result, key=lambda x: x['rate'])

def compute_correlation_matrix(df, sensitive_col, numeric_cols):
    """Correlation of sensitive attribute with numeric features"""
    try:
        df_encoded = df.copy()
        le_vals = {v: i for i, v in enumerate(df[sensitive_col].unique())}
        df_encoded['_sens_enc'] = df_encoded[sensitive_col].map(le_vals)
        correlations = {}
        for col in numeric_cols:
            if col != sensitive_col:
                try:
                    corr = df_encoded['_sens_enc'].corr(pd.to_numeric(df_encoded[col], errors='coerce'))
                    if not np.isnan(corr):
                        correlations[col] = round(float(corr), 4)
                except:
                    pass
        return correlations
    except:
        return {}

def generate_bias_flags(di_score, spd, eq_odds, group_sizes):
    """Generate actionable bias flags"""
    flags = []

    if di_score is not None:
        if di_score < 80:
            flags.append({
                'level': 'critical',
                'icon': '🚨',
                'title': 'Disparate Impact Detected',
                'detail': f'Fairness ratio is {di_score}% — below the legal 80% threshold (4/5ths rule). Minority groups are significantly disadvantaged.',
                'fix': 'Consider resampling, reweighting, or applying fairness-aware algorithms.'
            })
        elif di_score < 90:
            flags.append({
                'level': 'warning',
                'icon': '⚠️',
                'title': 'Mild Disparate Impact',
                'detail': f'Fairness ratio is {di_score}% — approaching the 80% threshold. Monitor closely.',
                'fix': 'Review data collection and labeling processes for hidden biases.'
            })
        else:
            flags.append({
                'level': 'pass',
                'icon': '✅',
                'title': 'Disparate Impact: Pass',
                'detail': f'Fairness ratio is {di_score}% — above the 80% threshold.',
                'fix': None
            })

    if spd > 0.2:
        flags.append({
            'level': 'critical',
            'icon': '🚨',
            'title': 'High Statistical Parity Difference',
            'detail': f'Outcome gap between groups is {spd:.2%}. Groups are treated very differently.',
            'fix': 'Apply pre-processing bias mitigation (e.g., reweighing) or post-processing calibration.'
        })
    elif spd > 0.1:
        flags.append({
            'level': 'warning',
            'icon': '⚠️',
            'title': 'Moderate Statistical Parity Gap',
            'detail': f'Outcome gap is {spd:.2%}. Some differential treatment exists.',
            'fix': 'Investigate root causes in the data pipeline.'
        })

    if eq_odds:
        if eq_odds['tpr_diff'] > 0.1:
            flags.append({
                'level': 'critical',
                'icon': '🚨',
                'title': 'Unequal True Positive Rates (Opportunity Bias)',
                'detail': f'TPR difference across groups: {eq_odds["tpr_diff"]:.2%}. Some groups are less likely to be correctly identified as positive.',
                'fix': 'Use equalized odds post-processing or threshold optimization per group.'
            })
        if eq_odds['fpr_diff'] > 0.1:
            flags.append({
                'level': 'critical',
                'icon': '🚨',
                'title': 'Unequal False Positive Rates (Harm Bias)',
                'detail': f'FPR difference: {eq_odds["fpr_diff"]:.2%}. Some groups face higher false accusation rates.',
                'fix': 'Consider individual fairness or calibration-based approaches.'
            })

    # Check representation imbalance
    sizes = list(group_sizes.values())
    if sizes:
        min_rep = min(sizes)
        max_rep = max(sizes)
        if min_rep / max_rep < 0.1:
            flags.append({
                'level': 'warning',
                'icon': '⚠️',
                'title': 'Severe Underrepresentation',
                'detail': f'Smallest group has <10% the samples of the largest. Results may be unreliable for minority groups.',
                'fix': 'Collect more data for underrepresented groups or use oversampling techniques (SMOTE).'
            })

    return flags

def compute_overall_score(di_score, spd, flags):
    """Compute an overall fairness score 0-100"""
    score = 100
    if di_score is not None:
        if di_score < 80:
            score -= 40
        elif di_score < 90:
            score -= 15

    if spd > 0.2:
        score -= 30
    elif spd > 0.1:
        score -= 15

    critical_count = sum(1 for f in flags if f['level'] == 'critical')
    warning_count = sum(1 for f in flags if f['level'] == 'warning')
    score -= critical_count * 5
    score -= warning_count * 2

    return max(0, min(100, score))

# ─── Routes ─────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV and JSON files are supported'}), 400

    filename = secure_filename(file.filename)
    uid = str(uuid.uuid4())[:8]
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{uid}_{filename}")
    file.save(save_path)

    try:
        if filename.endswith('.json'):
            df = pd.read_json(save_path)
        else:
            df = pd.read_csv(save_path)

        # Basic info
        columns = df.columns.tolist()
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        shape = df.shape

        # Preview (first 10 rows)
        preview = df.head(10).fillna('').astype(str).to_dict(orient='records')

        # Missing values
        missing = df.isnull().sum().to_dict()
        missing = {k: int(v) for k, v in missing.items() if v > 0}

        # Unique counts
        unique_counts = {col: int(df[col].nunique()) for col in df.columns}

        return jsonify({
            'success': True,
            'file_id': save_path,
            'filename': filename,
            'shape': {'rows': shape[0], 'cols': shape[1]},
            'columns': columns,
            'dtypes': dtypes,
            'numeric_cols': numeric_cols,
            'categorical_cols': categorical_cols,
            'preview': preview,
            'missing': missing,
            'unique_counts': unique_counts
        })
    except Exception as e:
        return jsonify({'error': f'Failed to parse file: {str(e)}'}), 400

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        file_id = data.get('file_id')
        sensitive_col = data.get('sensitive_col')
        target_col = data.get('target_col')
        predicted_col = data.get('predicted_col')
        extra_sensitive = data.get('extra_sensitive', [])

        if not file_id or not sensitive_col or not target_col:
            return jsonify({'error': 'Missing required parameters'}), 400

        # Load data
        if file_id.endswith('.json'):
            df = pd.read_json(file_id)
        else:
            df = pd.read_csv(file_id)

        # Validate columns exist
        for col in [sensitive_col, target_col]:
            if col not in df.columns:
                return jsonify({'error': f'Column "{col}" not found in dataset'}), 400

        # Convert target to numeric
        df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
        df = df.dropna(subset=[target_col, sensitive_col])

        if len(df) == 0:
            return jsonify({'error': 'No valid rows after cleaning'}), 400

        # ── Core Metrics ──────────────────────────────────────────────────
        di_score, selection_rates = compute_disparate_impact(df, sensitive_col, target_col)
        spd, _ = compute_statistical_parity(df, sensitive_col, target_col)
        group_counts, group_props = compute_group_sizes(df, sensitive_col)
        eq_odds = compute_equalized_odds(df, sensitive_col, target_col, predicted_col)

        # Correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        correlations = compute_correlation_matrix(df, sensitive_col, numeric_cols)

        # Intersectional analysis
        all_sensitive = [sensitive_col] + [c for c in extra_sensitive if c in df.columns]
        intersectional = compute_intersectional(df, all_sensitive, target_col) if len(all_sensitive) > 1 else None

        # ── Flags & Score ─────────────────────────────────────────────────
        flags = generate_bias_flags(di_score, spd, eq_odds, group_props)
        overall_score = compute_overall_score(di_score, spd, flags)

        # ── Selection rates for chart ──────────────────────────────────────
        sel_rates_dict = {str(k): round(float(v), 4) for k, v in selection_rates.items()}

        # ── Distribution per group ─────────────────────────────────────────
        group_distributions = {}
        for group in df[sensitive_col].unique():
            grp_data = df[df[sensitive_col] == group][target_col]
            group_distributions[str(group)] = {
                'mean': round(float(grp_data.mean()), 4),
                'std': round(float(grp_data.std()), 4),
                'count': int(len(grp_data)),
                'positive': int(grp_data.sum()),
                'negative': int(len(grp_data) - grp_data.sum())
            }

        # ── Mitigation Suggestions ────────────────────────────────────────
        mitigations = get_mitigations(di_score, spd, flags)
        # -------- YOUR AI FAIRNESS ADVISOR --------
        if di_score < 50:
            ai_advice = """
            Severe bias detected.
            Recommendations:
            • Urgent data rebalancing needed
            • Remove proxy discriminatory features
            • Use adversarial debiasing
            • Apply fairness constraints before deployment
            """
        elif di_score < 80:
            ai_advice = """
            Moderate bias detected.
            Recommendations:
            • Apply reweighing
            • Use disparate impact remover
            • Monitor sensitive group outcomes
            """

        else:
            ai_advice = """
            Low bias detected.
            Recommendations:
            • Continue monitoring
            • Periodic fairness audits
            • Validate model drift over time
            """

        result = {
            'success': True,
            'overall_score': overall_score,
            'disparate_impact': di_score,
            'statistical_parity_diff': spd,
            'selection_rates': sel_rates_dict,
            'group_counts': {str(k): int(v) for k, v in group_counts.items()},
            'group_proportions': {str(k): round(float(v), 4) for k, v in group_props.items()},
            'group_distributions': group_distributions,
            'equalized_odds': eq_odds,
            'correlations': correlations,
            'intersectional': intersectional,
            'flags': flags,
            'mitigations': mitigations,
            'ai_advice': ai_advice,
            'dataset_info': {
                'total_rows': len(df),
                'sensitive_col': sensitive_col,
                'target_col': target_col,
                'predicted_col': predicted_col,
                'groups': [str(g) for g in df[sensitive_col].unique().tolist()]
            }
        }
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def get_mitigations(di_score, spd, flags):
    """Return structured mitigation strategies"""
    critical = any(f['level'] == 'critical' for f in flags)
    mitigations = []

    mitigations.append({
        'phase': 'Pre-Processing',
        'icon': '🔧',
        'color': '#6366f1',
        'strategies': [
            {
                'name': 'Reweighing',
                'desc': 'Assign different weights to training instances to reduce discrimination.',
                'when': 'High statistical parity difference',
                'complexity': 'Low'
            },
            {
                'name': 'Disparate Impact Remover',
                'desc': 'Transform feature values to reduce correlation with the sensitive attribute.',
                'when': 'Disparate impact below 80%',
                'complexity': 'Medium'
            },
            {
                'name': 'Oversampling (SMOTE)',
                'desc': 'Synthetically increase underrepresented group samples.',
                'when': 'Severe underrepresentation',
                'complexity': 'Low'
            }
        ]
    })

    mitigations.append({
        'phase': 'In-Processing',
        'icon': '⚙️',
        'color': '#f59e0b',
        'strategies': [
            {
                'name': 'Adversarial Debiasing',
                'desc': 'Train a classifier to predict target while an adversary tries to predict the sensitive attribute.',
                'when': 'Complex bias patterns',
                'complexity': 'High'
            },
            {
                'name': 'Fairness Constraints',
                'desc': 'Add fairness regularization terms to the model\'s loss function.',
                'when': 'Model training phase',
                'complexity': 'Medium'
            },
            {
                'name': 'Meta Fair Classifier',
                'desc': 'Optimize directly for fairness metrics during training.',
                'when': 'Custom fairness metric needed',
                'complexity': 'High'
            }
        ]
    })

    mitigations.append({
        'phase': 'Post-Processing',
        'icon': '📊',
        'color': '#10b981',
        'strategies': [
            {
                'name': 'Equalized Odds Post-Processing',
                'desc': 'Adjust classifier outputs to equalize TPR and FPR across groups.',
                'when': 'Unequal true/false positive rates',
                'complexity': 'Medium'
            },
            {
                'name': 'Calibrated Equalized Odds',
                'desc': 'Apply calibration with equalized odds simultaneously.',
                'when': 'Need probability outputs + fairness',
                'complexity': 'Medium'
            },
            {
                'name': 'Reject Option Classifier',
                'desc': 'Give favorable outcomes to disadvantaged groups in uncertain cases.',
                'when': 'High confidence in uncertain zones',
                'complexity': 'Low'
            }
        ]
    })

    return mitigations

@app.route('/sample-data')
def sample_data():
    """Return sample datasets for demo"""
    datasets = {
        'loan': {
            'name': 'Loan Approval Dataset',
            'description': 'Historical loan approval data with gender and race bias',
            'data': [
                {'gender': 'Male', 'race': 'White', 'income': 75000, 'credit_score': 720, 'loan_approved': 1},
                {'gender': 'Male', 'race': 'White', 'income': 82000, 'credit_score': 750, 'loan_approved': 1},
                {'gender': 'Female', 'race': 'White', 'income': 78000, 'credit_score': 730, 'loan_approved': 1},
                {'gender': 'Female', 'race': 'Black', 'income': 71000, 'credit_score': 710, 'loan_approved': 0},
                {'gender': 'Male', 'race': 'White', 'income': 65000, 'credit_score': 680, 'loan_approved': 1},
                {'gender': 'Female', 'race': 'Hispanic', 'income': 69000, 'credit_score': 695, 'loan_approved': 0},
                {'gender': 'Male', 'race': 'Black', 'income': 80000, 'credit_score': 740, 'loan_approved': 0},
                {'gender': 'Male', 'race': 'White', 'income': 92000, 'credit_score': 780, 'loan_approved': 1},
                {'gender': 'Female', 'race': 'Asian', 'income': 88000, 'credit_score': 760, 'loan_approved': 1},
                {'gender': 'Female', 'race': 'Black', 'income': 73000, 'credit_score': 715, 'loan_approved': 0},
                {'gender': 'Male', 'race': 'Hispanic', 'income': 66000, 'credit_score': 690, 'loan_approved': 0},
                {'gender': 'Male', 'race': 'White', 'income': 95000, 'credit_score': 800, 'loan_approved': 1},
                {'gender': 'Female', 'race': 'White', 'income': 85000, 'credit_score': 755, 'loan_approved': 1},
                {'gender': 'Male', 'race': 'Asian', 'income': 91000, 'credit_score': 770, 'loan_approved': 1},
                {'gender': 'Female', 'race': 'Hispanic', 'income': 62000, 'credit_score': 675, 'loan_approved': 0},
                {'gender': 'Male', 'race': 'Black', 'income': 74000, 'credit_score': 720, 'loan_approved': 0},
                {'gender': 'Female', 'race': 'White', 'income': 89000, 'credit_score': 765, 'loan_approved': 1},
                {'gender': 'Male', 'race': 'White', 'income': 67000, 'credit_score': 700, 'loan_approved': 1},
                {'gender': 'Female', 'race': 'Black', 'income': 76000, 'credit_score': 725, 'loan_approved': 0},
                {'gender': 'Male', 'race': 'Hispanic', 'income': 70000, 'credit_score': 705, 'loan_approved': 0},
            ]
        },
        'hiring': {
            'name': 'Hiring Decision Dataset',
            'description': 'Resume screening with gender bias',
            'data': [
                {'gender': 'Male', 'years_exp': 5, 'education': 'Bachelor', 'interview_score': 82, 'hired': 1},
                {'gender': 'Male', 'years_exp': 3, 'education': 'Master', 'interview_score': 78, 'hired': 1},
                {'gender': 'Female', 'years_exp': 5, 'education': 'Bachelor', 'interview_score': 83, 'hired': 0},
                {'gender': 'Female', 'years_exp': 7, 'education': 'PhD', 'interview_score': 91, 'hired': 1},
                {'gender': 'Male', 'years_exp': 2, 'education': 'Bachelor', 'interview_score': 70, 'hired': 1},
                {'gender': 'Female', 'years_exp': 4, 'education': 'Master', 'interview_score': 85, 'hired': 0},
                {'gender': 'Male', 'years_exp': 6, 'education': 'Master', 'interview_score': 88, 'hired': 1},
                {'gender': 'Female', 'years_exp': 3, 'education': 'Bachelor', 'interview_score': 76, 'hired': 0},
                {'gender': 'Male', 'years_exp': 8, 'education': 'PhD', 'interview_score': 92, 'hired': 1},
                {'gender': 'Female', 'years_exp': 6, 'education': 'Master', 'interview_score': 87, 'hired': 1},
                {'gender': 'Male', 'years_exp': 1, 'education': 'Bachelor', 'interview_score': 65, 'hired': 0},
                {'gender': 'Female', 'years_exp': 5, 'education': 'Master', 'interview_score': 84, 'hired': 0},
                {'gender': 'Male', 'years_exp': 4, 'education': 'Bachelor', 'interview_score': 77, 'hired': 1},
                {'gender': 'Female', 'years_exp': 8, 'education': 'PhD', 'interview_score': 93, 'hired': 1},
                {'gender': 'Male', 'years_exp': 3, 'education': 'Bachelor', 'interview_score': 72, 'hired': 1},
            ]
        }
    }
    return jsonify(datasets)

@app.route('/export-report', methods=['POST'])
def export_report():
    """Generate a JSON report"""
    data = request.get_json()
    return jsonify({'report': data, 'format': 'json'})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
