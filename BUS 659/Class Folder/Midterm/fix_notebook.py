import json
import re

def fix_notebook(notebook_path):
    """Fix all known issues in the Question_2_Complete notebook."""

    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    fixes_applied = []

    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue

        source = ''.join(cell['source'])
        original_source = source

        # Fix 1: Variable naming inconsistencies
        # Replace model_baseline with baseline_model, etc.
        replacements = {
            'model_baseline': 'baseline_model',
            'model_loglog': 'log_model',
            'model_logit': 'logit_model'
        }

        for old, new in replacements.items():
            if old in source:
                source = re.sub(r'\b' + old + r'\b', new, source)
                fixes_applied.append(f"Cell {i}: Renamed {old} to {new}")

        # Fix 2: Import statement for calibration_curve
        if 'from sklearn.metrics import calibration_curve' in source:
            source = source.replace(
                'from sklearn.metrics import calibration_curve',
                'from sklearn.calibration import calibration_curve'
            )
            fixes_applied.append(f"Cell {i}: Fixed calibration_curve import (sklearn.metrics -> sklearn.calibration)")

        # Fix 3: Add probability clipping before calibration_curve
        if 'calibration_curve' in source and 'np.clip' not in source:
            # Look for calibration_curve calls and add clipping
            if 'prob_true, prob_pred = calibration_curve' in source:
                # Find the y_prob variable being used
                match = re.search(r'calibration_curve\([^,]+,\s*(\w+)', source)
                if match:
                    y_prob_var = match.group(1)
                    # Add clipping before the calibration_curve call
                    clip_line = f"{y_prob_var}_clipped = np.clip({y_prob_var}, 0, 1)\n"
                    # Replace the variable in calibration_curve call
                    source = re.sub(
                        f'calibration_curve\\(([^,]+),\\s*{y_prob_var}',
                        f'calibration_curve(\\1, {y_prob_var}_clipped',
                        source
                    )
                    # Add the clipping line before calibration_curve
                    source = re.sub(
                        r'(prob_true, prob_pred = calibration_curve)',
                        clip_line + r'\1',
                        source
                    )
                    fixes_applied.append(f"Cell {i}: Added probability clipping for calibration_curve")

        # Fix 4: Add optimal threshold calculation if variables are used but not defined
        # Check if this cell uses optimal_threshold_he, y_pred_opt, or metrics_opt
        # but doesn't define them
        uses_opt_threshold = 'optimal_threshold_he' in source
        defines_opt_threshold = 'optimal_threshold_he =' in source or 'optimal_threshold_he=' in source

        if uses_opt_threshold and not defines_opt_threshold:
            # Check if this is after an ROC calculation that we can use
            if 'y_pred_opt' in source or 'metrics_opt' in source:
                # This cell needs the optimal threshold defined
                # We need to add Youden's J calculation
                # Look for earlier context - check if fpr, tpr, thresholds exist
                if 'roc_curve' not in source:
                    # Need to add the optimal threshold calculation
                    threshold_calc = """# Calculate optimal threshold using Youden's J statistic
# Note: Assumes fpr, tpr, thresholds from previous roc_curve call
if 'fpr' in locals() and 'tpr' in locals() and 'thresholds' in locals():
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    optimal_threshold_he = thresholds[best_idx]
else:
    # If ROC curve data not available, use default threshold
    optimal_threshold_he = 0.5

"""
                    # Add at the beginning of the cell
                    source = threshold_calc + source
                    fixes_applied.append(f"Cell {i}: Added optimal threshold calculation (Youden's J)")

        # Fix 5: General error handling - add try/except for cells that might fail
        # This is more conservative, only do if specifically needed

        # Update the cell source if changes were made
        if source != original_source:
            cell['source'] = source.split('\n')
            # Ensure each line ends with \n except the last
            cell['source'] = [line + '\n' if i < len(cell['source']) - 1 else line
                             for i, line in enumerate(cell['source'])]

    return nb, fixes_applied

# Main execution
if __name__ == '__main__':
    notebook_path = 'Question_2_Complete.ipynb'
    backup_path = 'Question_2_Complete_backup.ipynb'

    # Create backup
    import shutil
    shutil.copy(notebook_path, backup_path)
    print(f"Backup created: {backup_path}")

    # Fix the notebook
    fixed_nb, fixes = fix_notebook(notebook_path)

    # Save the fixed notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(fixed_nb, f, indent=1, ensure_ascii=False)

    print(f"\nFixed notebook saved: {notebook_path}")
    print(f"\nTotal fixes applied: {len(fixes)}")
    for fix in fixes:
        print(f"  - {fix}")
