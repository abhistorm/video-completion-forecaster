from pathlib import Path
import joblib
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / 'models' / 'model.joblib'
FEATURES = ["study_hours_week", "attendance_pct", "quiz_avg", "assignment_lateness", "prior_score", "class_size", "screen_time_hours", "content_duration_min", "engagement_clicks", "interaction_pause_count"]
def get_bundle():
    if not MODEL_PATH.exists(): raise RuntimeError('Train first: python src/train.py --limit 20000')
    return joblib.load(MODEL_PATH)
def predict(values):
    bundle = get_bundle()
    row = {k: float(values.get(k, 0.0)) for k in FEATURES}
    return float(bundle['model'].predict(pd.DataFrame([row], columns=FEATURES))[0])

if __name__ == '__main__':
    print('Train first with: python src/train.py --limit 20000')
    print('Then predict with: python src/predict.py data/sample_request.json')
