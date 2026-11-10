import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# Load the trained Random Forest model (ml_farmer.lb).
# NOTE: In farmer_ml.ipynb the model was trained on StandardScaler-scaled
# features with LabelEncoder-encoded crop labels, but only the model file
# was saved. Both transforms are deterministic, so they are rebuilt here
# from the same dataset, exactly as the notebook fitted them.
_df = pd.read_csv(os.path.join(BASE_DIR, 'new_Cleaned_farmer.csv'))

# The cleaned dataset stores crop labels as integer codes, so the
# code -> crop-name mapping is rebuilt from the original farmer.csv,
# using the same LabelEncoder ordering that created the codes.
_raw = pd.read_csv(os.path.join(BASE_DIR, 'farmer.csv'))
_label_encoder = LabelEncoder()
_label_encoder.fit(_raw['label'])

_scaler = StandardScaler()
_scaler.fit(_df[FEATURES])

_model = joblib.load(os.path.join(BASE_DIR, 'ml_farmer.lb'))


def predict_crop(n, p, k, temp, humidity, ph, rainfall):
    """Predict the best crop for the given soil/weather values.

    Returns a dict with the recommended crop, the model's confidence
    (top predicted probability, in %) and the top-3 alternatives.
    """
    values = np.array([[n, p, k, temp, humidity, ph, rainfall]], dtype=float)
    scaled = _scaler.transform(values)

    pred_idx = int(_model.predict(scaled)[0])
    crop = str(_label_encoder.inverse_transform([pred_idx])[0])

    confidence = None
    top3 = []
    if hasattr(_model, 'predict_proba'):
        probs = _model.predict_proba(scaled)[0]
        confidence = round(float(probs[pred_idx]) * 100, 1)
        for idx in np.argsort(probs)[::-1][:3]:
            top3.append({
                'crop': str(_label_encoder.inverse_transform([int(idx)])[0]),
                'confidence': round(float(probs[idx]) * 100, 1),
            })

    return {'crop': crop, 'confidence': confidence, 'top3': top3}
