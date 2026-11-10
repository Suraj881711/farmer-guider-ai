# 🌾 Farmer Guider AI — Crop Recommendation System

A web app that recommends the best crop to grow based on soil nutrients and
weather conditions, using a trained Random Forest machine-learning model.

## What it does

Enter seven values about your field — Nitrogen (N), Phosphorus (P),
Potassium (K), temperature, humidity, soil pH and rainfall — and the app
predicts the most suitable crop (22 possible crops: rice, maize, chickpea,
cotton, banana, mango, grapes, watermelon, pomegranate, lentil, and more),
along with the model's confidence and the next-best alternatives.

## Features

- Real ML prediction (Random Forest classifier) — not hard-coded rules
- Model confidence + top-3 crop options on the result page
- Your entered values are echoed back on the result page for checking
- Friendly error message on invalid input instead of a crash
- `/health` endpoint

## Verified model performance

- **Test accuracy: 99.32%** (Random Forest, 80/20 train–test split,
  `random_state=42`, reproduced from `farmer_ml.ipynb` on the cleaned
  dataset of 2,200 samples)

## Tech stack

- Python, Flask (web app)
- scikit-learn (Random Forest, StandardScaler, LabelEncoder)
- pandas, NumPy, joblib

## How to run

```bash
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:5000 and click **Predict Crop**.

## Files

- `app.py` — Flask routes (`/`, `/predict`, `/health`)
- `model.py` — loads `ml_farmer.lb` and runs prediction
- `ml_farmer.lb` — trained Random Forest model
- `farmer_ml.ipynb` — model training notebook (Decision Tree, Random
  Forest and Naive Bayes compared)
- `Farmer.ipynb` — data cleaning notebook
- `farmer.csv` — original Crop Recommendation dataset (2,200 rows)
- `new_Cleaned_farmer.csv` — cleaned dataset used for training

## Dataset note

The classic Kaggle "Crop Recommendation" dataset: soil N/P/K, temperature,
humidity, pH and rainfall labelled with the ideal crop for those conditions.

## Note on the model file

The notebook trained the model on scaled features with encoded labels but
saved only the model. `model.py` rebuilds the (deterministic) scaler and
label mapping from the datasets, exactly as the notebook fitted them, so
predictions match the training setup.
