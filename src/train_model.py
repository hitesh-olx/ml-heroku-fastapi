""" 
Script to train the machine learning model.
"""

import logging
from sklearn.model_selection import train_test_split
from utils import load_data, process_data
from model import train_model, compute_model_metrics, inference
import joblib


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

# Import data
logging.info("Importing data")
DATA_PATH = 'data/clean_census.csv'
data = load_data(DATA_PATH)

# Train test split
train, test = train_test_split(data, test_size=0.20)


# Categorical features
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Preprocess the data
logging.info("Preprocessing data")
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True
)

X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary",
     training=False, encoder=encoder, lb=lb)


# Train model
logging.info("Training model")
model = train_model(X_train, y_train)

# Scoring
logging.info("Scoring on test set")
y_pred = inference(model, X_test)
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
logging.info(f"Precision: {precision: .2f}. Recall: {recall: .2f}. Fbeta: {fbeta: .2f}")

# Save artifacts
logging.info("Saving artifacts")
joblib.dump(model, 'model/model.pkl')
joblib.dump(encoder, 'model/encoder.pkl')
joblib.dump(lb, 'model/lb.pkl')