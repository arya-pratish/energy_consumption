import pickle
from sklearn.metrics import r2_score
import pandas as pd
import preprocessing as p


# Load CSV file
df = pd.read_csv('data.csv')

# Pre-processing 
df = p.pre_process(df)
X_train, X_test, y_train, y_test = p.train_and_test(df)


# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Predict and evaluate
y_pred = model.predict(X_test)
R2 = r2_score(y_test, y_pred)
print(f"Model accuracy: {R2}")

# Threshold condition
THRESHOLD = 0.85
def retrain_or_not():

    if R2 < THRESHOLD:
        print("Accuracy below threshold. Retraining model...")
    # Import your training logic
        from model import train_model
        train_model()
        return 1
    else:
        print("Model accuracy is sufficient. Skipping retraining.")
        return 0