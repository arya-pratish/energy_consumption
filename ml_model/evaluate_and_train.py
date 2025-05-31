import pickle
from sklearn.metrics import r2_score
import pandas as pd
import preprocessing as p
from sklearn.linear_model import LinearRegression

# Load CSV file
df = pd.read_csv('data.csv')

# Pre-processing 
df = p.pre_process(df)
X_train, X_test, y_train, y_test = p.train_and_test(df)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
R2 = r2_score(y_test, y_pred)
print(f"Model accuracy: {R2}")

# Threshold condition
THRESHOLD = 0.85

if R2 < THRESHOLD:
    print("Accuracy below threshold. This model is not OK!")  

else:
    print("Model accuracy is sufficient. OK!")
    