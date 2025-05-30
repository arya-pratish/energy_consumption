import pandas as pd
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, r2_score
import preprocessing as p

# Load CSV file
df = pd.read_csv('data.csv')

# Show original data
print("Original Data:")
print(df.head())

# pre-process the data set
df=p.pre_process(df)

def train_model():
    X_train, X_test, y_train, y_test = p.train_and_test(df)
# Train Linear Regression Model
    model = LinearRegression()
    model.fit(X_train, y_train)

# Predict the model
    y_pred = model.predict(X_test)
    print("RMSE:", root_mean_squared_error(y_test, y_pred))
    print("R² Score:", r2_score(y_test, y_pred))

# Save model to a .pkl file
    with open("model.pkl", "wb") as file:
        pickle.dump(model, file)


train_model()
print("Model trained and saved as model.pkl!")


