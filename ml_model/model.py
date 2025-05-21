import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load CSV file
df = pd.read_csv('data.csv')

# Show original data
print("Original Data:")
print(df.head())

# Strip whitespace from column names
df.columns = df.columns.str.strip()

# Remove leading/trailing whitespace from string values
df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

# Convert numerical string into integer
#df=df.map(lambda x: (int(x) if x.isdigit() else pd.NA if len(x)==0 else x)
#                               if isinstance(x,str) else x)

# Drop rows with any missing values
df.dropna(inplace=True)

# Drop duplicate rows
df.drop_duplicates(inplace=True)

# Map string variable to integer
df['Building Type'] = df['Building Type'].map({'Residential':0,'Commercial':1,'Industrial':2})
df['Day of Week'] = df['Day of Week'].map({'Weekday':0,'Weekend':1})

# Features (X) and Target (y)
X = df[['Building Type', 'Square Footage', 'Number of Occupants', 'Appliances Used', 'Average Temperature', 'Day of Week' ]]
y = df['Energy Consumption']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model to a .pkl file
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model trained and saved as model.pkl!")


