import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("stroke_dataset.csv")

# pre-processing dataset
df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})

df.replace('other', np.nan)
df.replace('Unknown', np.nan)
df.replace('N/A', np.nan)

# one-hot encodingss
df = pd.get_dummies(df, columns=['work_type', 'smoking_status'])

X = df.drop(columns=["stroke"]) # drop label
y = df['stroke'] # label

# imputation
df['gender'].fillna(df['gender'].mean(), inplace=True)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

# Split the dataset into 75% train, 25% test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Develop a model
model = keras.Sequential([keras.layers.Dense(10, activation="relu"),
                          keras.layers.Dense(8, activation="relu"),
                          keras.layers.Dense(8, activation="relu"),
                          keras.layers.Dense(4, activation="relu"),
                          keras.layers.Dense(1, activation="sigmoid")])

# Train the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)

print("loss", loss, "accuracy: ", accuracy)