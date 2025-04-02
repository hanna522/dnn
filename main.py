import pandas as pd
import numpy as np
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Replace specified values with NaN
    df.replace(['other', 'Unknown', 'N/A'], np.nan, inplace=True)
    
    # Map binary categorical variables
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['ever_married'] = df['ever_married'].map({'Yes': 1, 'No': 0})
    df['Residence_type'] = df['Residence_type'].map({'Urban': 1, 'Rural': 0})
    
    # Impute missing values for binary columns using mode
    for col in ['gender', 'ever_married', 'Residence_type']:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Impute missing values for categorical columns using mode
    for col in ['work_type', 'smoking_status']:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Impute missing values for numerical columns using mode (consider mean/median as needed)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # One-hot encoding for categorical variables
    df = pd.get_dummies(df, columns=['work_type', 'smoking_status'])
    
    # Split features and label (assuming 'stroke' is the label)
    X = df.drop(columns=["stroke"])
    y = df["stroke"]
    
    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y

def build_model(input_dim):
    # Build the neural network model with the specified architecture:
    # (10 neurons/ReLU) -> (8 neurons/ReLU) -> (8 neurons/ReLU) -> (4 neurons/ReLU) -> (1 neuron/Sigmoid)
    model = keras.Sequential([
        keras.layers.Dense(32, activation="relu", input_shape=(input_dim,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model

def main():
    # Load and preprocess the data
    X, y = load_and_preprocess_data("stroke_dataset.csv")
    
    # Split the dataset into training and testing sets (75% train, 25% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Build the model
    model = build_model(X_train.shape[1])
    
    # Train the model
    weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
    class_weights = dict(enumerate(weights))
    print("class weights: ", class_weights)
    
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=1, class_weight=class_weights)
    
    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
    print("loss:", loss, "accuracy:", accuracy)
    
     # Predict on test set
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype("int32")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Confusion matrix (optional)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    

if __name__ == "__main__":
    main()
