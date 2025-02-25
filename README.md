# Multilayer (Deep) Neural Network

## Configuration

1. Model
  Model: (10/ReLU) – (8/ReLU)² – (4/ReLU) – (1/Sigmoid)
  The specification above means the following:
  layer 1 with 10 neurons and ReLU activation function
  layers 2 and 3 (2 layers shown as power 2 in the spec) with 8 neurons per each layer and ReLU activation function
  layer 4 with 4 neurons and ReLU activation function
  one (last) layer 5 with one neuron and Sigmoid activation function

2. Dataset
   Stroke Prediction Dataset
   Attribute Information
    1) id: unique identifier
    2) gender: "Male", "Female" or "Other"
    3) age: age of the patient
    4) hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension
    5) heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease
    6) ever_married: "No" or "Yes"
    7) work_type: "children", "Govt_jov", "Never_worked", "Private" or "Self-employed"
    8) Residence_type: "Rural" or "Urban"
    9) avg_glucose_level: average glucose level in blood
    10) bmi: body mass index
    11) smoking_status: "formerly smoked", "never smoked", "smokes" or "Unknown"*
    12) stroke: 1 if the patient had a stroke or 0 if not
    *Note: "Unknown" in smoking_status means that the information is unavailable for this patient

## Outcome
  <img width="1365" alt="image" src="https://github.com/user-attachments/assets/2f4595c2-0203-45ef-a11b-744e03fd1bf5" />
