# Multilayer (Deep) Neural Network

## Lab1

   1. Model Architecture

      *File Name: "main.py"*

      | Layer Type       | Layer Index | Description                                                                 |
      |------------------|-------------|-----------------------------------------------------------------------------|
      | **Input Layer**  | N/A         | Implicitly defined using `input_shape=(input_dim,)`                         |
      | **Hidden Layer** | Layer 1     | `Dense(10, activation="relu")` – first hidden layer                         |
      |                  | Layer 2     | `Dense(8, activation="relu")` – second hidden layer                         |
      |                  | Layer 3     | `Dense(8, activation="relu")` – third hidden layer                          |
      |                  | Layer 4     | `Dense(4, activation="relu")` – fourth hidden layer                         |
      | **Output Layer** | Layer 5     | `Dense(1, activation="sigmoid")` – output layer for binary classification   |

   
   2. Training Set
      
      *File Name: "stroke_dataset.csv"*
      
      - Stroke Prediction Dataset
      - Attribute
        | No. | Column Name         | Description                                                                 |
        |-----|---------------------|-----------------------------------------------------------------------------|
        | 1   | `id`                | Unique identifier                                                           |
        | 2   | `gender`            | Gender: `"Male"`, `"Female"`, or `"Other"`                                  |
        | 3   | `age`               | Age of the patient                                                          |
        | 4   | `hypertension`      | `0` if no hypertension, `1` if the patient has hypertension                 |
        | 5   | `heart_disease`     | `0` if no heart disease, `1` if the patient has a heart disease             |
        | 6   | `ever_married`      | Marital status: `"No"` or `"Yes"`                                          |
        | 7   | `work_type`         | Type of work: `"children"`, `"Govt_job"`, `"Never_worked"`, `"Private"`, `"Self-employed"` |
        | 8   | `Residence_type`    | Type of residence: `"Rural"` or `"Urban"`                                  |
        | 9   | `avg_glucose_level` | Average glucose level in blood                                              |
        | 10  | `bmi`               | Body Mass Index (BMI)                                                       |
        | 11  | `smoking_status`    | Smoking status: `"formerly smoked"`, `"never smoked"`, `"smokes"`, or `"Unknown"` |
        | 12  | `stroke`            | Target: `1` if the patient had a stroke, `0` if not                         |
    
## Lab2

1. Model Training Configuration

   *File Name: "main.py"*
  
   - **Epochs**: 10
   - **Batch size**: 32
   - **Verbose**: 1 (Displays training progress per epoch)
   - **Optimizer**: Adam
   - **Loss Function**: binary_crossentropy
  
2. Training Result
   
  <img width="1365" alt="image" src="https://github.com/user-attachments/assets/2f4595c2-0203-45ef-a11b-744e03fd1bf5" />

## Lab3
1. Model Tuning
2. 
