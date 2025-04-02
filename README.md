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
      
      - **Dataset Description**
        : The dataset contains records of patients along with their health and lifestyle attributes. It is commonly used in healthcare-related machine learning tasks, particularly for early detection and prevention strategies.

         - **Total Attributes:** 12 columns (11 input features + 1 target)
         - **Target Column:** `stroke` (1 = stroke, 0 = no stroke)
         - **Data Types:** mixture of numerical and categorical variables
         - **Missing Values:** Some missing values in the `bmi` and `smoking_status` columns
      
      - **Dataset Attribute**
        
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

      - **Data Pre-processing**
        
         1. **Missing Value Imputation**
            - For binary columns (`gender`, `ever_married`, `Residence_type`), missing values were filled with the **mode**.
            - For categorical columns (`work_type`, `smoking_status`), missing values were filled with the **mode**.
            - For numerical columns (e.g., `age`, `avg_glucose_level`, `bmi`, etc.), missing values were also filled using the **mode**. *(Note: Using mean or median could also be considered depending on distribution.)*

         2. **One-Hot Encoding**
            - Multi-class categorical features (`work_type`, `smoking_status`) were transformed into one-hot encoded vectors using `pd.get_dummies()`.

         3. **Feature-Label Separation**
            - The target variable `stroke` was separated from the feature set.

         4. **Feature Scaling**
            - All feature values were scaled to the range [0, 1] using `MinMaxScaler` from `sklearn.preprocessing`.


## Lab2

1. Model Training Configuration

   *File Name: "main.py"*
  
   - **Epochs**: 10
   - **Batch size**: 32
   - **Verbose**: 1 (Displays training progress per epoch)
   - **Optimizer**: Adam
   - **Loss Function**: binary_crossentropy
  
2. Training Result

   - The model achieved a test accuracy of 78.25%, with a precision of 0.18 and recall of 0.68 for stroke cases. This indicates that the model is able to detect a majority of stroke patients while keeping false positives relatively controlled. The performance suggests a reasonable trade-off between sensitivity and precision, suitable for early-stage medical screening.
   - Over the course of 10 epochs, the training loss consistently decreased from 1.04 to 0.50, and training accuracy improved from 23.5% to 74.1%. This indicates that the model successfully learned patterns in the data over time. Final evaluation on the test set yielded a loss of 0.4381 and an accuracy of 78.25%, suggesting effective generalization to unseen data.

   <img width="901" alt="image" src="https://github.com/user-attachments/assets/1a93c419-0752-4e56-9526-a913cc1d3980" />


## Lab3

1. Model Tuning
2. 
