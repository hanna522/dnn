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
   - **Class Weight**: Applied using class_weight='balanced' to handle class imbalance in the target variable (stroke)
  
2. Training Result

   **Performance Calculation**
   - The model’s performance was also evaluated using a confusion matrix, which provides insight into the number of correctly and incorrectly classified instances for each class.
   - A threshold of 0.5 was applied to the model’s predicted probabilities to convert them into binary class labels.

   **Model Performance**
   - The initial model achieved an overall accuracy of 59%, with relatively high precision (0.73) but very low recall (0.31) for stroke cases. This means that while the model was conservative in predicting stroke (only predicting stroke when it was fairly confident), it missed the majority of actual stroke patients — identifying only 16 out of 51 cases (as shown in the confusion matrix).
   - The imbalance between false negatives (35 cases) and true positives highlights a serious limitation in sensitivity, making the model less suitable for applications where catching positive cases is critical, such as in medical diagnostics.
     
     <img width="342" alt="image" src="https://github.com/user-attachments/assets/0075f7d5-3bed-477c-a3ff-3d7a63bfb9d2" />

## Lab3

1. Model Tuning

   The model was initially trained using a basic neural network architecture with three hidden layers and no regularization. To improve classification performance, especially for the minority class (stroke=1), several key hyperparameters were tuned:

   | Parameter      | Before                              | After                                 |
   |----------------|--------------------------------------|----------------------------------------|
   | Hidden Layers  | 4 layers (10, 8, 8, 4 units)         | 3 layers (32, 128, 32 units)           |
   | Dropout        | None                                 | `Dropout(0.2)` after the first layer   |
   | Epochs         | 10                                   | 30                                     |
   | Batch Size     | 32                                   | 32                                     |
   | class_weight   | Applied (`balanced`)                 | Applied (`balanced`)                   |

3. Tuning Result and Comparison
   
   <img width="350" alt="image" src="https://github.com/user-attachments/assets/20d40c93-d244-4580-a2b8-882a750aa522" />
   
   | Metric              | Before Tuning | After Tuning |
   |---------------------|----------------|---------------|
   | Accuracy            | 0.59           | 0.74          |
   | Precision (stroke)  | 0.73           | 0.70          |
   | Recall (stroke)     | 0.31           | 0.86          |
   | F1-score (stroke)   | 0.44           | 0.77          |
   | False Positives     | 6              | 19            |
   | False Negatives     | 35             | 7             |
   
   - Before tuning: The model had high precision for stroke (0.73) but very low recall (0.31), meaning it missed most stroke cases.
     
   - After tuning: Recall improved dramatically to 0.86, and F1-score rose from 0.44 to 0.77, indicating a much better balance between false positives and false negatives.
     
   - Overall accuracy improved from 59% to 74%, showing better generalization.

      

   
