# Heart Disease Prediction using Machine Learning Models

This project aims to predict the likelihood of heart disease in patients based on a set of 13 health-related features. By leveraging machine learning models, we analyze patient data to identify patterns and indicators of cardiovascular disease. The models evaluated include K-Nearest Neighbors (KNN), Decision Tree, and Support Vector Machine (SVM).

## Dataset

The analysis is performed on the `heart_disease.csv` dataset, which contains 303 patient records and 14 columns.

### Feature Descriptions

| Feature    | Description                                                                                             |
|------------|---------------------------------------------------------------------------------------------------------|
| `age`      | Age of the patient.                                                                                     |
| `sex`      | Gender of the patient (1 = male; 0 = female).                                                           |
| `cp`       | Chest pain type (0 = typical angina, 1 = atypical angina, 2 = non-anginal pain, 3 = asymptomatic).       |
| `trestbps` | Resting blood pressure (in mm Hg).                                                                      |
| `chol`     | Serum cholesterol in mg/dl.                                                                             |
| `fbs`      | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false).                                                    |
| `restecg`  | Resting electrocardiographic results (0 = normal, 1 = ST-T wave abnormality, 2 = left ventricular hypertrophy). |
| `thalach`  | Maximum heart rate achieved.                                                                            |
| `exang`    | Exercise-induced angina (1 = yes; 0 = no).                                                              |
| `oldpeak`  | ST depression induced by exercise relative to rest.                                                     |
| `slope`    | The slope of the peak exercise ST segment (0 = upsloping, 1 = flat, 2 = downsloping).                    |
| `ca`       | Number of major vessels (0-4) colored by fluoroscopy.                                                   |
| `thal`     | Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect).                                      |
| `target`   | Diagnosis of heart disease (1 = presence; 0 = absence).                                                 |

## Project Workflow

The project follows these key steps:

1.  **Data Preprocessing:**
    *   The `heart_disease.csv` dataset is loaded.
    *   The dataset is checked for missing values (none found).
    *   One duplicated row is identified and removed, resulting in a clean dataset of 302 samples.

2.  **Exploratory Data Analysis (EDA):**
    *   The distribution of the target variable is analyzed, showing 164 samples with heart disease and 138 without.
    *   A correlation matrix is generated to understand relationships between features. Key findings include:
        *   **Positive Correlation with Target:** `cp`, `thalach`, `slope`.
        *   **Negative Correlation with Target:** `exang`, `oldpeak`, `ca`.
    *   Visualizations (boxplots and countplots) are used to analyze the distribution of numerical, binary, and categorical features against the target variable.

3.  **Feature Selection:**
    *   A `RandomForestClassifier` is used to evaluate the importance of each feature.
    *   All 13 features demonstrated some level of importance and were retained for model training.

4.  **Model Development:**
    *   The data is split into an 80% training set and a 20% test set.
    *   Numerical features (`age`, `trestbps`, `chol`, `thalach`, `oldpeak`) are scaled using `StandardScaler`.
    *   Three classification models are trained and evaluated:
        *   **K-Nearest Neighbors (KNN):** An optimal K value of 5 was identified to maximize accuracy.
        *   **Decision Tree Classifier**
        *   **Support Vector Machine (SVM)**

## Results

### Model Performance Comparison

The performance of the three models on the training and test sets is summarized below. The difference (Test - Train) highlights model generalization.

| Model           | Dataset     | Accuracy | Precision | Recall   | F1-Score | Confusion Matrix      |
|-----------------|-------------|----------|-----------|----------|----------|-----------------------|
| **KNN**         | Train (80%) | 0.8548   | 0.8559    | 0.9000   | 0.8541   | `[[89, 22], [13, 117]]` |
|                 | Test (20%)  | 0.8852   | 0.9048    | 1.0000   | 0.8821   | `[[20, 7], [0, 34]]`   |
|                 | *Difference*| *+0.0304*| *+0.0489* | *+0.1000*| *+0.0280*|                       |
|-----------------|-------------|----------|-----------|----------|----------|-----------------------|
| **Decision Tree**| Train (80%)| 1.0000   | 1.0000    | 1.0000   | 1.0000   | `[[111, 0], [0, 130]]` |
|                 | Test (20%)  | 0.8525   | 0.8523    | 0.8824   | 0.8521   | `[[22, 5], [4, 30]]`   |
|                 | *Difference*| *-0.1475*| *-0.1477* | *-0.1176*| *-0.1479*|                       |
|-----------------|-------------|----------|-----------|----------|----------|-----------------------|
| **SVM**         | Train (80%) | 0.8880   | 0.8917    | 0.9462   | 0.8871   | `[[91, 20], [7, 123]]` |
|                 | Test (20%)  | 0.8689   | 0.8814    | 0.9706   | 0.8660   | `[[20, 7], [1, 33]]`   |
|                 | *Difference*| *-0.0191*| *-0.0103* | *+0.0244*| *-0.0211*|                       |

### Conclusion

The **K-Nearest Neighbors (KNN)** model is selected as the champion model.

*   **High Performance:** It achieved the highest accuracy on the test set (88.5%).
*   **Good Generalization:** The Decision Tree model shows significant overfitting, with a perfect score on the training data but a large performance drop on the test data. In contrast, KNN and SVM show much smaller and more acceptable differences between their training and test scores, indicating better generalization.
*   **Excellent Recall:** Most importantly, the KNN model achieved a Recall of 1.0 on the test set. This means it correctly identified all patients with heart disease (**zero false negatives**), which is a critical outcome for a medical diagnostic tool.

## How to Run

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/pavi040315/heart-disease-prediction-ml-project.git
    cd heart-disease-prediction-ml-project
    ```

2.  **Install dependencies:**
    Ensure you have Python and the following libraries installed:
    ```bash
    pip install pandas matplotlib seaborn scikit-learn jupyter
    ```

3.  **Run the notebook:**
    Launch Jupyter Notebook or JupyterLab and open the `Heart_Disease_Prediction_ML_Models(KNN, DT, SVM).ipynb` file.
    ```bash
    jupyter notebook
    ```
    Make sure the `heart_disease.csv` file is in the same directory as the notebook. Run all cells in the notebook to reproduce the analysis and results.
