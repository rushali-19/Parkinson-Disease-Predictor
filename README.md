# Parkinson's Disease Predictor

This project applies machine learning techniques to predict the presence of Parkinson's disease using voice and audio recording features. The analysis is implemented in a Python environment and utilizes a dataset of biomedical voice measurements from patients to classify them as either healthy or diagnosed.

## Dataset

The dataset consists of voice measurements from individuals with and without Parkinson's disease. 

* **Features:** Multiple columns containing acoustic attributes from voice recordings (including frequency measurements, variations in amplitude, noise ratios, and signal complexity metrics).
* **Target:** Status (1 = Parkinson's, 0 = Healthy).

## Workflow

1. **Data Loading and Exploration:** The data is loaded and inspected to understand its structure and check for any missing values.
2. **Preprocessing:** The dataset is prepared and split into training and testing sets.
3. **Modeling:** Multiple machine learning algorithms, including Logistic Regression and Random Forest, are trained on the features.
4. **Evaluation:** Models are evaluated using classification accuracy and other performance metrics to compare their effectiveness.
5. **Visualization:** Graphs and plots are generated to analyze feature distributions and model performance.

## Technologies Used

* Python 3
* Jupyter Notebook
* pandas, numpy
* matplotlib, seaborn
* scikit-learn

## Setup and Usage

1. **Prerequisites:** Ensure you have Python and the required libraries installed via pip:
   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn
