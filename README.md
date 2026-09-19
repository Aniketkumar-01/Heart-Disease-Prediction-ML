# Heart Disease Prediction Model using ML

**Author**: Aniket Kumar  
**Program**: IBM SkillsBuild Data Analytics with AI Academic Internship Program (conducted by BharatCares in association with AICTE)

## Project Overview
This project focuses on building a Machine Learning pipeline to predict the risk of heart disease based on clinical parameters. By analyzing various medical indicators (such as age, blood pressure, cholesterol levels, etc.), the model predicts whether a patient is at risk for heart disease, aiding in early diagnosis and proactive healthcare.

## Dataset
The project utilizes the **UCI Heart Disease Dataset** (Cleveland).
* **Dataset Link**: [UCI Machine Learning Repository - Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease)

## Technologies Used
* **Programming Language**: Python
* **Libraries/Frameworks**: 
  * `pandas` and `numpy` for data manipulation and preprocessing
  * `scikit-learn` for machine learning algorithms, evaluation metrics, and preprocessing pipelines
  * `xgboost` (optional) for gradient boosting models
  * Jupyter Notebook for interactive data exploration and model development

## Key Information
The pipeline includes data cleaning, outlier handling (IQR capping), and feature scaling (StandardScaler) before training multiple classification models. Models evaluated typically include Random Forest, Logistic Regression, and Multi-Layer Perceptron (MLP) to determine the best performing algorithm.

## Setup & Run Instructions
1. **Prerequisites**: Ensure you have Python 3.8+ installed on your system.
2. **Install Dependencies**: Open a terminal and run the following command to install required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the Notebook (Optional)**: Launch Jupyter Notebook or Jupyter Lab from your terminal:
   ```bash
   jupyter notebook
   ```
   Open the file `Aniket_Kumar_Heart_Disease_Prediction_Model_Using_ML.ipynb` and run the cells.

4. **Run the Streamlit User Interface**:
   To launch the interactive web application, run the following command in your terminal:
   ```bash
   streamlit run app.py
   ```
   This will open the application in your default web browser, allowing you to input clinical parameters and get real-time predictions.
