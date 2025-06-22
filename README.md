# AAI_510_final_project
Repository for MSAAI 510 Machine Learning Final Project

## Obesity Level Estimation Project  

This project predicts obesity levels based on lifestyle and health data using machine learning. It helps identify risk groups and supports preventive healthcare strategies.

### **Dataset**  
**Estimation of Obesity Levels**  
- Contains features like eating habits, physical activity, and demographic data.  
- Target variable: Obesity levels (e.g., Underweight, Normal, Overweight, Obese).  
- Dataset Source: https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

## **Setup & Installation**  
1. **Clone the repository**:  
   - git clone https://github.com/antan123/AAI_510_final_project.git
   - cd AAI_510_final_project
2. **Create a Virtual Environment**:
   - python3 -m venv env
   - env\Scripts\activate.bat (For Windows)
3. **Install dependencies**:  
   pip install -r requirements.txt 

---

## **Project Workflow**  
1. **Exploratory Data Analysis (EDA)**: Visualized data distributions, correlations, and outliers.  
2. **Modelling**: Trained and compared three models:  
   - `RandomForestClassifier`  
   - `XGBoost`  
   - `SVC` (Support Vector Classifier)  
3. **Evaluation**: Used metrics like accuracy, precision, recall, and F1-score.  
4. **Hyperparameter Tuning**: Optimized model performance with GridSearchCV.  
5. **Deployment**: Saved the best model (`XGBoost`) using `joblib` for future use.  

---

## **Results**  
**Best Model**: XGBoost  
- **Accuracy**: 98.8%  
- **Precision**: 98.8%  
- **F1-Score**: 98.8%  

---

## **Real-World Impact**  
This tool can:  
* **Group individuals by obesity risk** (e.g., "High-Risk," "Moderate," "Low-Risk") for targeted interventions.  
* **Help healthcare systems** allocate resources efficiently (e.g., prioritize high-risk groups for check-ups).  
* **Guide personalized lifestyle changes**:  
   - Suggest diet/exercise plans based on predicted risk level.  
   - Monitor progress by re-evaluating users over time.  

