import mlflow
import pandas as pd
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from joblib import load
from sklearn.preprocessing import StandardScaler

# --- 1. DEFINE YOUR FULL INPUT DATA SCHEMA (with aliasing) ---
# We use Field(alias=...) to handle original names with special characters
class StudentData(BaseModel):
    marital_status: int = Field(alias='Marital status')
    application_mode: int = Field(alias='Application mode')
    application_order: int = Field(alias='Application order')
    course: int = Field(alias='Course')
    daytime_evening_attendance: int = Field(alias='Daytime/evening attendance')
    previous_qualification: int = Field(alias='Previous qualification')
    nacionality: int = Field(alias='Nacionality')
    mother_s_qualification: int = Field(alias="Mother's qualification")
    father_s_qualification: int = Field(alias="Father's qualification")
    mother_s_occupation: int = Field(alias="Mother's occupation")
    father_s_occupation: int = Field(alias="Father's occupation")
    displaced: int = Field(alias='Displaced')
    educational_special_needs: int = Field(alias='Educational special needs')
    debtor: int = Field(alias='Debtor')
    tuition_fees_up_to_date: int = Field(alias='Tuition fees up to date')
    gender: int = Field(alias='Gender')
    scholarship_holder: int = Field(alias='Scholarship holder')
    age_at_enrollment: int = Field(alias='Age at enrollment')
    international: int = Field(alias='International')
    curricular_units_1st_sem_credited: int = Field(alias='Curricular units 1st sem (credited)')
    curricular_units_1st_sem_enrolled: int = Field(alias='Curricular units 1st sem (enrolled)')
    curricular_units_1st_sem_evaluations: int = Field(alias='Curricular units 1st sem (evaluations)')
    curricular_units_1st_sem_approved: int = Field(alias='Curricular units 1st sem (approved)')
    curricular_units_1st_sem_grade: float = Field(alias='Curricular units 1st sem (grade)')
    curricular_units_1st_sem_without_evaluations: int = Field(alias='Curricular units 1st sem (without evaluations)')
    curricular_units_2nd_sem_credited: int = Field(alias='Curricular units 2nd sem (credited)')
    curricular_units_2nd_sem_enrolled: int = Field(alias='Curricular units 2nd sem (enrolled)')
    curricular_units_2nd_sem_evaluations: int = Field(alias='Curricular units 2nd sem (evaluations)')
    curricular_units_2nd_sem_approved: int = Field(alias='Curricular units 2nd sem (approved)')
    curricular_units_2nd_sem_grade: float = Field(alias='Curricular units 2nd sem (grade)')
    curricular_units_2nd_sem_without_evaluations: int = Field(alias='Curricular units 2nd sem (without evaluations)')
    unemployment_rate: float = Field(alias='Unemployment rate')
    inflation_rate: float = Field(alias='Inflation rate')
    gdp: float = Field(alias='GDP')

# --- 2. INITIALIZE THE APP AND LOAD ARTIFACTS ---
app = FastAPI()

scaler = load('scaler.joblib')
model_columns = load('model_columns.joblib')

# Load the model directly from the file path
MODEL_PATH = "mlruns/132888162167750235/models/m-fb9f56db7e994432b44093f1f807ebfd/artifacts"
model = mlflow.pyfunc.load_model(MODEL_PATH)

prediction_labels = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}

# --- 3. CREATE THE PREDICTION ENDPOINT ---
@app.post("/predict")
def predict(data: StudentData):
    # Pydantic's aliasing automatically handles the conversion.
    # We get a DataFrame with the correct, original column names.
    input_df = pd.DataFrame([data.dict(by_alias=True)])

    # --- 4. REPLICATE THE FULL PREPROCESSING PIPELINE ---
    
    # A. Feature Engineering (This will work now!)
    input_df['sem1_pass_rate'] = input_df['Curricular units 1st sem (approved)'] / input_df['Curricular units 1st sem (enrolled)']
    input_df['sem1_pass_rate'].fillna(0, inplace=True)
    input_df['sem1_eval_completion_rate'] = input_df['Curricular units 1st sem (evaluations)'] / input_df['Curricular units 1st sem (enrolled)']
    input_df['sem1_eval_completion_rate'].fillna(0, inplace=True)
    input_df['is_mature_student'] = (input_df['Age at enrollment'] > 25).astype(int)
    input_df['financial_strain'] = ((input_df['Debtor'] == 1) & (input_df['Tuition fees up to date'] == 0)).astype(int)

    # B. One-Hot Encoding
    categorical_cols = [
        'Marital status', 'Application mode', 'Course', 'Daytime/evening attendance',
        'Previous qualification', 'Nacionality', "Mother's qualification", "Father's qualification",
        "Mother's occupation", "Father's occupation", 'Displaced', 'Educational special needs',
        'Debtor', 'Tuition fees up to date', 'Gender', 'Scholarship holder', 'International'
    ]
    input_df = pd.get_dummies(input_df, columns=categorical_cols, drop_first=True)

    # C. Align columns with the model's training columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # D. Scaling
    numerical_cols = scaler.feature_names_in_
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

    # --- 5. MAKE THE PREDICTION ---
    prediction = model.predict(input_df)
    prediction_value = int(prediction[0])
    prediction_label = prediction_labels[prediction_value]

    return {"prediction": prediction_label}

@app.get("/")
def read_root():
    return {"message": "Student Dropout Prediction API is running!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

