End-to-End Student Dropout Prediction Pipeline
An end-to-end machine learning project designed to predict student dropout risk. This repository contains the complete pipeline, from raw data processing and feature engineering to model training, hyperparameter tuning with MLflow, and deployment as a containerized API with FastAPI and Docker.

🚀 Project Overview
The goal of this project is to build a reliable and deployable early-warning system for educational institutions to identify students at a high risk of dropping out. By leveraging a rich dataset of demographic, socio-economic, and academic factors, the system provides a proactive tool for advisors to intervene and offer targeted support.

The project demonstrates a full MLOps lifecycle, emphasizing reproducibility, experiment tracking, and production-readiness.

Key Features:
Data Preprocessing & Feature Engineering: Implements a robust pipeline to clean data and create insightful new features like academic success rates and financial strain indicators.

Model Training & Tuning: Utilizes XGBoost for classification and systematically tunes hyperparameters to achieve optimal performance.

Experiment Tracking: Integrates MLflow to log all experiment parameters, metrics, and model artifacts, ensuring full reproducibility.

API Deployment: The best-performing model is served via a FastAPI endpoint for real-time, single-student predictions.

Containerization: The entire application is containerized with Docker, making it portable, scalable, and easy to deploy in any environment.

🛠️ Tech Stack
Data Science & ML: Pandas, Scikit-learn, XGBoost

Backend & MLOps: FastAPI, Uvicorn, MLflow

Containerization: Docker

Core Language: Python 3.12

📈 Model Performance
The champion model, after hyperparameter tuning, achieved the following performance on the held-out test set:

Accuracy: 77.4%

This result demonstrates a strong predictive capability, providing a reliable basis for an early-warning system.

📂 Project Structure
student-dropout-prediction/
├── .venv/                   # Virtual environment
├── main.py                  # FastAPI application script
├── 01_data_analysis.ipynb   # Notebook for EDA and feature engineering (Example)
├── 02_model_training.ipynb  # Notebook for training, tuning, and MLflow logging (Example)
├── scaler.joblib            # Saved scaler object from training
├── model_columns.joblib     # Saved list of model feature names
├── requirements.txt         # Python dependencies
├── Dockerfile               # Blueprint for building the Docker container
└── README.md                # This file

⚙️ Setup and Installation
Follow these steps to set up and run the project locally.

1. Prerequisites
Python 3.12+

Docker Desktop

2. Clone the Repository
git clone [https://github.com/your-username/student-dropout-prediction.git](https://github.com/your-username/student-dropout-prediction.git)
cd student-dropout-prediction

3. Set Up Virtual Environment & Install Dependencies
# Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required libraries
pip install -r requirements.txt

4. Running the Application
There are two ways to run the prediction service:

A) Locally with Uvicorn (for development)
This requires you to have run the training notebooks to generate the .joblib files and have an active MLflow run to load the model from.

# Start the FastAPI server
uvicorn main:app --reload

The API will be available at http://127.0.0.1:8000.

B) With Docker (Recommended - for production/testing)
This is the simplest way to run the application as it's self-contained.

# 1. Build the Docker image
docker build -t student-predictor .

# 2. Run the Docker container
docker run -p 8000:8000 student-predictor
