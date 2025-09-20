# 🎓 Student Dropout Prediction Pipeline

An end-to-end machine learning project designed to predict student dropout risk. This repository contains the complete pipeline, from raw data processing and feature engineering to model training, hyperparameter tuning with MLflow, and deployment as a containerized API with FastAPI and Docker.

## 🚀 Project Overview

The goal of this project is to build a reliable and deployable early-warning system for educational institutions to identify students at a high risk of dropping out. By leveraging a rich dataset of demographic, socio-economic, and academic factors, the system provides a proactive tool for advisors to intervene and offer targeted support.

The project demonstrates a full MLOps lifecycle, emphasizing reproducibility, experiment tracking, and production-readiness.

## ✨ Key Features

- **Data Preprocessing & Feature Engineering**: Implements a robust pipeline to clean data and create insightful new features like academic success rates and financial strain indicators.

- **Model Training & Tuning**: Utilizes XGBoost for classification and systematically tunes hyperparameters to achieve optimal performance.

- **Experiment Tracking**: Integrates MLflow to log all experiment parameters, metrics, and model artifacts, ensuring full reproducibility.

- **API Deployment**: The best-performing model is served via a FastAPI endpoint for real-time, single-student predictions.

- **Containerization**: The entire application is containerized with Docker, making it portable, scalable, and easy to deploy in any environment.

## 🛠️ Tech Stack

| Category | Technologies |
|----------|-------------|
| **Data Science & ML** | Pandas, Scikit-learn, XGBoost |
| **Backend & MLOps** | FastAPI, Uvicorn, MLflow |
| **Containerization** | Docker |
| **Core Language** | Python 3.12 |

## 📈 Model Performance

The champion model, after hyperparameter tuning, achieved the following performance on the held-out test set:

- **Accuracy**: 77.4%

This result demonstrates a strong predictive capability, providing a reliable basis for an early-warning system.

## 📂 Project Structure

```
student-dropout-prediction/
├── .venv/                   # Virtual environment
├── main.py                  # FastAPI application script
├── main_notebook.ipynb      # Main notebook for EDA, training, and MLflow logging
├── dataset (1).csv          # Dataset file
├── scaler.joblib            # Saved scaler object from training
├── model_columns.joblib     # Saved list of model feature names
├── requirements.txt         # Python dependencies
├── Dockerfile               # Blueprint for building the Docker container
├── mlruns/                  # MLflow experiment tracking directory
└── README.md                # This file
```

## ⚙️ Setup and Installation

Follow these steps to set up and run the project locally.

### 1. Prerequisites

- Python 3.12+
- Docker Desktop (optional, for containerized deployment)

### 2. Clone the Repository

```bash
git clone https://github.com/shaguntembhurne/student-dropout-prediction.git
cd student-dropout-prediction
```

### 3. Set Up Virtual Environment & Install Dependencies

```bash
# Create and activate the virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install required libraries
pip install -r requirements.txt
```

### 4. Running the Application

There are two ways to run the prediction service:

#### A) Locally with Python (for development)

This requires you to have run the training notebooks to generate the `.joblib` files and have an active MLflow run to load the model from.

```bash
# Activate virtual environment
source .venv/bin/activate

# Start the FastAPI server
python main.py
```

The API will be available at `http://localhost:8000`.

#### B) With Docker (Recommended - for production/testing)

This is the simplest way to run the application as it's self-contained.

```bash
# 1. Build the Docker image
docker build -t student-predictor .

# 2. Run the Docker container
docker run -p 8000:8000 student-predictor
```

## 🔍 API Usage

### Interactive API Documentation

Once the server is running, you can access:

- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Sample API Request

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "Marital status": 1,
       "Application mode": 15,
       "Application order": 2,
       "Course": 33,
       "Daytime/evening attendance": 1,
       "Previous qualification": 1,
       "Nacionality": 1,
       "Mother'\''s qualification": 19,
       "Father'\''s qualification": 13,
       "Mother'\''s occupation": 4,
       "Father'\''s occupation": 10,
       "Displaced": 1,
       "Educational special needs": 0,
       "Debtor": 0,
       "Tuition fees up to date": 1,
       "Gender": 1,
       "Scholarship holder": 0,
       "Age at enrollment": 20,
       "International": 0,
       "Curricular units 1st sem (credited)": 0,
       "Curricular units 1st sem (enrolled)": 6,
       "Curricular units 1st sem (evaluations)": 6,
       "Curricular units 1st sem (approved)": 6,
       "Curricular units 1st sem (grade)": 13.67,
       "Curricular units 1st sem (without evaluations)": 0,
       "Curricular units 2nd sem (credited)": 0,
       "Curricular units 2nd sem (enrolled)": 6,
       "Curricular units 2nd sem (evaluations)": 6,
       "Curricular units 2nd sem (approved)": 6,
       "Curricular units 2nd sem (grade)": 13.5,
       "Curricular units 2nd sem (without evaluations)": 0,
       "Unemployment rate": 10.8,
       "Inflation rate": 1.4,
       "GDP": 1.74
     }'
```

### Sample Response

```json
{
  "prediction": "Graduate"
}
```

## 🔬 MLflow Tracking

To view experiment tracking and model registry:

```bash
# Activate virtual environment
source .venv/bin/activate

# Start MLflow UI
mlflow ui
```

Visit `http://localhost:5000` to explore experiments, metrics, and model artifacts.

## 🐳 Docker Commands

```bash
# Build the image
docker build -t student-predictor .

# Run the container
docker run -p 8000:8000 student-predictor

# Run in detached mode
docker run -d -p 8000:8000 --name student-api student-predictor

# Stop the container
docker stop student-api

# Remove the container
docker rm student-api
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: [Higher Education Students Performance Evaluation](https://archive.ics.uci.edu/ml/datasets/Higher+Education+Students+Performance+Evaluation)
- MLflow for experiment tracking
- FastAPI for the web framework
- XGBoost for the machine learning model

---

**Made with ❤️ by [Shagun Tembhurne](https://github.com/shaguntembhurne)**
