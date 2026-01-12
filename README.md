# AI Crime Type Prediction

Django web app that classifies AI-related crime types (social engineering, misinformation, hacking, autonomous weapon systems) using a dataset and ML models. Includes separate flows for Service Provider and Remote User roles, plus reporting and dataset export.

## Features
- Crime type prediction using scikit-learn
- Remote user registration and login
- Service provider dashboards and charts
- Dataset export (XLS) and prediction logs

## Tech stack
- Python, Django
- MySQL
- pandas, scikit-learn
- xlwt

## Setup
1) Create and activate a virtual environment.
2) Install dependencies:

```bash
pip install django pandas scikit-learn xlwt mysqlclient
```

3) Create a MySQL database named `artificial_intelligence_crime`.
4) Update DB credentials in `artificial_intelligence_crime/artificial_intelligence_crime/settings.py`.
5) Apply migrations (if you are not importing a database dump):

```bash
python manage.py migrate
```

## Data files
- Place `Datasets.csv` in the repository root (`Datasets.csv` is read by the app).
- The app writes `Results.csv` in the repository root.
- A sample SQL dump is stored under `Database/` in the original project; it is ignored by default in `.gitignore`.

## Run
```bash
python artificial_intelligence_crime/manage.py runserver
```

Then open `http://127.0.0.1:8000/`.
