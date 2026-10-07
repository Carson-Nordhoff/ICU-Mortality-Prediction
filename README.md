# ICU-Mortality-Prediction
This machine learning system operates as a patient screening step
after a patient has been in the ICU for 24 hours. The system flags
high risk patients to provide healthcare workers with another layer
of help to identify at risk patients to provide extra care.

## Table of Contents
- [Project Overview](#project-overview)
- [Data & Preprocessing](#data--preprocessing)
- [Cohort & Label](#cohort--label)
- [Features](#features)
- [Design Choices](#design-choices)
- [Model Training & Evaluation](#model-training--evaluation)
- [Results](#results)
- [Limitations & Next Steps](#limitations-and-next-steps)
- [Repo Structure](#repository-structure)
- [System Requirements](#system-requirements)
- [Installation & Setup](#installation--setup)
  - [Postgres 18.6](#postgres-186)
  - [Repository](#repository)
- [Disclaimer](#disclaimer)

## Project Overview
This project implements a machine learning system using the MIMIC-III Database.
It trains a classification model to provide predictions for in-ICU mortality.
These predictions are meant as an extra level of screening for healthcare professionals,
NOT as a replacement for professional healthcare workers.

## Data & Preprocessing
This project is based on the MIMIC-III Clinical Database Demo.
No data is replicated or uploaded to GitHub. Simply download the zip file and unzip into the empty data/ folder.
- Download the dataset: [MIMIC-III Clinical Database Demo](https://physionet.org/content/mimiciii-demo/1.4/)

## Cohort & Label
Cohort:

Because this system is meant to predict in-ICU deaths, only a death during ICU stay (after at least a day of ICU stay) counts as a death.
ICU admissions with early deaths or discharges ARE NOT used in the dataset to avoid easy predictions and data leakage respectively.
ICU admissions with missing labs or vitals ARE included and will be handled in later versions.

Label: 

The MIMIC-III dataset does not come with an in-ICU death flag. This system uses 'icu_expire_flag'.
'icu_expire_flag' is constructed from the provided 'hospital_expire_flag' and
mapping any deaths that are after 24 hours of an ICU intime to 2 hours after an ICU outtime. The 2 interval after
any outtime is to catch any deaths in which the deathtime was not immediately logged. 


## Features

Categorical Features:
- marital_status
- religion
- language
- insurance
- admission_type
- first_careunit
- gender

Labs and Vitals are averaged over all data from 6 hours before ICU intime to 24 hours after.
This is to ensure that any data in the ED-to-ICU window is included and enough data is gathered to be meaningful.

Vitals (averages)
- Heart rate
- Systolic Blood Pressure
- Diastolic Blood Pressure
- Respiratory rate
- Body temperature (Celsius)
- SPO2 (Peripheral Capillary Oxygen Saturation)

Labs (averages)
- Creatinine
- BUN (Blood Urea Nitrogen)
- WBC (White Blood Cell Count)
- Hemoglobin
- Platelets
- Sodium
- Potassium
- Glucose
- Lactate

Other:
- Age (In MIMIC-III, ages above 89 are shifted for privacy. I capped any age above 89 to 91.4, which is standard.)

## Design Choices
- All deaths or ICU discharges within 24 hours of ICU stay are not included. This is to AVOID: Sub-24 hours of data being treated as a 24 hour window and providing the model trivial predictions.
- All chart events and labs are averaged from 6 hours before ICU intime and 24 hours after ICU intime. This is to include data within any ED-to-ICU windows and gain a more holistic physiological baseline.
- StratifiedGroupKFold is used instead of a standard split to avoid entity level leakage (same patient in train/test)

## Model Training & Evaluation
Baseline Model: Logistic Regression
Logistic Regression trains quickly enough to allow many iterations while adjusting the baseline system. It also provides better interpretability in comparison to more complex models.

Framework: Scikit-learn

Primary Metrics: Recall was prioritized to account for the extreme cost of missed positive cases. A missed positive case could lead to a preventable death.

## Results
This version of the project uses the MIMIC-III Demo Database, which is only 100 patients.
Results are noise at this stage and not included.

## Limitations and Next Steps
- No missingness checks (Add missingness flags as features)
- No multi-model analysis (Add XGBoost, Light GBM, Random Forest, SVM)
- Add hyperparameter tuning
- Add MLFlow for experiment tracking, model packaging, etc.
- Add severity scoring
- No interpretability (SHAP)
- No adults only cutoff for data
- Demographic data (language, religion, etc.) are currectly used. Further analysis is needed on a larger dataset.

## Repository Structure
```bash
ICU-Mortality-Prediction/
├── artifacts/
│   ├── logs/
│   ├── metrics/
│   └── models/
├── configs/
│   └── model_selection.yaml
├── data/
├── sql/
│   └── clean_data.sql
├── src/
│   ├── data/
│   │   ├── clean_data.py
│   │   └── load_data.py
│   ├── models/
│   │   ├── evaluate.py
│   │   ├── model_selection.py
│   │   ├── save_model.py
│   │   └── train_model.py
│   └──pipeline.py
├── utils/
│   ├── db_config.py
│   ├── directories.py
│   ├── logger.py
│   └── read_yaml.py
├── README.md
└── requirements.txt
```

## System Requirements
- Python 3.14.7
- Windows 11
- Postgres 18.6

## Installation & Setup
Containerization with Docker is a planned improvement of this project to avoid manual environment setups.

### Repository
1. Clone the repo using 'git clone https://github.com/Carson-Nordhoff/ICU-Mortality-Prediction.git'
2. Unzip the MIMIC-III csv files into the main data folder (ICU-Mortality-Prediction/data)
3. Run these lines of code in succession:
cd ICU-Mortality-Prediction
pip install -r requirements.txt
python src/pipeline.py

### Postgres 18.6
1. Go to the [installer](https://www.postgresql.org/download/)
2. Click 'Windows'
3. Click the red 'Download the Installer' text
4. For the row that says '18.6' as the PostgreSQL Version and Windows x86-64, click the download icon.
5. Run 'createdb mimic3' in a terminal

### Environment Setup
Once the repo is cloned there will be a file named '.env.example'. Rename this to '.env'
Then replace 'yourusername' and 'yourpassword' with said variables you created while installing Postgres 18.6

## Disclaimer
This system is meant as a proof-of-concept and is not intended for clinical use.