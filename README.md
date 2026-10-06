# Revenue Demand Forecasting

## Project Overview

Revenue and demand forecasting is a core problem in retail, e-commerce,
transportation, and airline operations. The objective of this project is
to use historical sales and operational information to estimate future
demand and expected revenue.

This project uses the **Store Sales - Time Series Forecasting** dataset
and is designed as an end-to-end time-series machine learning system.

The project is not limited to a forecasting notebook. It covers:

1.  Loading and validating multiple related datasets.
2.  Performing time-series EDA.
3.  Cleaning and joining operational data.
4.  Creating calendar, lag, and rolling-window features.
5.  Using chronological train/validation/test splits.
6.  Establishing forecasting baselines.
7.  Comparing machine learning models.
8.  Tuning the strongest candidate model.
9.  Evaluating the final model on unseen future-period data.
10. Saving the trained model as a reusable artifact.
11. Exposing predictions through FastAPI.
12. Providing a browser-based prediction interface.
13. Providing predefined sample inputs for testing.
14. Running the application through a single `run.py` entry point.
15. Packaging the application with Docker.
16. Providing automated tests with Pytest.

The overall workflow is:

``` text
Business Problem
      ↓
Raw Time-Series Data
      ↓
Data Ingestion
      ↓
Data Validation
      ↓
EDA & Time-Series Analysis
      ↓
Time-Based Data Cleaning
      ↓
Feature Engineering
      ↓
Lag + Rolling Features
      ↓
Chronological Train / Validation / Test Split
      ↓
Forecasting Baselines
      ↓
Model Comparison
      ↓
Hyperparameter Tuning
      ↓
Final Model
      ↓
Test Evaluation
      ↓
Error Analysis
      ↓
Model Serialization
      ↓
FastAPI
      ↓
Web Dashboard
      ↓
Docker
      ↓
Deployment-ready Application
```

------------------------------------------------------------------------

## Business Problem

A business needs to estimate future product demand using historical
sales and operational information.

Accurate demand forecasts can help a business with:

-   Inventory planning
-   Stock replenishment
-   Workforce planning
-   Promotion planning
-   Supply-chain decisions
-   Store-level demand planning
-   Revenue estimation
-   Identification of unusually high or low demand periods

For an airline-oriented interpretation, the same forecasting
architecture can later be adapted to problems such as passenger demand,
booking demand, route-level demand, or ancillary-revenue forecasting.

The current project intentionally uses the Store Sales dataset because
it provides a realistic multi-store, multi-product time-series problem
with several external variables.

The objective is therefore not simply:

> "Train a regression model on a CSV."

The objective is to build a complete forecasting workflow in which the
model learns from the **past** and is evaluated on **future periods**
without leaking future information into the training features.

------------------------------------------------------------------------

## Dataset

The project uses the **Store Sales - Time Series Forecasting** dataset
from the Kaggle competition:

**Store Sales - Time Series Forecasting**

Dataset source:

`https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data`

The dataset contains several related files rather than a single flat
table.

### Main files

``` text
train.csv
test.csv
stores.csv
oil.csv
holidays_events.csv
transactions.csv
```

These files are kept under:

``` text
data/raw/
```

The raw files are intentionally kept separate from processed/generated
data.

``` text
data/
├── raw/
│   ├── train.csv
│   ├── test.csv
│   ├── stores.csv
│   ├── oil.csv
│   ├── holidays_events.csv
│   └── transactions.csv
│
└── processed/
    └── train_featured.csv
```

### Important repository decision

The raw dataset and trained model are intentionally **not committed to
GitHub** because they are large.

The repository contains placeholder `.gitkeep` files so that the
required folders remain visible.

After cloning the project, the user should download the dataset and
place the required CSV files inside:

``` text
data/raw/
```

The trained model is generated locally and stored under:

``` text
models/
```

This keeps the GitHub repository lightweight while still preserving the
complete project architecture.

------------------------------------------------------------------------

## Dataset Components

### `train.csv`

The primary historical sales dataset.

Important columns include:

``` text
id
date
store_nbr
family
sales
onpromotion
```

The main prediction target is:

``` text
sales
```

The data contains sales observations for different product families
across multiple stores over time.

------------------------------------------------------------------------

### `test.csv`

Contains future-period rows for which sales are not provided.

Important columns include:

``` text
id
date
store_nbr
family
onpromotion
```

The test structure represents the type of future observations that a
forecasting system would ultimately need to predict.

------------------------------------------------------------------------

### `stores.csv`

Contains store-level information.

Important fields include:

``` text
store_nbr
city
state
type
cluster
```

These variables provide additional information about the store
associated with each sales observation.

------------------------------------------------------------------------

### `oil.csv`

Contains daily oil-price information:

``` text
date
dcoilwtico
```

Oil price is treated as an external time-dependent variable.

------------------------------------------------------------------------

### `holidays_events.csv`

Contains holiday/event information.

Important fields include:

``` text
date
type
locale
locale_name
description
transferred
```

Holiday information is transformed into features that can help the model
capture demand changes around special dates.

------------------------------------------------------------------------

### `transactions.csv`

Contains store-level transaction information:

``` text
date
store_nbr
transactions
```

Transactions provide an additional operational signal related to
customer activity.

------------------------------------------------------------------------

## Target Variable

The target variable is:

``` text
sales
```

It represents the sales/demand value for a particular:

``` text
date
+
store
+
product family
```

The forecasting problem can therefore be represented as:

``` text
Historical information
        ↓
Store + Product Family + Date
        ↓
Previous Sales
        +
Calendar Features
        +
Promotions
        +
Transactions
        +
Oil
        +
Holiday Information
        ↓
Predicted Sales
```

------------------------------------------------------------------------

## Why This Is a Time-Series Problem

This project cannot be treated like a normal tabular regression problem.

In a normal machine learning problem, random train/test splitting can
often be appropriate.

For forecasting, random splitting can cause **future information to
appear in the training set while older observations appear in the test
set**.

That does not represent the real forecasting scenario.

The real-world direction is:

``` text
Past
 ↓
Present
 ↓
Future
```

Therefore the project uses chronological splitting.

``` text
Earlier Period
      ↓
Training

Recent Period
      ↓
Validation

Latest Unseen Period
      ↓
Test
```

No random train/test split is used for the forecasting evaluation.

------------------------------------------------------------------------

## Data Validation

Before feature engineering and modeling, the project validates the raw
inputs.

Validation includes:

-   Required file existence
-   Required column existence
-   Data type inspection
-   Date parsing
-   Missing-value inspection
-   Duplicate checks
-   Negative sales checks
-   Negative transaction checks
-   Store ID consistency
-   Date continuity checks
-   Basic sanity checks

The objective is to detect data-quality problems before they enter the
modeling pipeline.

The validation stage is implemented separately from model training so
that data-quality logic can be reused.

------------------------------------------------------------------------

## Exploratory Data Analysis

The EDA is designed around forecasting behavior rather than only
generating generic charts.

Important analysis includes:

### Sales distribution

Understanding the distribution of the target:

``` text
sales
```

This helps identify:

-   Skewness
-   High-volume product families
-   Low-volume product families
-   Potential extreme values

------------------------------------------------------------------------

### Daily sales trend

Daily sales are aggregated and visualized over time.

This helps identify:

-   Long-term trend
-   Sudden demand changes
-   Unusual periods
-   Potential structural changes

------------------------------------------------------------------------

### Monthly behavior

Sales are aggregated by month to understand:

-   Monthly seasonality
-   Growth/decline
-   Repeating demand patterns

------------------------------------------------------------------------

### Day-of-week behavior

The project examines demand by:

``` text
Monday
Tuesday
Wednesday
Thursday
Friday
Saturday
Sunday
```

This helps identify weekly seasonality.

------------------------------------------------------------------------

### Product family behavior

Demand is analyzed across different:

``` text
family
```

values.

Different product categories can have completely different demand
patterns.

------------------------------------------------------------------------

### Store behavior

The project also examines demand at store level.

This is important because:

``` text
Store A demand ≠ Store B demand
```

A model therefore needs store information rather than treating every
observation as identical.

------------------------------------------------------------------------

### Promotion behavior

The `onpromotion` feature is investigated to understand how promotional
activity relates to demand.

------------------------------------------------------------------------

### Transactions

Transactions are examined as an operational signal that can help explain
demand changes.

------------------------------------------------------------------------

### Oil and holiday variables

External variables such as:

``` text
dcoilwtico
holiday_count
is_holiday
```

are examined to determine whether they provide useful information for
forecasting.

------------------------------------------------------------------------

## Time-Series Analysis

A dedicated notebook is used for time-series analysis.

The analysis includes:

-   Date continuity
-   Daily aggregation
-   Monthly trend
-   Day-of-week seasonality
-   Month seasonality
-   Rolling statistics
-   Lag relationships
-   Recent-period analysis
-   Autocorrelation-style lag inspection

The project specifically investigates whether previous demand values are
useful predictors of future demand.

For example:

``` text
sales(t-1)
sales(t-7)
sales(t-14)
sales(t-28)
```

The intuition is:

``` text
Yesterday's demand
       +
Last week's demand
       +
Previous seasonal patterns
       ↓
Future demand
```

------------------------------------------------------------------------

## Data Cleaning

The project performs time-series-aware cleaning rather than blindly
filling every missing value.

Important operations include:

-   Date conversion
-   Sorting by store/product/date
-   Handling missing external variables
-   Handling missing transaction values
-   Handling missing holiday indicators
-   Creating consistent calendar variables
-   Removing rows that do not contain enough historical information for
    required lag features

The raw dataset remains unchanged.

Generated datasets are saved under:

``` text
data/processed/
```

------------------------------------------------------------------------

## Feature Engineering

Feature engineering is one of the most important parts of this project.

The project creates several categories of features.

------------------------------------------------------------------------

### Calendar Features

The date is transformed into:

``` text
year
month
day
day_of_week
day_of_year
week_of_year
quarter
is_weekend
is_month_start
is_month_end
```

These features allow the model to learn recurring calendar patterns.

------------------------------------------------------------------------

### Lag Features

The project creates historical demand features:

``` text
sales_lag_1
sales_lag_7
sales_lag_14
sales_lag_28
```

Conceptually:

``` text
sales_lag_1
    = previous day's sales

sales_lag_7
    = sales from approximately one week earlier

sales_lag_14
    = sales from approximately two weeks earlier

sales_lag_28
    = sales from approximately four weeks earlier
```

These features provide the model with historical demand context.

------------------------------------------------------------------------

## Leakage-Safe Rolling Features

Rolling statistics are created carefully to avoid using the current
target value.

The rolling calculations are based on shifted historical sales.

Examples:

``` text
sales_rolling_mean_7
sales_rolling_std_7

sales_rolling_mean_14
sales_rolling_std_14

sales_rolling_mean_28
sales_rolling_std_28
```

The important principle is:

``` text
Past sales
    ↓
Rolling calculation
    ↓
Current prediction
```

and not:

``` text
Current sales
    ↓
Feature
    ↓
Prediction
```

Using the target value from the same timestamp as an input feature would
create target leakage.

------------------------------------------------------------------------

## External Feature Integration

The project combines several related datasets.

The main sales table is enriched with:

``` text
stores
transactions
oil
holidays_events
```

The store data is joined using store identifiers.

Time-dependent external data such as oil prices and transactions are
aligned by date.

The implementation uses time-aware merging where appropriate so that the
feature dataset represents the information available for the
corresponding time period.

------------------------------------------------------------------------

## Final Feature Dataset

After feature engineering and merging, the processed dataset contains
features such as:

``` text
id
store_nbr
family
onpromotion

year
month
day
day_of_week
day_of_year
week_of_year
quarter
is_weekend
is_month_start
is_month_end

sales_lag_1
sales_lag_7
sales_lag_14
sales_lag_28

sales_rolling_mean_7
sales_rolling_std_7
sales_rolling_mean_14
sales_rolling_std_14
sales_rolling_mean_28
sales_rolling_std_28

city
state
type
cluster

transactions
dcoilwtico

holiday_count
is_holiday

sales
```

The generated feature dataset is stored at:

``` text
data/processed/train_featured.csv
```

------------------------------------------------------------------------

## Model Preparation

The forecasting dataset is separated into:

``` text
Features (X)
Target (y)
```

The target is:

``` text
sales
```

The `date` column is used to establish chronological order and is not
directly passed as a raw datetime object to the model.

Categorical features include variables such as:

``` text
family
city
state
type
```

These are encoded using:

``` text
OrdinalEncoder
```

with unknown categories handled safely.

The preprocessing is kept inside the model pipeline so that the same
transformation is used during inference.

------------------------------------------------------------------------

## Time-Based Train / Validation / Test Split

The project uses a chronological split approximately following:

``` text
70% → Training
15% → Validation
15% → Test
```

Conceptually:

``` text
|---------------------------|--------------|--------------|
          Train                 Validation       Test
          70%                    15%             15%
```

The validation period is used for model comparison and tuning.

The final test period remains unseen until final evaluation.

This better represents a real forecasting workflow.

------------------------------------------------------------------------

## Forecasting Baselines

Before evaluating complex machine learning models, the project
establishes simple baselines.

The baselines include:

### Previous-day baseline

``` text
Prediction = previous day's sales
```

### Previous-week baseline

``` text
Prediction = approximately previous week's sales
```

These baselines answer an important question:

> Is the machine learning model actually learning something useful
> beyond simple historical assumptions?

A sophisticated model should demonstrate meaningful improvement over a
reasonable baseline.

------------------------------------------------------------------------

## Models Compared

The modeling stage compares multiple approaches.

### HistGradientBoosting

``` text
HistGradientBoostingRegressor
```

A tree-based gradient boosting model suitable for large tabular
datasets.

------------------------------------------------------------------------

### Random Forest

``` text
RandomForestRegressor
```

An ensemble of decision trees that can model nonlinear relationships and
interactions.

------------------------------------------------------------------------

### XGBoost

``` text
XGBRegressor
```

An optional gradient boosting model using XGBoost.

The project keeps model comparison modular so that the final system can
select the strongest candidate based on validation performance.

------------------------------------------------------------------------

## Evaluation Metrics

The project evaluates forecasting performance using:

### MAE

``` text
Mean Absolute Error
```

MAE represents the average absolute difference between:

``` text
Actual Sales
vs
Predicted Sales
```

Lower is better.

------------------------------------------------------------------------

### RMSE

``` text
Root Mean Squared Error
```

RMSE gives greater weight to larger errors.

Lower is better.

------------------------------------------------------------------------

### R²

``` text
R² Score
```

Measures how much variation in the target is explained by the model.

Higher is generally better.

------------------------------------------------------------------------

### MAPE

A safe implementation of:

``` text
Mean Absolute Percentage Error
```

is used.

Special care is required because sales can contain zero values, making
ordinary percentage-error calculations unstable.

------------------------------------------------------------------------

## Model Comparison

Validation performance is compared using a table such as:

``` text
Model                    MAE       RMSE       R²
----------------------------------------------------
Previous Day Baseline    ...
Previous Week Baseline   ...
HistGradientBoosting     ...
Random Forest             ...
XGBoost                   ...
```

The primary selection criterion is validation MAE, while other metrics
are also examined.

The objective is not to choose a model simply because it is more
complex.

The objective is to select the model that provides the strongest
forecasting performance on unseen validation periods.

------------------------------------------------------------------------

## Final Model

For the current project version, the selected forecasting pipeline is
based on:

``` text
RandomForestRegressor
```

The trained artifact is stored as:

``` text
models/best_forecasting_model.joblib
```

The serialized object contains the preprocessing and model pipeline
required for inference.

This means the application does not need to manually reproduce the
training-time preprocessing logic.

------------------------------------------------------------------------

## Hyperparameter Tuning

Model tuning is performed after the initial model comparison.

The tuning workflow is:

``` text
Initial Model Comparison
        ↓
Best Candidate
        ↓
Hyperparameter Search
        ↓
Validation Comparison
        ↓
Train + Validation
        ↓
Final Test Evaluation
```

For the Random Forest candidate, parameters such as the following can be
evaluated:

``` text
n_estimators
max_depth
min_samples_leaf
max_features
```

The tuned model is only used as the final model when it improves the
validation objective.

This prevents tuning from becoming an unnecessary step that
automatically replaces a stronger existing model.

------------------------------------------------------------------------

## Error Analysis

After final evaluation, the project examines prediction errors rather
than looking only at a single metric.

Error analysis can be performed across:

-   Store
-   Product family
-   Time period
-   High-demand observations
-   Low-demand observations
-   Promotional periods
-   Weekend/weekday periods

A useful forecasting model should not only have a good overall score but
should also behave reasonably across different business segments.

------------------------------------------------------------------------

## Model Artifacts

The project produces artifacts such as:

``` text
artifacts/
├── model_comparison_validation.csv
├── final_test_metrics.csv
└── test_predictions.csv
```

The model itself is saved under:

``` text
models/
└── best_forecasting_model.joblib
```

These files separate:

``` text
Model
vs
Evaluation Results
vs
Prediction Results
```

which makes the project easier to maintain.

------------------------------------------------------------------------

## Project Structure

``` text
Revenue Demand Forecasting/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   │
│   ├── static/
│   │   ├── script.js
│   │   └── style.css
│   │
│   └── templates/
│       └── index.html
│
├── artifacts/
│   ├── model_comparison_validation.csv
│   ├── final_test_metrics.csv
│   └── test_predictions.csv
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── stores.csv
│   │   ├── oil.csv
│   │   ├── holidays_events.csv
│   │   └── transactions.csv
│   │
│   └── processed/
│       └── train_featured.csv
│
├── examples/
│   └── sample_inputs.json
│
├── models/
│   └── best_forecasting_model.joblib
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Time_Series_Analysis.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Modeling.ipynb
│   └── 05_Model_Tuning.ipynb
│
├── src/
│   ├── __init__.py
│   ├── logger.py
│   ├── exception.py
│   ├── utils.py
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── feature_engineering.py
│   │   ├── forecasting.py
│   │   ├── model_trainer.py
│   │   └── model_tuning.py
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── train_pipeline.py
│       ├── forecast_pipeline.py
│       └── tuning_pipeline.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_forecasting.py
│   └── test_pipeline.py
│
├── Dockerfile
├── .dockerignore
├── .env.example
├── .gitignore
├── README.md
├── requirements.txt
└── run.py
```

------------------------------------------------------------------------

## Role of Each Major Folder

### `data/`

Contains input and generated datasets.

``` text
data/raw/
```

contains the original downloaded CSV files.

``` text
data/processed/
```

contains generated feature datasets.

The raw dataset should not be modified directly.

------------------------------------------------------------------------

### `notebooks/`

Contains the exploratory and experimentation workflow.

``` text
01_EDA
        ↓
02_Time_Series_Analysis
        ↓
03_Feature_Engineering
        ↓
04_Modeling
        ↓
05_Model_Tuning
```

The notebooks are designed to explain the reasoning behind the
production pipeline.

------------------------------------------------------------------------

### `src/components/`

Contains reusable production components.

Examples:

``` text
data_ingestion.py
```

Loads the raw datasets.

``` text
data_validation.py
```

Checks whether the data is valid.

``` text
feature_engineering.py
```

Creates forecasting features.

``` text
forecasting.py
```

Contains forecasting metrics and baseline logic.

``` text
model_trainer.py
```

Handles model training, comparison, and evaluation.

``` text
model_tuning.py
```

Handles tuning-related logic.

------------------------------------------------------------------------

### `src/pipeline/`

Connects the individual components into workflows.

``` text
train_pipeline.py
```

represents the main training workflow.

``` text
forecast_pipeline.py
```

handles inference/forecasting using a trained model.

``` text
tuning_pipeline.py
```

handles the tuning workflow.

------------------------------------------------------------------------

### `models/`

Contains serialized trained models.

The main artifact is:

``` text
best_forecasting_model.joblib
```

------------------------------------------------------------------------

### `artifacts/`

Contains outputs generated during training and evaluation.

These are useful for:

-   Comparing models
-   Reviewing metrics
-   Analyzing predictions
-   Reproducing results

------------------------------------------------------------------------

### `app/`

Contains the deployment layer.

``` text
main.py
```

contains the FastAPI application.

``` text
schemas.py
```

defines request/response schemas.

``` text
templates/index.html
```

contains the browser interface.

``` text
static/
```

contains frontend CSS and JavaScript.

------------------------------------------------------------------------

### `tests/`

Contains automated tests for:

-   Data availability
-   Feature dataset
-   Model loading
-   Model prediction
-   Pipeline imports
-   Component imports

------------------------------------------------------------------------

## FastAPI Application

The trained model is integrated into a FastAPI application.

The API loads:

``` text
models/best_forecasting_model.joblib
```

when the application starts.

The application also exposes a browser-based interface.

------------------------------------------------------------------------

## API Endpoints

### Home

``` text
GET /
```

Returns the forecasting web application.

------------------------------------------------------------------------

### Health Check

``` text
GET /health
```

Used to verify:

-   API is running
-   Model is loaded

A healthy response conceptually looks like:

``` json
{
  "status": "healthy",
  "model_loaded": true
}
```

------------------------------------------------------------------------

### Prediction

``` text
POST /predict
```

Accepts the model feature set and returns predicted sales.

Conceptually:

``` json
{
  "predicted_sales": 1234.56
}
```

The actual request schema corresponds to the engineered feature set used
by the trained model.

------------------------------------------------------------------------

### Swagger Documentation

FastAPI automatically provides interactive API documentation.

When running locally:

``` text
http://127.0.0.1:8000/docs
```

This can be used to test the `/predict` endpoint directly without the
frontend.

------------------------------------------------------------------------

## Web Application

The project includes a browser-based frontend so that the user does not
need to manually construct API requests.

The interface provides:

``` text
Sample Input
      ↓
Load Sample
      ↓
Feature Values
      ↓
Predict Demand
      ↓
Predicted Sales
```

The frontend communicates with:

``` text
FastAPI
```

through:

``` text
POST /predict
```

------------------------------------------------------------------------

## Sample Inputs

The project includes predefined examples under:

``` text
examples/sample_inputs.json
```

These examples are intended for quick testing of the application.

They allow a user to:

1.  Open the dashboard.
2.  Select a sample.
3.  Load the sample feature values.
4.  Send the values to the API.
5.  View the predicted demand.

The sample data is for application testing and demonstration; it is not
additional training data.

------------------------------------------------------------------------

## Running Locally

### 1. Clone the repository

``` powershell
git clone https://github.com/aman1165/revenue-demand-forecasting.git
cd revenue-demand-forecasting
```

------------------------------------------------------------------------

### 2. Create a virtual environment

Windows PowerShell:

``` powershell
python -m venv .venv
```

Activate:

``` powershell
.\.venv\Scripts\Activate.ps1
```

------------------------------------------------------------------------

### 3. Install dependencies

``` powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Important Model Compatibility

The serialized model was created using:

``` text
scikit-learn 1.7.2
```

Therefore the project pins:

``` text
scikit-learn==1.7.2
```

This is important because serialized scikit-learn models can produce
compatibility problems when loaded using a substantially different
scikit-learn version.

The project should therefore be run using the provided
environment/dependency versions.

------------------------------------------------------------------------

## 4. Add the Dataset

Download the Store Sales dataset from Kaggle:

``` text
https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data
```

Place these files inside:

``` text
data/raw/
```

Required files:

``` text
train.csv
test.csv
stores.csv
oil.csv
holidays_events.csv
transactions.csv
```

Do not rename the files unless the ingestion configuration/code is also
changed.

------------------------------------------------------------------------

## 5. Prepare the Feature Dataset

The feature engineering workflow creates:

``` text
data/processed/train_featured.csv
```

The feature dataset contains the engineered calendar, lag, rolling,
store, transaction, oil, and holiday features required by the
forecasting model.

------------------------------------------------------------------------

## 6. Train / Prepare the Model

The project contains a production training pipeline under:

``` text
src/pipeline/train_pipeline.py
```

The training workflow conceptually performs:

``` text
Data Ingestion
      ↓
Data Validation
      ↓
Feature Engineering
      ↓
Model Training
      ↓
Model Comparison
      ↓
Evaluation
      ↓
Model Serialization
```

The resulting model is stored as:

``` text
models/best_forecasting_model.joblib
```

------------------------------------------------------------------------

## 7. Run the Application

The easiest project entry point is:

``` powershell
python run.py
```

`run.py` is intended to simplify project startup instead of requiring
the user to remember the complete Uvicorn command.

The application can then be opened in the browser at:

``` text
http://127.0.0.1:8000
```

The dashboard can be used to load sample inputs and request predictions.

------------------------------------------------------------------------

## Manual FastAPI Startup

The API can also be started directly:

``` powershell
uvicorn app.main:app --reload
```

Then open:

``` text
http://127.0.0.1:8000
```

Swagger:

``` text
http://127.0.0.1:8000/docs
```

------------------------------------------------------------------------

## Testing

The project uses Pytest.

Run:

``` powershell
pytest -v
```

The current project test suite contains:

``` text
9 tests
```

The latest local test run completed successfully:

``` text
9 passed
```

The tests cover:

### Data tests

-   Required raw files
-   Feature dataset existence
-   Required feature columns

### Model tests

-   Model file existence
-   Model loading
-   Model prediction

### Pipeline tests

-   Training pipeline imports
-   Forecast pipeline imports
-   Component imports

Testing ensures that the repository structure, model artifact, and core
pipeline components remain functional.

------------------------------------------------------------------------

## Docker

The project is containerized using Docker.

Docker provides a reproducible environment containing:

``` text
Python
+
Required dependencies
+
FastAPI
+
Frontend
+
Trained model
```

------------------------------------------------------------------------

## Dockerfile

The application image uses:

``` text
python:3.13-slim
```

The runtime image installs the project requirements and copies the
application and model.

The container starts FastAPI using:

``` text
uvicorn app.main:app
```

------------------------------------------------------------------------

## Build Docker Image

From the project root:

``` powershell
docker build -t revenue-demand-forecasting .
```

------------------------------------------------------------------------

## Run Docker Container

``` powershell
docker run --rm -p 8000:8000 revenue-demand-forecasting
```

Then open:

``` text
http://localhost:8000
```

Swagger:

``` text
http://localhost:8000/docs
```

------------------------------------------------------------------------

## Docker Architecture

The Docker workflow is:

``` text
Project Files
     ↓
Docker Build
     ↓
Python Base Image
     ↓
Install Dependencies
     ↓
Copy FastAPI Application
     ↓
Copy Trained Model
     ↓
Expose Port 8000
     ↓
Start Uvicorn
     ↓
Browser / API Client
```

------------------------------------------------------------------------

## Important Docker Note

The GitHub repository intentionally does not contain the large raw
dataset or the large trained model artifact.

Therefore a fresh clone is not expected to immediately contain:

``` text
data/raw/*.csv
models/best_forecasting_model.joblib
```

The required data must be downloaded separately, and the model must be
prepared locally before building a Docker image that expects the trained
model.

This separation is intentional:

``` text
GitHub
   ↓
Code + Notebooks + Configuration + Tests
   ↓
User downloads dataset
   ↓
User prepares/trains model
   ↓
Model artifact
   ↓
Docker image
```

This keeps the source repository manageable while preserving the
complete reproducible workflow.

------------------------------------------------------------------------

## Deployment Architecture

The current application is deployment-ready at the application/container
level.

The architecture is:

``` text
User
  ↓
Browser
  ↓
FastAPI
  ↓
Validation / Request Schema
  ↓
Saved ML Pipeline
  ↓
Prediction
  ↓
JSON Response
  ↓
Frontend
```

With Docker:

``` text
User
  ↓
Internet / Local Network
  ↓
Docker Container
  ↓
FastAPI
  ↓
ML Model
  ↓
Prediction
```

The project is therefore separated into:

``` text
Data Science Layer
        +
ML Layer
        +
Application Layer
        +
Container Layer
```

Cloud deployment is intentionally treated as a separate learning phase
rather than being mixed into the core project.

------------------------------------------------------------------------

## Why Time-Based Validation Matters

One of the most important lessons from this project is that forecasting
requires respecting time.

A random split could create a situation such as:

``` text
Training:
2025-05
2025-07
2025-09
2025-11

Test:
2025-06
2025-08
2025-10
```

That would allow the model to learn from future periods before
predicting earlier periods.

The project instead follows:

``` text
Past
 ↓
Train

Recent Future
 ↓
Validation

Latest Future
 ↓
Test
```

This is much closer to the real-world forecasting workflow.

------------------------------------------------------------------------

## Why Lag Features Matter

Demand often has temporal dependency.

For example:

``` text
Today's demand
        ↑
Yesterday's demand
        +
Last week's demand
        +
Recent rolling average
```

The model therefore receives historical context through:

``` text
sales_lag_1
sales_lag_7
sales_lag_14
sales_lag_28
```

Without these features, the model would have much less information about
recent demand behavior.

------------------------------------------------------------------------

## Why Rolling Features Matter

A single previous-day value can be noisy.

Rolling statistics provide a broader view:

``` text
7-day mean
7-day standard deviation

14-day mean
14-day standard deviation

28-day mean
28-day standard deviation
```

This helps the model understand:

-   Recent average demand
-   Demand volatility
-   Short-term trends
-   Longer-term local behavior

------------------------------------------------------------------------

## Why Baselines Matter

A model should not automatically be considered good just because it has
a reasonable metric.

For forecasting, simple methods such as:

``` text
Previous Day
Previous Week
```

are useful reference points.

If a complex model cannot outperform these baselines, the additional
complexity may not be justified.

------------------------------------------------------------------------

## Why the Project Uses a Pipeline

Instead of manually performing:

``` text
Encode
↓
Transform
↓
Model
```

during every prediction, the project stores the preprocessing and model
together in a pipeline.

Conceptually:

``` text
Raw Prediction Input
        ↓
Preprocessing
        ↓
Encoding
        ↓
Model
        ↓
Prediction
```

The serialized artifact can therefore be reused by the API.

This reduces the risk of training-time and inference-time preprocessing
becoming inconsistent.

------------------------------------------------------------------------

## Key Learnings

This project covers an end-to-end forecasting workflow.

### Data Science

-   Data validation
-   Exploratory Data Analysis
-   Time-series analysis
-   Data cleaning
-   Calendar feature engineering
-   Lag features
-   Rolling features
-   External feature integration
-   Chronological splitting
-   Baseline forecasting
-   Model comparison
-   Hyperparameter tuning
-   Error analysis

### Machine Learning

-   Regression
-   Random Forest
-   Gradient Boosting
-   XGBoost
-   Feature preprocessing
-   Model selection
-   Evaluation metrics
-   Model serialization

### Time-Series Concepts

-   Temporal ordering
-   Trend
-   Seasonality
-   Autocorrelation
-   Lag variables
-   Rolling statistics
-   Forecasting baselines
-   Future-period evaluation
-   Data leakage prevention

### Software Engineering

-   Modular source code
-   Configuration
-   Logging
-   Exception handling
-   Reusable pipelines
-   Model serialization
-   Automated tests
-   Project structure

### Deployment

-   FastAPI
-   Browser-based frontend
-   Docker
-   Containerized inference
-   Health checks
-   API documentation
-   Reproducible runtime environment

------------------------------------------------------------------------

## Why This Project Is More Than a Notebook

The project moves from:

``` text
Raw Data
   ↓
EDA
   ↓
Time-Series Analysis
   ↓
Feature Engineering
   ↓
Modeling
   ↓
Tuning
   ↓
Evaluation
   ↓
Saved Model
   ↓
FastAPI
   ↓
Web Interface
   ↓
Docker
   ↓
Deployment-ready Application
```

The notebooks explain the experimentation and reasoning.

The `src/` directory contains reusable production-oriented logic.

The `app/` directory contains the inference/application layer.

The Dockerfile packages the runtime environment.

The tests provide basic automated validation.

This separation makes the project closer to a real ML application rather
than a single notebook containing all code.

------------------------------------------------------------------------

## Future Improvements

Potential future improvements include:

-   Cloud deployment
-   AWS deployment
-   HTTPS
-   Authentication
-   Model monitoring
-   Data drift monitoring
-   Experiment tracking
-   CI/CD
-   Automated retraining
-   More advanced time-series models
-   Dedicated forecasting dashboards
-   Probabilistic forecasting
-   Forecast intervals
-   Multi-step recursive forecasting
-   Model monitoring and alerting

These improvements are intentionally separated from the current core
project so that the main architecture remains understandable.

------------------------------------------------------------------------

## Tech Stack

``` text
Python
Pandas
NumPy
Scikit-learn
XGBoost
Matplotlib
Seaborn
Joblib
FastAPI
Pydantic
Uvicorn
Pytest
Docker
Jupyter
```

------------------------------------------------------------------------

## Author

**Aman Soni**

Data Science / Machine Learning Portfolio Project

GitHub:

`https://github.com/aman1165/revenue-demand-forecasting`

------------------------------------------------------------------------

## Project Status

``` text
Dataset Setup             ✅
Data Validation           ✅
EDA                       ✅
Time-Series Analysis      ✅
Feature Engineering       ✅
Baseline Forecasting      ✅
Model Comparison          ✅
Model Evaluation          ✅
Hyperparameter Tuning     ✅
Final Model               ✅
Error Analysis            ✅
Model Serialization       ✅
FastAPI                   ✅
Web UI                    ✅
Sample Inputs             ✅
Automated Testing         ✅
Docker                    ✅
Docker Runtime Testing    ✅
GitHub                    ✅
Cloud Deployment          ⏳
```

------------------------------------------------------------------------

## Complete Project Structure

``` text
Revenue Demand Forecasting/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   │
│   ├── static/
│   │   ├── script.js
│   │   └── style.css
│   │
│   └── templates/
│       └── index.html
│
├── artifacts/
│   ├── model_comparison_validation.csv
│   ├── final_test_metrics.csv
│   └── test_predictions.csv
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── raw/
│   │   ├── train.csv
│   │   ├── test.csv
│   │   ├── stores.csv
│   │   ├── oil.csv
│   │   ├── holidays_events.csv
│   │   └── transactions.csv
│   │
│   └── processed/
│       └── train_featured.csv
│
├── examples/
│   └── sample_inputs.json
│
├── models/
│   └── best_forecasting_model.joblib
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Time_Series_Analysis.ipynb
│   ├── 03_Feature_Engineering.ipynb
│   ├── 04_Modeling.ipynb
│   └── 05_Model_Tuning.ipynb
│
├── src/
│   ├── __init__.py
│   ├── logger.py
│   ├── exception.py
│   ├── utils.py
│   │
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── feature_engineering.py
│   │   ├── forecasting.py
│   │   ├── model_trainer.py
│   │   └── model_tuning.py
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── train_pipeline.py
│       ├── forecast_pipeline.py
│       └── tuning_pipeline.py
│
├── tests/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_forecasting.py
│   └── test_pipeline.py
│
├── .dockerignore
├── .env.example
├── .gitignore
├── Dockerfile
├── README.md
├── requirements.txt
└── run.py
```
