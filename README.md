# 📈 Revenue Demand Forecasting

An end-to-end Machine Learning and Time-Series Forecasting project for predicting store-level product demand using historical sales data, calendar patterns, store characteristics, promotions, transactions, oil prices, and holiday information.

The project is designed as a production-style Machine Learning application rather than a notebook-only experiment.

It covers the complete workflow:

- Business problem understanding
- Data ingestion
- Data validation
- Exploratory Data Analysis
- Time-series analysis
- Time-aware data cleaning
- Leakage-safe feature engineering
- Lag features
- Rolling statistics
- Calendar features
- Chronological train/validation/test split
- Baseline forecasting
- Machine Learning model comparison
- Hyperparameter tuning
- Final model selection
- Model serialization
- FastAPI inference API
- Interactive web dashboard
- Automated testing
- Docker containerization
- Single-entry-point execution using `run.py`

---

# 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Business Problem](#-business-problem)
3. [Project Objective](#-project-objective)
4. [Why Demand Forecasting](#-why-demand-forecasting)
5. [Dataset](#-dataset)
6. [Dataset Files](#-dataset-files)
7. [Dataset Setup](#-dataset-setup)
8. [Forecasting Strategy](#-forecasting-strategy)
9. [Time-Based Validation](#-time-based-validation)
10. [Feature Engineering](#-feature-engineering)
11. [Machine Learning Approach](#-machine-learning-approach)
12. [Model Evaluation](#-model-evaluation)
13. [Hyperparameter Tuning](#-hyperparameter-tuning)
14. [Final Model](#-final-model)
15. [Project Architecture](#-project-architecture)
16. [End-to-End Workflow](#-end-to-end-workflow)
17. [Application Architecture](#-application-architecture)
18. [Web Dashboard](#-web-dashboard)
19. [FastAPI API](#-fastapi-api)
20. [API Request Features](#-api-request-features)
21. [Automated Runner](#-automated-runner)
22. [Project Configuration](#-project-configuration)
23. [Logging and Exception Handling](#-logging-and-exception-handling)
24. [Testing](#-testing)
25. [Docker](#-docker)
26. [Docker Runtime](#-docker-runtime)
27. [Repository and Large Files](#-repository-and-large-files)
28. [Installation](#-installation)
29. [How to Run](#-how-to-run)
30. [First Run vs Subsequent Runs](#-first-run-vs-subsequent-runs)
31. [Troubleshooting](#-troubleshooting)
32. [Technology Stack](#-technology-stack)
33. [Key Engineering Decisions](#-key-engineering-decisions)
34. [Future Improvements](#-future-improvements)
35. [Project Status](#-project-status)
36. [Project Structure](#-project-structure)
37. [Author](#-author)

---

# 🧠 Project Overview

Revenue Demand Forecasting is a machine learning system that predicts expected store sales demand using historical demand behavior and business-related information.

The project is based on the Kaggle Store Sales - Time Series Forecasting dataset.

Instead of treating the problem as a normal tabular regression task, the project explicitly preserves the chronological structure of the data.

Historical sales are transformed into predictive signals through:

- Lag features
- Rolling means
- Rolling standard deviations
- Calendar features
- Store attributes
- Promotion information
- Transactions
- Oil prices
- Holiday indicators

The resulting feature dataset is used to train and evaluate machine learning forecasting models.

The final trained model is exposed through a FastAPI application and an interactive browser-based dashboard.

---

# 🎯 Business Problem

Businesses need to estimate future demand to make better operational decisions.

Incorrect demand estimates can lead to:

- Overstocking
- Stock shortages
- Poor inventory allocation
- Inefficient promotion planning
- Revenue planning difficulties
- Store-level operational inefficiencies

A demand forecasting system can help estimate future sales based on historical behavior and known business factors.

---

# 🎯 Project Objective

The main objective of this project is:

> Predict store-level product demand using historical sales patterns, calendar information, store information, promotions, transactions, holidays, and external factors.

The final system should allow a user to:

1. Provide forecasting-related feature values.
2. Send those values to the trained model.
3. Receive predicted store sales.
4. View the prediction through a web dashboard.

---

# 💼 Business Use Cases

Potential applications include:

### Inventory Planning

Estimate expected demand to support inventory decisions.

### Promotion Planning

Understand how promotion activity can influence expected demand.

### Store-Level Forecasting

Generate demand estimates for individual stores and product families.

### Revenue Planning

Use predicted sales demand as an input for broader business planning.

### Operational Planning

Use expected demand to improve planning of resources and store operations.

---

# 📊 Dataset

This project uses the:

**Store Sales - Time Series Forecasting**

dataset from Kaggle.

The dataset is associated with:

**Corporación Favorita**

### Official Dataset Source

https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data

---

# 📁 Dataset Files

The forecasting workflow uses the following files:

```text
train.csv
test.csv
stores.csv
oil.csv
holidays_events.csv
transactions.csv