# 🚜 Heavy Machinery AI Price Predictor
**A Capstone Machine Learning Project by Group 2**

👤 **Obil Nathaniel** (271048001)  
👤 **Muhammad Abdullah** (281134982)  

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-FF4B4B?logo=streamlit)
![LightGBM](https://img.shields.io/badge/Engine-LightGBM-brightgreen)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange?logo=scikit-learn)

---

## 📌 Executive Summary
Asset depreciation in heavy machinery is highly non-linear, compounded by operational hours, brand prestige, drivetrain configurations, and hydraulic capabilities. Standard linear statistical models fundamentally fail to capture this multi-dimensional variance. 

This capstone project engineers a robust machine learning pipeline to predict the auction clearing price of heavy equipment. By processing over **400,000 historical auction records**, we evaluated five distinct algorithmic approaches, ultimately proving that Gradient Boosting Ensembles vastly outperform traditional statistical baselines.

## 📊 The Dataset
The data is sourced from the famous [Kaggle Blue Book for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers) competition. It contains over 60 physical and temporal features for machines sold at auction.
* **Target Variable:** `SalePrice` (Log-transformed for metric evaluation)
* **Key Engineered Features:** Machine Age, Cyclical Auction Months, Imputed Operating Hours, and Categorical Encodings (Brand, Size, Enclosure).

## 🏆 Model Architecture & Leaderboard
We conducted extensive hyperparameter optimization across five distinct architectures. Tree-based ensembles achieved the highest variance explanation ($R^2$), proving the non-linear nature of equipment wear and tear.

| Rank | Algorithm | R-Squared ($R^2$) | RMSE | Notes |
|:---:|:---|:---:|:---:|:---|
| 🥇 | **XGBoost** | 0.9179 | 0.1989 | Peak accuracy via sequential error correction. |
| 🥈 | **LightGBM** | 0.9085 | 0.2100 | **Selected for Production.** Blazing fast inference via histogram binning. |
| 🥉 | **Random Forest** | 0.9110 | 0.2071 | Strong baseline ensemble, but computationally heavy. |
| 4 | **K-Nearest Neighbors** | 0.7595 | 0.3412 | Struggled with the curse of dimensionality. |
| 5 | **Linear Regression** | 0.1119 | 0.6545 | Failed to converge; cannot model non-linear decay. |

## 🚀 The Enterprise Dashboard
To operationalize our model, we engineered a 5-page **Streamlit** web application that transforms raw predictions into an interactive Data Product.

**Dashboard Features:**
1. **Exploratory Data Analysis (EDA):** Visualizes the core market distributions.
2. **Explainable AI (XAI):** Unpacks the "Black Box" of our models using Feature Importance (Tree Splits) and Permutation Importance, proving exactly *why* the AI prices a machine the way it does.
3. **Live AI Inference Engine:** An interactive heuristic simulator allowing users to input an 8-dimensional feature vector (Brand, Size, Age, Hours, Cab type, Drive system) to generate a live valuation and a 20-year depreciation trajectory.

## 💻 Local Setup & Installation
To run this project and the live dashboard locally:

**1. Clone the repository:**
```bash
git clone [https://github.com/YourUsername/Bulldozer-AI-Price-Predictor.git](https://github.com/YourUsername/Bulldozer-AI-Price-Predictor.git)
cd Bulldozer-AI-Price-Predictor
