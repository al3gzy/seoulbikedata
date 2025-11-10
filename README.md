# Seoul Bike Data Analysis and Modeling

This repository contains Octave/MATLAB code for data analysis and predictive modeling on the **Seoul Bike Sharing Dataset**. The project implements various machine learning techniques including classification and regression, along with feature selection and evaluation through cross-validation.

---

## Project Overview

The code explores different approaches to predict bike rental counts based on weather and calendar data using:

- **Classification tasks** with K-Nearest Neighbors (KNN), Linear Regression (for classification), Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA) and Logistic Regression.
- **Regression tasks** using Linear Regression, Ridge Regression, Lasso Regression and KNN Regression.
- **Feature selection** through backward stepwise regression.
- **K-Fold cross-validation** for model performance evaluation.

---

## Dataset

The data used in this project is from the **Seoul Bike Sharing Demand** dataset, publicly available at the UCI Machine Learning Repository:

> Seoul Bike Sharing Demand [Dataset]. (2020). UCI Machine Learning Repository.  
> https://doi.org/10.24432/C5F62R

This dataset contains hourly and daily rental bike counts along with weather and calendar information collected in Seoul, South Korea.

---

## Results Summary

### 1) Classification Results

| Model                  | Average Error |
|------------------------|---------------|
| K-Nearest Neighbors     | 0.5884        |
| Linear Classification   | 0.5684        |
| Linear Discriminant Analysis (LDA) | 0.5598 |
| Quadratic Discriminant Analysis (QDA) | 0.7985 |

---

### 2) Regression Results

- **Optimal columns (from backward stepwise selection):** 1, 2, 3, 6, 7, 8, 5  

| Model              | Mean Squared Error (MSE) |
|--------------------|--------------------------|
| Linear Regression  | 220,353.40               |
| K-Fold Cross-validation (avg MSE) | 267,520.84  |
| Ridge Regression   | 202,642.98               |
| Lasso Regression   | 202,628.03               |
| KNN Regression (K=7) | 115,465.46             |

---

### 3) Binary Classification Results (`bin_klasif`)

| Model                  | Average Error |
|------------------------|---------------|
| K-Fold Linear          | 0.1473        |
| K-Nearest Neighbors    | 0.1503        |
| Logistic Regression    | 0.1453        |
