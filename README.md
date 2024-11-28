# Fraud Detection on Synthetic Financial Datasets

This project demonstrates fraud detection using machine learning techniques on a synthetic financial dataset. The goal is to predict whether a transaction is fraudulent or not based on several features like transaction type, amount, and customer information.

## Dataset

The dataset used in this project is the **Synthetic Financial Datasets For Fraud Detection**, available on Kaggle:

[Download Dataset from Kaggle](https://www.kaggle.com/datasets/ealaxi/paysim1)

### Dataset Description:
- **step**: The unit of time in the real world. Each step corresponds to an hour.
- **type**: Transaction type: CASH-IN, CASH-OUT, DEBIT, PAYMENT, and TRANSFER.
- **amount**: Amount of the transaction.
- **nameOrig**: Customer initiating the transaction.
- **oldbalanceOrg**: The customer's balance before the transaction.
- **newbalanceOrig**: The customer's balance after the transaction.
- **nameDest**: The recipient of the transaction.
- **oldbalanceDest**: Recipient’s balance before the transaction.
- **newbalanceDest**: Recipient’s balance after the transaction.
- **isFraud**: Flag indicating if the transaction is fraudulent (1 for fraud, 0 for non-fraud).
- **isFlaggedFraud**: Indicates whether the transaction was flagged due to massive transfers (only values greater than 200,000).

## Approach

### 1. **Data Preprocessing**
- Handling missing values and normalizing the dataset.
- Balancing the dataset using techniques like SMOTE (Synthetic Minority Over-sampling Technique).

### 2. **Models Implemented**
- **Logistic Regression (LR)**
- **Decision Tree (DT)**
- **XGBoost (XGB)**
- **Naive Bayes (NB)**

### 3. **Performance Metrics**
- The models are evaluated using the following metrics:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-Score**
  - **ROC-AUC**
  - **Confusion Matrix**

### 4. **Evaluation and Comparison**
- The models' performance is compared using **bar graphs** for precision, recall, F1-score, and accuracy.
- A **ROC curve** is plotted for each model to assess classification performance.
- **Confusion matrix** visualization for model comparison.

## Installation

Clone this repository and install the necessary dependencies:

```bash
git clone https://github.com/yourusername/Fraud_Detection.git
cd Fraud_Detection
pip install -r requirements.txt
