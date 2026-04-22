# Loan Risk Prediction

This project uses machine learning to predict the risk associated with loan applicants based on historical data. It provides scripts for training three classification models, compares their performance, and chooses the best model to predict loan risk - helping financial institutions assess loan applications more efficiently.

## Features

- Data preprocessing and cleaning
- Benchmarks Naive Bayes, KNN, Decision Tree and plots a ROC curve
- Model training and evaluation
- Predicting loan applicant risk
- Easy-to-use scripts for training and prediction

## Project Structure

```
loan_risk_prediction_dataset.csv
train_model.py
predict_applicant.py
loan_risk.py
```

## Getting Started

### Prerequisites

- Python 3.7+
- Required packages (install with pip):

```sh
pip install -r requirements.txt
```
### Compare three models

```sh
python loan_risk.py
```

### Training the Model

```sh
python train_model.py
```

### Making Predictions

```sh
python predict_applicant.py
```

## Dataset

- `loan_risk_prediction_dataset.csv`: Contains historical loan applicant data used for training and testing.

## License

This project is licensed under the MIT License.
