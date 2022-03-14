# EV_affect_on_emissions

Modelling the current and predicting the future affect of electric vehicles on greenhouse gas emissions using Long-Short Term Memory (LSMT) models and Non Linear Regressions (NLR).

Code

* The LSTM model is in LSTM_model.ipynb
* The Non Linear Regression is in Non_Linear_regression_correlation_study.ipynb

Data

* EU_em
* UK_em
* US_em
* EU_veh
* UK_veh
* US_veh
* Air pollution data

Outputs

* Model performance - a record of the performance of LSTMs
* NLR model outputs - the correlation and error of the NLR

Figures

* Comparison of model with EVs and without pred_EV_or_not
* Combined model predictions EU, UK, US
* Loss of the US US_loss
* NLR plots

Models

* EU_model is the EU model
* EU_UK_model is trained on the EU then the UK data
* RNN_em is a model used for hyperparameter tuning
