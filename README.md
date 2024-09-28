# Analyzing-Dogecoin-and-Shiba-Inu-Prices-with-LSTM-Models
This project focuses on predicting the prices of Dogecoin and Shiba Inu Coin based on historical data. It compares the performance of LSTM models used for both cryptocurrencies and identifies key factors influencing their prices, such as trading volume and market capitalization.
# Cryptocurrency Price Prediction: Dogecoin and Shiba Inu Coin

## Objectives
- Predict the prices of Dogecoin and Shiba Inu Coin based on historical data.
- Compare the performance of the Linear Regression model used for both coins.
- Identify key factors that significantly affect prices, such as trading volume and market capitalization.

## Model Performance Comparison

### Linear Regression Model
#### Dogecoin
- **Mean Squared Error (MSE):** 0.00405
- **R-squared:** 0.2305

#### Shiba Inu
- **Mean Squared Error (MSE):** 7.858892e-11
- **R-squared:** 0.3499

### LSTM (Long Short-Term Memory) Model
#### Dogecoin
- **Mean Squared Error (MSE):** 0.0018
- **R-squared:** 0.6200

#### Shiba Inu
- **Mean Squared Error (MSE):** 4.5e-11
- **R-squared:** 0.6000

## Advantages and Disadvantages

### Linear Regression
**Advantages:**  
- Simple and interpretable model, providing insights when the relationships between variables are linear.

**Disadvantages:**  
- Lower performance as indicated by R-squared and higher MSE values, struggling to capture the complexities of the data.

### LSTM
**Advantages:**  
- Best performance with the lowest MSE and highest R-squared for both Dogecoin and Shiba Inu, excelling at capturing complex and temporal patterns in the data, particularly in time series predictions.

**Disadvantages:**  
- More complex in implementation, requiring more time for training and additional computational resources.
