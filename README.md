# Time Series Forecasting with XGBoost

This project demonstrates how to forecast time series data using XGBoost, a popular machine learning algorithm. The dataset used is hourly energy consumption data, and the notebook includes steps for data preparation, visualization, feature engineering, and model training.

## Project Structure

1. **Data Loading**  
   Load the dataset and examine its structure.

2. **Data Preparation**  
   Set the index to `Datetime` and convert it to a DateTime format.

3. **Visualization**  
   Plot the time series data and visualize different aspects of the data.

4. **Train/Test Split**  
   Split the data into training and test sets based on a specified date.

5. **Feature Engineering**  
   Create additional time-based features from the index.

6. **Model Training**  
   Train an XGBoost Regressor on the training data and evaluate its performance on the test data.

7. **Feature Importance**  
   Analyze the importance of different features used in the model.

8. **Prediction and Evaluation**  
   Make predictions on the test set and evaluate the model's performance using Root Mean Squared Error (RMSE).


## Requirements

- pandas
- numpy
- matplotlib
- seaborn
- xgboost
- scikit-learn

You can install the required packages using pip:

```bash
pip install pandas numpy matplotlib seaborn xgboost scikit-learn
```

## Notes

- Ensure that the dataset path in the script is correctly specified.
- This notebook assumes that the dataset is in CSV format and contains a `Datetime` column which is used as the index.

