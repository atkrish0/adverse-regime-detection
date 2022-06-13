# regime-detection

### Dataset: 
Macroeconomic dataset from FRED St. Louis desiged by McCracken and Ng (2015).
Involves 129 macroeconomic monthly time series data from 1959 to 2018.
8 Categories: Output and income, labor maket, housing, consumption, orders and inventories, money and credit, interest and exchange rate, prices in the stock market.

### NBER Recession Dates:
• Labelling based on NBER dataset
• 8 recession periods during the time period in consideration
• 628 'normal' periods and '93' recession periods

### Data Cleaning:
• Removal of variables with missing observation/ imputation of some sort
• Add lags of all variables as additional features
• Test stationarity of the time series
• Standardize the dataset

### Add lags of the variables as additional features:
• Add 3, 6, 9, 12, 18 month lags for each variable
• Shift labels for 1 month ahead prediction
• 699 observation points and 710 features

### Stationarity:
• Augmented Dickey Fuller Test. Null hypothesis of ADFis that the time series is non stationary with the alternative that it is stationary
• If p value > significance level, we cannot reject null hypothesis. Then take first order difference
• adfuller function from statsmodels is used

### Standardization:
• Standardization of feature vectors by removing mean and scaling to unit variance
• StandardScaler from scikit-learn is used

### Methodology:
• Perform feature selection to get the most important variables for the forecasts
• Separate dataset into training and validation datasets. 1960 - 1996: Training, 1996 - 2018: Validation
• Evaluate the performance of ML Algos on training set with Cross Validation
• Select the best performing models based on average accuracy and std dev of the CV results. Logistic Regression chosen as benchmark
• Make predictions on the validation dataset with selection models. Use GridSearchCV to find the best combination of hyperparameters. Evaluate the validation modela nd report accuracy metrics.

### Cross Validation:
• K Fold CV used:
	○ Train the model on (k-1) folds of the training data
	○ The resulting model is validated on the remaining part of the data
• 'TimeSeriesSplit' is CV technique for time series data. Use first k sets as training, (k+1) as test set

### Evaluation Metric: 
* ROC AUC Score
