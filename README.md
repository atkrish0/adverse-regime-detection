# regime-detection

* Adverse regime forecasting using leading macroeconomic indicators ((1)output and income, (2)labor market, (3)housing, (4)consumption, orders and inventories, (5)money and credit, (6)interest and exchange rates, (7)prices, (8)the stock market), and machine learning algorithms, including common regression algorithms and Hidden Markov Models. The main focus here is to identify periods of economic expansion and contraction, and exploit them accordingly.

* After intitial Exploratory Data Analysis, feature selection is done because the high number of features ~ 129.

* The following steps are taken to clean the data and make it ready for feature selection process.

  1.Remove the variables with missing observations
  
  2.Add lags of the variables as additional features
  
  3.Test stationarity of time series
  
  4.Standardize the dataset
