# regime-detection

* Adverse regime forecasting using leading macroeconomic indicators ((1)output and income, (2)labor market, (3)housing, (4)consumption, orders and inventories, (5)money and credit, (6)interest and exchange rates, (7)prices, (8)the stock market), and machine learning algorithms, including common regression algorithms and Hidden Markov Models. The main focus here is to identify periods of economic expansion and contraction, and exploit them accordingly.

* After intitial Exploratory Data Analysis, feature selection is done because the high number of features ~ 129.

* The following steps are taken to clean the data and make it ready for feature selection process.

  1.Remove the variables with missing observations
  
  2.Add lags of the variables as additional features
  
  3.Test stationarity of time series
  
  4.Standardize the dataset
  
* We have two binary outcomes that we want to classify with certain variables. Here we will summarize our approach to predict recessions with machine learning algorithms.

  *We will perform feature selection before making our forecasts. We will use  𝐿1  regularized logistic regression for that purpose.

  *Separate dataset into training and validation datasets. Split based dataset based on time: the period over 1960-1996 is selected for training and the period over 1996-2018 is kept for validation

  *Evaluate performances of the machine learning algorithms on training dataset with cross validation (CV). Since we have time series structure we will use a special type of CV function in Python,TimeSeriesSplit. We will use Receiver operating characteristic (ROC) as scoring metric in our models. Related Python functions for this metric are roc_auc_score and roc_curve.

  *Select the best performing models based on average accuracy and standard deviation of the CV results. We will take logistic regression as a benchmark model since this is the traditional method has been used to approach this problem.

  *Then we make predictions on the validation dataset with selected models. First, we use GridSearchCV for selected model on training dataset to find best combination of parameters for the model. Then we evaluate the model on validation dataset and report accuracy metrics and feature importance results.
