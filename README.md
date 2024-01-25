Dataset size: 33,497 datapoints (166 Sole Traders) (0.5%) 11 Defaulted, 2 late, 5 cancelled, 16 repaying,132 repaid
Data below correct up to 31/07/2022 and statistics are updated quarterly as per the FCA requirement. Data is from Lending Works.

10 Features being explored in this dataset:

Day of Start Date
Term (Months)
Day of Maturity Date
Loan Purpose - Personal Loan, Sole Trader
Loan Status - Active, Settled, Cancelled, Defaulted
Repayment Status - Repaying, Repaid, (cancelled), Late, Defaulted
Default Reason - Bankruptcy, Debt Arrangement Scheme (DAS), Debt Management Plan (DMP), Non-Payment, Payment Arrangement, Protected Trust Deed, Partial Settlement
Amount
Gross Rate
Principal Outstanding

(Explain why certain features were left out of the dataset, data cleaning, utf-8 encoding error, outliers, duplicates, normalisation/scaling, the limitations of the dataset provided and how this may affect the final results)
Explorative Data Analysis, Use of Excel

The purpose of this project is to gain meaningful insights and analytics using statistical and supervised machine learning based approaches. This can be done using decision trees, K-nearest neighbours, SVM or neural network based
classifiers. We'll be exploring all of these classifiers to see which provides the highest model accuracy after backtesting. Calculations for the mean, standard deviation (variance) and correlation coefficients can also be made to
further understand the numerical features.

The goals for this project are listed below:

Repayment Status Prediction - Predict the probability of a loan being either repaying, repaid, cancelled, late or defaulted based on the features listed above. (Classification)

Principal Outstanding Prediction: Predict how much of the loan will have been repaid. (Regression)

Results:

K Nearest Neighbours Classification Accuracy: 0.9613432835820895
Decision Tree Classification Accuracy: 0.9765671641791045
Random Forest Classification Accuracy: 0.9865671641791045
Neural Network Classification Accuracy: 0.9728358208955223

Linear Regression RMSE: 1891.5619618620067
Decision Tree Regression RMSE: 792.4701968535371
Random Forest Regression RMSE: 616.4537802669631
Linear Support Vector Regression RMSE: 2587.574176059397
Polynomial Support Vector Regression RMSE: 1861.5411303617402
Neural Network Regression RMSE: 13.5373

References: https://www.lendingworks.co.uk/about-us/statistics
            https://www.doria.fi/bitstream/handle/10024/182846/himberg_tomi.pdf?sequence=2&isAllowed=y
            https://www.analyticsvidhya.com/blog/2022/04/predicting-possible-loan-default-using-machine-learning/
            https://github.com/Nyandwi/machine_learning_complete
