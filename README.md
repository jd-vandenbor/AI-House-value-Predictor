# AI-House-value-Predictor
This programs uses Machine Learning algorithms to predict the house value of districts in California. 
The Data Set is pulled from https://github.com/ageron/handson-ml, and the following steps were taken to create the final Product.

1.	Frame the problem : it is a supervised regression algorithm with batch learning. 

2.	Select a performance measure. We will be using the Root Mean square. 

3.	Downloaded the data set ans split it into two straitfied sets for training and test set

4.	Cleaned up the data.
-	Fill in missing values with the median
-	Converted text features to numbers features.
-	Implemented feature scaling.
-	Created a pipeline that encapsulates all data cleaning

6.	Chose a model
- Tested three models with cross validation (linear regression, decision tree, and random forrest)
- Random forest had the best results

7.	Fine tuned the model by grid searching the best hyperameters, and of course cross validating the dataset

The final MSE = 48310.63053212081
given that the data set ranges in the hundreds of thousands this is an acceptable level, although room for imporvement does exist.
