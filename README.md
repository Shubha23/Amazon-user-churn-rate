**********************************
Project Description -
**********************************
Dataset - 
Prediction of user convergence based on past user subscription behaviour.
The dependent variable have binary value, 1 for converged user and 0 for not or true/false form.
There are several independent variables describing user's past subscription behaviours.
******************************************************************************************************************

Methodology -
Machine learning classification algorithms such as Random Forest and Logistic Regression are used for predicition.
Algorithms are run using Python's bulit-in package Scikit-learn. 
Data preprocessing is performed for transforming string to int.
Model is then trained by splitting input data with 80 percent train size and 20 percent test size.
Learned model is run on test data to generate predictions.
Accuracy is used a metric for performance evaluation, calculated by comparing actual class to produced predictions.
*******************************************************************************************************************

To execute - 
python predictor.py
***********************************

* Data was in a Csv file format. For other formats use other read function of pandas.
* Update the file path to local directory before running the file.

******************************************** END OF FILE ***********************************************************
