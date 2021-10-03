# Diabetes_prediction
Using various Machine Learning models to determine whether a person is diabetic or not using various features.

Since it is a categorial dataset.So we use various classification supervised machine learning algorithm.

## Dataset description
Pregnancies: Number of times pregnant

Glucose: Plasma glucose concentration over 2 hours in an oral glucose tolerance test

BloodPressure: Diastolic blood pressure (mm Hg)

SkinThickness: Triceps skin fold thickness (mm)

Insulin: 2-Hour serum insulin (mu U/ml)

BMI: Body mass index (weight in kg/(height in m)2)

DiabetesPedigreeFunction: Diabetes pedigree function (a function which scores likelihood of diabetes based on family history)

Age: Age of the person (years)

Outcome: Class variable (0 if non-diabetic, 1 if diabetic)

## K Nearest Neighbors.
The k-nearest neighbors (KNN) algorithm is a simple, easy-to-implement supervised machine learning algorithm that can be used to solve both classification and regression problems. It is also called a lazy learner algorithm because it does not learn from the training set immediately instead it stores the dataset and at the time of classification, it performs an action on the dataset.KNN algorithm at the training phase just stores the dataset and when it gets new data, then it classifies that data into a category that is much similar to the new data.

## Logistic Regression.
Logistic regression is one of the most popular Machine Learning algorithms, which comes under the Supervised Learning technique. It is used for predicting the categorical dependent variable using a given set of independent variables.Logistic regression predicts the output of a categorical dependent variable. Therefore the outcome must be a categorical or discrete value. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1.

## Naive Bayes classifier.
Naïve Bayes algorithm is a supervised learning algorithm, which is based on Bayes theorem and used for solving classification problems.Naïve Bayes Classifier is one of the simple and most effective Classification algorithms which helps in building the fast machine learning models that can make quick predictions.It basically use probability for prediction.

## Support vector Machine.
Support Vector Machine or SVM is one of the most popular Supervised Learning algorithms, which is used for Classification as well as Regression problems. However, primarily, it is used for Classification problems in Machine Learning.The goal of the SVM algorithm is to create the best line or decision boundary that can segregate n-dimensional space into classes so that we can easily put the new data point in the correct category in the future. This best decision boundary is called a hyperplane.

## Decision Tree.
Decision Tree is a Supervised learning technique that can be used for both classification and Regression problems, but mostly it is preferred for solving Classification problems. It is a tree-structured classifier, where internal nodes represent the features of a dataset, branches represent the decision rules and each leaf node represents the outcome.

## Random Forest.
Random Forest is a popular machine learning algorithm that belongs to the supervised learning technique. It can be used for both Classification and Regression problems in ML. Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset. Instead of relying on one decision tree, the random forest takes the prediction from each tree and based on the majority votes of predictions, and it predicts the final output.

# Reference
www.javatpoint.com/

www.wikipedia.com/

## zTraining and Testing Accuracy of various machine learning algorithm to social network dataset.

For Decision tree

  Training Accuracy  -  100.0%
  Testing Accuracy   -  69.35%
  
For Logistic Regression

  Training Accuracy  -  82.02%
  Testing Accuracy   -  73.39%
  
For Naive Bayes

  Training Accuracy  -  79.39%
  Testing Accuracy   -  71.77%
  
For K- Nearest Neighbour 

  Training Accuracy  -  84.44%
  Testing Accuracy   -  73.39%
  
For Support Vector Machine

  Training Accuracy  -  82.22%
  Testing Accuracy   -  73.39%
  
For Random Forest

  Training Accuracy  -  97.98%
  Testing Accuracy   -  70.97%
  
For Kernel Support Vector Machine

  Training Accuracy  -  86.67%
  Testing Accuracy   -  70.97%

From this after removing outliers KNN performs well.
