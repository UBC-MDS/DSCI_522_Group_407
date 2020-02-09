# This is the Machine learning pipeline for generating the confusion matrices, classification report, errors and AUC.

#  Outputs include the following :
#   1. auc_lgr.png
#   2. auc_rf.png
#   3. lgr_clssification.csv
#   4. lgr_train_confusion.csv
#   5. lgr_test_confusion.csv
#   6. rf_clssification.csv
#   7. rf_train_confusion.csv
#   8. rf_test_confusion.csv

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV
import time
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

# Import cleaned data
cleaned = pd.read_csv('data/cleaned_train_data.csv')
cleaned_test = pd.read_csv('data/cleaned_test_data.csv')

# Separate majority and minority classes
majority = cleaned[cleaned.C_SEV==2]
minority = cleaned[cleaned.C_SEV==1]

# Downsample majority class
maj_downsampled = resample(majority,
                                 replace=True,     # sample with replacement
                                 n_samples=2559, # to match majority class
                                 random_state=407) # reproducible results

# Combine majority class with upsampled minority class
resampled = pd.concat([minority, maj_downsampled])

# Display new class counts
resampled['C_SEV'].value_counts()

X_train = resampled.drop(['C_SEV'], axis = 1)
y_train = resampled['C_SEV']
X_test = cleaned_test.drop(['C_SEV'], axis = 1)
y_test = cleaned_test['C_SEV']

# Preprocesses the categorical features to one-hot-encode them
categorical_features = ['C_MNTH', 'C_WDAY', 'C_HOUR', 'C_RCFG', 'C_WTHR', 'C_RSUR', 'C_RALN',
       'C_TRAF', 'V_TYPE']
categorical_transformer = Pipeline(steps=[
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))
                                         ])

preprocessor = ColumnTransformer(
                                 transformers=[
                                    ('cat', categorical_transformer, categorical_features)
                                ])

# Initialize an empty dictionary
results_dict = dict()

# Assign pipeline for random forest classifier
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', RandomForestClassifier(bootstrap=True, class_weight=None,
                          criterion='gini', max_depth=None, max_features=6,
                          max_leaf_nodes=None,
                          min_impurity_decrease=0.0, min_impurity_split=None,
                          min_samples_leaf=1, min_samples_split=2,
                          min_weight_fraction_leaf=0.0,
                          n_jobs=None, oob_score=False, random_state=None,
                          verbose=0, warm_start=False))])
rf = pipe.fit(X_train, y_train)
tr_err = 1 - rf.score(X_train, y_train)
valid_err = 1 - rf.score(X_test, y_test)
results_dict['Random Forest'] = [round(tr_err,3), round(valid_err,3)]

# Create outputs for confusion matrix of the test, train and classification reports.
confusion_matrix_rf_train = confusion_matrix(y_train, pipe.predict(X_train))
confusion_matrix_rf_test = confusion_matrix(y_test, pipe.predict(X_test))
report_rf = classification_report(y_train, pipe.predict(X_train), output_dict = True)

# Create a plot for the ROC Curve
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, pipe.predict_proba(X_test)[:,1], pos_label=2)


# Pipeline to pass the Logistic regression model
pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', LogisticRegression(C=4.281332398719396, class_weight=None, dual=False,
                      fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                      max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                      random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                      warm_start=False))])

log = pipe.fit(X_train, y_train)
tr_err = 1 - log.score(X_train, y_train)
valid_err = 1 - log.score(X_test, y_test)
results_dict['Log'] = [round(tr_err,3), round(valid_err,3)]

# Save a csv file with the errors.
pd.DataFrame(results_dict, index = ['Train', 'Test']).round(2).to_csv("results/errors.csv")
print("Both models were run and the error file has been generated")

# Create outputs for confusion matrix of the test, train and classification reports.
confusion_matrix_lgr_train = confusion_matrix(y_train, pipe.predict(X_train))
confusion_matrix_lgr_test = confusion_matrix(y_test, pipe.predict(X_test))
report_lr = classification_report(y_train, pipe.predict(X_train), output_dict = True)

print("Generated confusion matrices for test and train")
print("Generated classification report for training data")

# Plot of the output for ROC Curve for logistic regression
log_fpr, log_tpr, log_thresholds = roc_curve(y_test, pipe.predict_proba(X_test)[:,1], pos_label=2)


plt.plot(rf_fpr, rf_tpr, c='g', label='Random Forest')
plt.plot(log_fpr, log_tpr, c='r', label='Logistic Regression')
plt.plot((0,1),(0,1),'--k');
plt.xlabel('false positive rate');
plt.ylabel('true positive rate');
plt.title('AUC');
plt.legend()
plt.savefig('results/auc.png')
plt.clf()

# Save results for random forest classification report
df_rf_classification = pd.DataFrame(report_rf).round(2)
df_rf_classification.to_csv('results/rf_classification.csv')


# Save results for logistic regression classification report
df_rf_classification = pd.DataFrame(report_lr).round(2)
df_rf_classification.to_csv("results/lgr_classification.csv")

# Confusion matrices foe train logistic regression
confusion_matrix_lgr_train_df = pd.DataFrame(confusion_matrix_lgr_train,
                                             index = ['Not fatal', 'Fatal'],
                                             columns=['Not fatal', 'Fatal']).round(0)
confusion_matrix_lgr_train_df.to_csv('results/lgr_train_confusion.csv')

# Confusion matrices for test logistic regression
confusion_matrix_lgr_test_df = pd.DataFrame(confusion_matrix_lgr_test,
                                             index = ['Not fatal', 'Fatal'],
                                             columns=['Not fatal', 'Fatal']).round(0)
confusion_matrix_lgr_test_df.to_csv('results/lgr_test_confusion.csv')


# Confusion matrices for train random forest
confusion_matrix_rf_train_df = pd.DataFrame(confusion_matrix_rf_train,
                                             index = ['Not fatal', 'Fatal'],
                                             columns=['Not fatal', 'Fatal']).round(0)
confusion_matrix_rf_train_df.to_csv('results/rf_train_confusion.csv')


# Confusion matrices for test random forest
confusion_matrix_rf_test_df = pd.DataFrame(confusion_matrix_rf_test,
                                             index = ['Not fatal', 'Fatal'],
                                             columns=['Not fatal', 'Fatal']).round(0)
confusion_matrix_rf_test_df.to_csv('results/rf_test_confusion.csv')
