#This is the Machine learning pipeline for generating the confusion matrices, classification report, errors and AUC.



#!/usr/bin/env python
# coding: utf-8

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

cleaned = pd.read_csv('data/cleaned_train_data.csv')
cleaned_test = pd.read_csv('data/cleaned_test_data.csv')
cleaned['C_SEV'].value_counts()


# In[74]:


cleaned['C_SEV'].value_counts()


# In[75]:


# Separate majority and minority classes
majority = cleaned[cleaned.C_SEV==2]
minority = cleaned[cleaned.C_SEV==1]

# downsample majority class
maj_downsampled = resample(majority,
                                 replace=True,     # sample with replacement
                                 n_samples=2559, # to match majority class
                                 random_state=407) # reproducible results

# Combine majority class with upsampled minority class
resampled = pd.concat([minority, maj_downsampled])

# Display new class counts
resampled['C_SEV'].value_counts()


# In[76]:


X_train = resampled.drop(['C_SEV'], axis = 1)
y_train = resampled['C_SEV']
X_test = cleaned_test.drop(['C_SEV'], axis = 1)
y_test = cleaned_test['C_SEV']


# In[77]:


categorical_features = ['C_MNTH', 'C_WDAY', 'C_HOUR', 'C_RCFG', 'C_WTHR', 'C_RSUR', 'C_RALN',
       'C_TRAF', 'V_TYPE']
categorical_transformer = Pipeline(steps=[
                                          ('onehot', OneHotEncoder(handle_unknown='ignore'))
                                         ])

preprocessor = ColumnTransformer(
                                 transformers=[
                                    ('cat', categorical_transformer, categorical_features)
                                ])


# In[78]:


### parameter tuning


# In[79]:


# rf_params = [{
#     'classifier' : [RandomForestClassifier()],
#     'classifier__n_estimators' : list(range(10,101,10)),
#     'classifier__max_features' : list(range(6,32,5))}]

# rf_model = { 'Random Forest': RandomForestClassifier()}


# In[80]:


# log_params = [{'classifier' : [LogisticRegression()],
#     'classifier__penalty' : ['l1', 'l2'],
#     'classifier__C' : np.logspace(-4, 4, 20),
#     'classifier__solver' : ['liblinear']}]

# log_model = { 'Logistic Regression': LogisticRegression()}


# In[81]:


# pipe = Pipeline(steps=[('preprocessor', preprocessor),
#                       ('classifier', LogisticRegression())])
# search = GridSearchCV(pipe, log_params,
#                       n_jobs=-1,
#                       scoring = "accuracy",
#                       cv = 5,
#                       refit = True)
# search.fit(X_train, y_train)
# tr_err = 1 - search.score(X_train, y_train)
# valid_err = 1 - search.score(X_test, y_test)
# results_dict['Logistic Regression'] = [round(tr_err,3), round(valid_err,3), search.best_params_]


# In[82]:


# pipe = Pipeline(steps=[('preprocessor', preprocessor),
#                       ('classifier', RandomForestClassifier())])
# search = GridSearchCV(pipe, rf_params,
#                       n_jobs=-1,
#                       scoring = "accuracy",
#                       cv = 5,
#                       refit = True)
# search.fit(X_train, y_train)
# tr_err = 1 - search.score(X_train, y_train)
# valid_err = 1 - search.score(X_test, y_test)
# results_dict['Random Forest'] = [round(tr_err,3), round(valid_err,3), search.best_params_]


# In[83]:


# pipe = Pipeline(steps=[('preprocessor', preprocessor),
#                       ('classifier', LogisticRegression())])

# search = GridSearchCV(pipe, log_params,
#                       n_jobs=-1,
#                       scoring = "accuracy",
#                       cv = 5,
#                       refit = True)
# search.fit(X_train, y_train)
# tr_err = 1 - search.score(X_train, y_train)
# valid_err = 1 - search.score(X_test, y_test)
# results_dict['Logistic Regression'] = [round(tr_err,3), round(valid_err,3), search.best_params_]


# In[84]:


results_dict = dict()


# In[85]:


# results = pd.DataFrame(results_dict).T
# results.columns = ("Train Score", "Validation Score", "Best Parameters")
# results


# In[86]:


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


# In[87]:


confusion_matrix_rf_train = confusion_matrix(y_train, pipe.predict(X_train))
confusion_matrix_rf_test = confusion_matrix(y_test, pipe.predict(X_test))
report_rf = classification_report(y_train, pipe.predict(X_train), output_dict = True)

fpr, tpr, thresholds = roc_curve(y_test, pipe.predict_proba(X_test)[:,1], pos_label=2)

plt.plot(fpr, tpr);
plt.plot((0,1),(0,1),'--k');
plt.xlabel('false positive rate');
plt.ylabel('true positive rate');
plt.title('AUC Random Forest');
plt.savefig('results/auc_rf.png')
plt.clf()
# In[88]:


pipe = Pipeline(steps=[('preprocessor', preprocessor),
                       ('classifier', LogisticRegression(C=4.281332398719396, class_weight=None, dual=False,
                      fit_intercept=True, intercept_scaling=1, l1_ratio=None,
                      max_iter=100, multi_class='auto', n_jobs=None, penalty='l2',
                      random_state=None, solver='liblinear', tol=0.0001, verbose=0,
                      warm_start=False))])


# In[89]:


log = pipe.fit(X_train, y_train)
tr_err = 1 - log.score(X_train, y_train)
valid_err = 1 - log.score(X_test, y_test)
results_dict['Log'] = [round(tr_err,3), round(valid_err,3)]


# In[90]:


print(results_dict)

pd.DataFrame(results_dict, index = ['Train', 'Test']).to_csv("results/errors.csv")
print("Both models were run and the error file has been generated")

# In[91]:


confusion_matrix_lgr_train = confusion_matrix(y_train, pipe.predict(X_train))
confusion_matrix_lgr_test = confusion_matrix(y_test, pipe.predict(X_test))
report_lr = classification_report(y_train, pipe.predict(X_train), output_dict = True)

print("Generated confusion matrices for test and train")
print("Generated classification report for training data")

fpr, tpr, thresholds = roc_curve(y_test, pipe.predict_proba(X_test)[:,1], pos_label=2)

plt.plot(fpr, tpr);
plt.plot((0,1),(0,1),'--k');
plt.xlabel('false positive rate');
plt.ylabel('true positive rate');
plt.title('AUC Logistic regression');
plt.savefig('results/auc_lgr.png')
plt.clf()
# In[92]:


confusion_matrix_lgr_train


# In[93]:


confusion_matrix_rf_train


# In[94]:


confusion_matrix_lgr_test


# In[95]:


confusion_matrix_rf_test


# In[96]:


report_lr


# In[97]:


print(report_rf)


# In[98]:


df_rf_classification = pd.DataFrame(report_rf)
df_rf_classification.to_csv('results/rf_classification.csv')

df_rf_classification = pd.DataFrame(report_lr)
df_rf_classification.to_csv("results/lgr_classification.csv")


# In[99]:


confusion_matrix_lgr_train_df = pd.DataFrame(confusion_matrix_lgr_train,
                                             index = ['Not fatal', 'Fatal'],
                                             columns=['Not fatal', 'Fatal'])
confusion_matrix_lgr_train_df.to_csv('results/lgr_train_confusion.csv')


# In[100]:


confusion_matrix_lgr_test_df = pd.DataFrame(confusion_matrix_lgr_test,
                                             index = ['Not fatal', 'Fatal'],
                                             columns=['Not fatal', 'Fatal'])
confusion_matrix_lgr_test_df.to_csv('results/lgr_test_confusion.csv')


# In[101]:


confusion_matrix_rf_train_df = pd.DataFrame(confusion_matrix_rf_train,
                                             index = ['Not fatal', 'Fatal'],
                                             columns=['Not fatal', 'Fatal'])
confusion_matrix_rf_train_df.to_csv('results/rf_train_confusion.csv')


# In[102]:


confusion_matrix_rf_test_df = pd.DataFrame(confusion_matrix_rf_test,
                                             index = ['Not fatal', 'Fatal'],
                                             columns=['Not fatal', 'Fatal'])
confusion_matrix_rf_test_df.to_csv('results/rf_test_confusion.csv')


# In[105]:


# from sklearn.metrics import roc_curve

# fpr, tpr, thresholds = roc_curve(y_test, pipe.predict_proba(X_test)[:,1])

# plt.plot(fpr, tpr);
# plt.plot((0,1),(0,1),'--k');
# plt.xlabel('false positive rate');
# plt.ylabel('true positive rate');
# #plt.savefig('../results/auc_rf.png')


# In[ ]:
