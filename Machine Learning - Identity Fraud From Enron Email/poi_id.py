#!/usr/bin/python

import sys
import pickle
import pandas as pd
import util
import matplotlib.pyplot as plt
import pprint
import numpy as np
from sklearn.feature_selection import SelectKBest,f_classif
from sklearn.preprocessing import MinMaxScaler
# sys.path.append("../tools/")

# from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
# features_list = ['poi','salary'] # You will need to use more features
### Load the dictionary containing the dataset
# with open("final_project_dataset.pkl", "r") as data_file:
#     data_dict = pickle.load(data_file)

target_label = 'poi'
email_features_list = ['from_messages', 'from_poi_to_this_person', 'from_this_person_to_poi',
                       'to_messages', 'shared_receipt_with_poi']
financial_payment_features_list =['salary', 'bonus', 'long_term_incentive',
                                  'deferral_payments', 'loan_advances', 'other',
                                  'expenses', 'director_fees', 'deferred_income', 'total_payments']
financial_stock_features_list = ['exercised_stock_options','restricted_stock','restricted_stock_deferred','total_stock_value']

'1.1 Extract data as pandas DataFrame to convenient analysis'
data = pd.read_pickle("final_project_dataset.pkl")
# convert to DataFrame type
data_df = pd.DataFrame(data).T
# print('Datasets:',data_df.shape)

"""1.2 Convert features types
Since the data types of all the features are Object,
we should convert numeric in some features which could be the type of numeric for investigating conveniently"""
email_df = data_df['email_address']
data_df = data_df.drop('email_address',axis=1)
data_df = data_df.apply(lambda x: pd.to_numeric(x, errors='coerce'))
data_df['email_address'] = email_df
# print data_df.info()


### Task 2: Remove outliers
'2.1 remove extramly value'
# plt.scatter(data_df['salary'],data_df['bonus'])
# plt.title("Identify Fraud from Enron Email")
# plt.xlabel('Salary')
# plt.ylabel('Bonus')
# plt.show()

## Finding out the outlier which data is
# print data_df[(data_df['salary']==max(data_df['salary'])) & (data_df['bonus']==max(data_df['bonus']))]

## why causes
## Verify whether sum(salary+bonus)/2 = (salary+bonus) of 'TOTAL'
# sum_salary = sum(data_df['salary'][~np.isnan(data_df['salary'])])
# sum_bonus = sum(data_df['bonus'][~np.isnan(data_df['bonus'])])
# print "Is 'TOTAL' the sum of numerical features that everey person in dataset? "
# print (sum_salary+sum_bonus)/2 == (data_df['salary']['TOTAL']+data_df['bonus']['TOTAL'])

# remove outlier
data_df = data_df.drop('TOTAL',0)
# print('After removing outlier Datasets:',data_df.shape)
# plt.scatter(data_df['salary'],data_df['bonus'])
# plt.title("Identify Fraud from Enron Email")
# plt.xlabel('Salary')
# plt.ylabel('Bonus')
# plt.show()
"""2.2 Check person name if corrects
   Person name is formated like last name + space + first name + space + middle name(if exists),
   In other word,it is less than three and greater than one space.
"""

# for x in data_df.index.values:
#     space_num = x.count(' ')
#     if space_num<1 or space_num>3:
#         print(x,space_num)
# 'THE TRAVEL AGENCY IN THE PARK' is not a real name.We should remove it.

data_df = data_df.drop('THE TRAVEL AGENCY IN THE PARK',0)  # remove outlier
# print('After removing outlier Datasets:',data_df.shape)



'2.3 Remove person who has all null of features and is not POI'
# count the null in each person
null_count=data_df.shape[1] - data_df.count(axis=1)
# null_count['LOCKHART EUGENE E']
n_features=len(email_features_list
               +financial_payment_features_list
               +financial_stock_features_list)
all_null_person = data_df[(null_count>n_features-1) & (data_df.poi==False)]
# print all_null_person
data_df = data_df.drop(all_null_person.index,0)  # remove outlier
# print('After removing outlier Datasets:',data_df.shape)


### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
# my_dataset = data_dict
# Creata a new Dataset in order to storing data which be modified
new_data_df = data_df.copy()

'3.1 Categorical Variables Transformed'
# Converting poi feature where True=1 and False=0.
new_data_df['poi'] = new_data_df['poi'].map( {True: 1, False: 0} ).astype(int)


'3.2 Missing values impution'
"""
-3.2.1 Fill in the missing value according to the formula
      If we have one missing value of them we can get it through the other values    
"""
# Fomula: total_payments = salary + bonus + long_term_incentive + deferral_payments + loan_advances + other
new_data_df = util.set_missing_val_by_formula(new_data_df,
                                              financial_payment_features_list[:9],
                                              financial_payment_features_list[9],
                                              financial_payment_features_list)
# Fomula: total_stock_value = exercised_stock_options + restricted_stock + restricted_stock_deferred
new_data_df = util.set_missing_val_by_formula(new_data_df,
                                               financial_stock_features_list[:3],
                                               financial_stock_features_list[3],
                                               financial_stock_features_list)

'-3.2.2 Fill in the missing value via RandomForest algorithm'
new_data_df = util.set_missing_val_by_RF(new_data_df,
                                 financial_stock_features_list[:3],
                                 financial_stock_features_list[:3])
new_data_df = util.set_missing_val_by_formula(new_data_df,
                                               financial_stock_features_list[:3],
                                               financial_stock_features_list[3],
                                               financial_stock_features_list)

'-3.2.3 Remove some features which have highly missing values'
remove_feature_list =['deferral_payments','director_fees','loan_advances']
new_data_df = new_data_df.drop(remove_feature_list, axis=1)  # remove outlier
for r_v in remove_feature_list:
    financial_payment_features_list.remove(r_v)

# print new_data_df.shape

'3.3 Create new feature combining existing features'
new_features_list = ['fraction_from_poi', 'fraction_to_poi']
new_data_df['fraction_from_poi'] = new_data_df['from_poi_to_this_person'] / new_data_df['from_messages']
new_data_df['fraction_to_poi'] = new_data_df['from_this_person_to_poi'] / new_data_df['to_messages']

### Extract features and labels from dataset for local testing
# data = featureFormat(my_dataset, features_list, sort_keys = True)
# labels, features = targetFeatureSplit(data)

'3.4 Extract the features specified in features_list'
features_list_without_new = financial_payment_features_list + financial_stock_features_list + email_features_list

target_without_new, features_without_new =  \
    util.featureFormat_to_target_Feature(new_data_df, features_list_without_new, target_label, sort_keys=True)

# full features_list
features_list = features_list_without_new + new_features_list
target, features = util.featureFormat_to_target_Feature(new_data_df, features_list, target_label, sort_keys=True)

'3.5 Univariate Feature Selection'
features_select = util.selectKbest(features_list,target_label,new_data_df,k=10)
pprint.pprint(features_select)

best_features_5= features_select['feature list'][4]
best_target, best_features = util.featureFormat_to_target_Feature(new_data_df, best_features_5, target_label)

'3.6 Scale feature'
scaler = MinMaxScaler()
best_features = scaler.fit_transform(best_features)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
clf_list=[]
from sklearn.naive_bayes import GaussianNB
gnb_clf = GaussianNB()
clf_list.append(gnb_clf)

from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier()
clf_list.append(tree_clf)

from sklearn.ensemble import RandomForestClassifier
rf_clf = RandomForestClassifier()
clf_list.append(rf_clf)

from sklearn.linear_model import LogisticRegression
lr_clf = LogisticRegression()
clf_list.append(lr_clf)

from sklearn.svm import LinearSVC
ls_clf = LinearSVC()
clf_list.append(ls_clf)

from sklearn.neighbors import KNeighborsClassifier
knn_clf = KNeighborsClassifier()
clf_list.append(knn_clf)


print('the scores of best features in each algorithm\n')
print(util.evaluate_clf(clf_list,best_features,best_target))



### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

from sklearn.model_selection import GridSearchCV
# 1. Decision Tree
dt_param = {'criterion':('gini', 'entropy'),
            'splitter':('best','random')}
dt_grid_search = GridSearchCV(estimator = DecisionTreeClassifier(), param_grid = dt_param)

print 'Tuning Decison Tree:\n'
print util.tune_params(dt_grid_search, best_features, best_target, dt_param)


# 2.LinearSVC
svm_param = {'tol': [1, 0.1, 0.01, 0.001, 0.0001],
'C': [0.1, 1, 10, 100, 1000]}
svm_grid_search = GridSearchCV(estimator = LinearSVC(), param_grid = svm_param)
print('LinearSVC:\n')
print(util.tune_params(svm_grid_search, best_features, best_target, svm_param))


# 3.knn
knn_param = {'algorithm':('auto', 'ball_tree', 'kd_tree', 'brute'),
    'n_neighbors': [1,9]}
knn_grid_search = GridSearchCV(estimator = KNeighborsClassifier(), param_grid = knn_param)

print 'Tuning Knn:\n'
print util.tune_params(knn_grid_search, best_features, best_target, knn_param)


'After tuning ,showing the evaluations in each algorithm'

final_clf_list = []
final_clf_list.append(GaussianNB())
final_clf_list.append(DecisionTreeClassifier(splitter = 'best',criterion = 'gini'))
final_clf_list.append(RandomForestClassifier(n_estimators = 8))
final_clf_list.append(LogisticRegression())
final_clf_list.append(LinearSVC(tol = 1, C = 0.1))
final_clf_list.append(KNeighborsClassifier(algorithm = 'auto'))
print(util.evaluate_clf(final_clf_list,best_features, best_target))

# Final I choose Gaussian Naive Byes as my algorithm
clf = gnb_clf

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(best_features, best_target, test_size=0.3, random_state=42)

from sklearn import cross_validation
from sklearn.cross_validation import StratifiedShuffleSplit


for score in ['recall', 'precision']:
    scores = cross_validation.cross_val_score(clf, best_features, best_target,
                                              cv=StratifiedShuffleSplit(best_target, n_iter=500),
                                              scoring=score)
    print score + " : ", np.average(scores)




### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

# Since I used the type of DataFrame in this dataset,
# I have to convert this to dict which can fit the function of parameter my_dataset

my_dataset = (new_data_df.fillna('NaN').T).to_dict(orient='dict')

dump_classifier_and_data(clf, my_dataset, [target_label]+best_features_5)


