import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
def get_illustration_4_NaN(old_df, new_df=pd.DataFrame()):
    """
      Illustrating the Nan distribution of features,
      if have new dataset,comparing modified before and after
    """
    old_size_name = 'NaN Size' if new_df.empty else 'Pre NaN Size'
    old_proportion_name = 'NaN %' if new_df.empty else 'Pre NaN %'

    email_df = old_df['email_address'].apply(lambda x: x == 'NaN')
    old_df = old_df.drop('email_address', axis=1)
    temp_old_df = old_df.apply(lambda x: x.isnull())
    temp_old_df['email_address'] = email_df

    NaN_info = pd.DataFrame({'Total Size': temp_old_df.count(),
                             old_size_name: temp_old_df.sum()},
                            columns=['Total Size', old_size_name])
    NaN_info[old_proportion_name] = (NaN_info[old_size_name] / NaN_info['Total Size']) * 100

    if new_df.empty:
        temp_poi = temp_old_df.loc[old_df[old_df.poi == True].index.values]
        NaN_info['POIs Size'] = len(temp_poi)
        NaN_info['POIs NaN Size'] = temp_poi.sum()
        NaN_info['POIs NaN %'] = (NaN_info['POIs NaN Size'] / NaN_info['POIs Size']) * 100
    else:
        email_df = new_df['email_address'].apply(lambda x: x == 'NaN')
        new_df = new_df.drop('email_address', axis=1)
        temp_new_df = new_df.apply(lambda x: x.isnull())
        temp_new_df['email_address'] = email_df
        NaN_info['After NaN Size'] = temp_new_df.sum()
        NaN_info['After NaN %'] = (NaN_info['After NaN Size'] / NaN_info['Total Size']) * 100

        NaN_info['Reduce %'] = (NaN_info[old_proportion_name] - NaN_info['After NaN %'])
        NaN_info['Reduce %'] = NaN_info['Reduce %'].apply(lambda x: x if x > 0 else '')

    return NaN_info

def set_missing_val_by_formula(df, addend_features, total_feature, nan_features):
    """
    Since total_payments = salary + bonus + long_term_incentive + deferral_payments + loan_advances + other
      and total_stock_value = exercised_stock_options + restricted_stock + restricted_stock_deferred
    if just missing one value of value,we can get this value by the others.
    """

    for nan_feature in nan_features:
        NaN_df = pd.DataFrame()

        full_features = addend_features + [total_feature]
        NaN_df = df[full_features][(np.isnan(df[nan_feature]))]

        if nan_feature in full_features:
            NaN_df[nan_feature] = 0

            # 1.Screen out all relevant featurea having value except nan feature
            for val in full_features:
                if val != nan_feature:
                    NaN_df = NaN_df[(~np.isnan(NaN_df[val]))]

            for add_val in addend_features:
                if add_val != nan_feature:
                    NaN_df[nan_feature] += NaN_df[add_val]

            if nan_feature != total_feature:
                NaN_df[nan_feature] = NaN_df[total_feature] - NaN_df[nan_feature]

        df.loc[NaN_df.index.values, nan_feature] = NaN_df[nan_feature]

    return df

def set_missing_val_by_RF(df,corr_features,missing_features):
    """
      Using RandomForest algorithm to fill missing data
    """
    from sklearn.ensemble import RandomForestRegressor

    for m_v in missing_features:

        if m_v in corr_features:
            idx_m_v = corr_features.index(m_v)

            # move missing value positon to first
            if idx_m_v != 0:
                corr_features = corr_features[idx_m_v:] + corr_features[:idx_m_v]

            # print corr_features
            corr_df = df[corr_features]

            # print idx_m_v
            knonwn_df = corr_df
            unknonwn_df = corr_df

            for c_v in corr_features:
                knonwn_df = knonwn_df[~np.isnan(knonwn_df[c_v])]

                if c_v == m_v:
                    unknonwn_df = unknonwn_df[np.isnan(unknonwn_df[c_v])]
                else:
                    unknonwn_df = unknonwn_df[~np.isnan(unknonwn_df[c_v])]


            if not unknonwn_df.empty:
                knonwn_matrix = knonwn_df.as_matrix()
                unknonwn_matrix = unknonwn_df.as_matrix()

                y = knonwn_matrix[:,idx_m_v]
                # print y

                x = knonwn_matrix[:,idx_m_v+1:]
                # print '------------'
                # print x

                rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
                rfr.fit(x,y)

                prd = rfr.predict(unknonwn_matrix[:,idx_m_v+1::])
                df.loc[unknonwn_df.index.values,m_v] = prd

    return df

def featureFormat_to_target_Feature(df,
                                    features,
                                    target,
                                    remove_NaN=True,
                                    remove_all_zeroes=True,
                                    remove_any_zeroes=False,
                                    sort_keys = False):
    """ convert Dataframe to numpy array of features and list of target
        remove_NaN = True will convert "NaN" string to 0.0
        remove_all_zeroes = True will omit any data points for which all the features you seek are 0.0
        remove_any_zeroes = True will omit any data points for which any of the features you seek are 0.0
        sort_keys = True True sorts keys by alphabetical order. Setting the value as a string opens
                    the corresponding pickle file with a preset key order
        return list of target, numpy array of features
    """
    full_features = [target]+features
    temp_df = df[full_features]

    if sort_keys:
        temp_df = temp_df.sort_index(ascending=True)

    if remove_all_zeroes:
        temp_df = temp_df.dropna(how='all')

    if remove_any_zeroes:
        temp_df = temp_df.dropna()

    if remove_NaN:
        temp_df = temp_df.fillna(0)

    temp_matrix = temp_df.as_matrix()

    return temp_matrix[:,:1][:,0],temp_matrix[:,1:]




def test_classifier(clf, cv, features, labels):

    """
    compute a number of evaluation metrics by the formulas
    :param clf: the algorithm
    :param cv: based on the small size of the dataset,usig stratified shuffle split cross validation.
    :param features:
    :param labels:
    :return: accuracy, precision, recall, f1, f2
    """
    true_negatives = 0
    false_negatives = 0
    true_positives = 0
    false_positives = 0
    for train_idx, test_idx in cv:
        features_train = []
        features_test = []
        labels_train = []
        labels_test = []
        for ii in train_idx:
            features_train.append(features[ii])
            labels_train.append(labels[ii])
        for jj in test_idx:
            features_test.append(features[jj])
            labels_test.append(labels[jj])

        ### fit the classifier using training set, and test on test set
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)

        for prediction, truth in zip(predictions, labels_test):
            if prediction == 0 and truth == 0:
                true_negatives += 1
            elif prediction == 0 and truth == 1:
                false_negatives += 1
            elif prediction == 1 and truth == 0:
                false_positives += 1
            elif prediction == 1 and truth == 1:
                true_positives += 1
            else:
                print("Warning: Found a predicted label not == 0 or 1.")
                print("All predictions should take value 0 or 1.")
                print("Evaluating performance for processed predictions:")
                break
    try:
        total_predictions = true_negatives + false_negatives + false_positives + true_positives
        accuracy = 1.0 * (true_positives + true_negatives) / total_predictions
        precision = 1.0 * true_positives / (true_positives + false_positives)
        recall = 1.0 * true_positives / (true_positives + false_negatives)
        f1 = 2.0 * true_positives / (2 * true_positives + false_positives + false_negatives)
        f2 = (1 + 2.0 * 2.0) * precision * recall / (4 * precision + recall)
        return accuracy, precision, recall, f1, f2

    except:
        print("Got a divide by zero when trying out:", clf)
        print("Precision or recall may be undefined due to a lack of true positive predicitons.")


def evaluate_clf(clf_list, features, labels, folds=1000):
    """
      get a bunch of evaluation metrics of classifiers  by the formulas
      :param clf: the list of algorithms
      :param features:
      :param labels:
      :return: a bunch of evaluation metrics of classifiers such as accuracy, precision, recall, f1, f2
      """

    from sklearn.cross_validation import StratifiedShuffleSplit

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC,SVC
    from sklearn.neighbors import KNeighborsClassifier

    cv = StratifiedShuffleSplit(labels, folds, random_state=42)

    accuracy_list = []
    precision_list = []
    recall_list = []
    F1_list = []
    clf_name = []

    for clf in clf_list:

        if isinstance(clf, GaussianNB):
            clf_name.append('Naive Bayes')
        elif isinstance(clf, SVC):
            clf_name.append('SVM')
        elif isinstance(clf, DecisionTreeClassifier):
            clf_name.append('Decision Tree')
        elif isinstance(clf, LogisticRegression):
            clf_name.append('Logistic Regression')
        elif isinstance(clf, RandomForestClassifier):
            clf_name.append('Random Forest')
        elif isinstance(clf, LinearSVC):
            clf_name.append('LinearSVC')
        elif isinstance(clf, KNeighborsClassifier):
            clf_name.append('KNN')
        else:
            clf_name.append('Other')

        accuracy, precision, recall, f1, f2 = test_classifier(clf, cv, features, labels)

        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        F1_list.append(f1)

    return pd.DataFrame({'accuracy': accuracy_list,
                         'precision_score': precision_list,
                         'recall_score': recall_list,
                         'F1': F1_list},
                        index=clf_name,
                        columns=['accuracy', 'precision_score', 'recall_score', 'F1'])


def tune_params(grid_search, features, labels, params, iters=80):
    """ given a grid_search and parameters list (if exist) for a specific model,
        along with features and labels list,
        it tunes the algorithm using grid search and prints out the average evaluation metrics
        results (accuracy, percision, recall) after performing the tuning for iter times,
        and the best hyperparameters for the model
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = []
    pre = []
    recall = []
    F1 = []
    for i in range(iters):
        features_train, features_test, labels_train, labels_test = \
            train_test_split(features, labels, test_size=0.3, random_state=i)
        grid_search.fit(features_train, labels_train)
        predicts = grid_search.predict(features_test)

        acc = acc + [accuracy_score(labels_test, predicts)]
        pre = pre + [precision_score(labels_test, predicts)]
        recall = recall + [recall_score(labels_test, predicts)]
        F1 = F1 + [f1_score(labels_test, predicts)]
    print("accuracy: {}".format(np.mean(acc)))
    print("precision: {}".format(np.mean(pre)))
    print("recall:    {}".format(np.mean(recall)))
    print("F1:        {}".format(np.mean(F1)))

    best_params = grid_search.best_estimator_.get_params()
    for param_name in params.keys():
        print("%s = %r, " % (param_name, best_params[param_name]))


def selectKbest(features_list, target_label, df, k):
    """
      List the evaluation values(such as accuracy,precision,recall,F1) with GaussianNB
      when K is greater than or equal to k by selected feature
    """
    from sklearn.feature_selection import SelectKBest, f_classif

    k_list, accuracy_list, precision_list, recall_list, F1_list, test_fetures_list \
        = [], [], [], [], [], []

    target, features = featureFormat_to_target_Feature(df, features_list, target_label)

    # 1. Using sklean.selection.SelectKBest to find out the top of k feature list
    best_features = SelectKBest(f_classif).fit(features, target)
    scores = best_features.scores_
    features_rank = list(reversed(sorted(zip(features_list, scores), key=lambda x: x[1])))
    features_list = list(pd.DataFrame(features_rank[:k])[0])

    # 2.List the evaluation values(such as accuracy,precision,recall,F1)
    #   when K is greater than or equal to k by selected feature

    for k in range(0, k):
        k = k + 1

        t_test_fetures_list = features_list[:k]

        target1, features1 = featureFormat_to_target_Feature(df, t_test_fetures_list, target_label)
        result_df = evaluate_clf([GaussianNB()], features1, target1)

        k_list.append(k)

        accuracy_list.append(result_df['accuracy'][0])
        precision_list.append(result_df['precision_score'][0])
        recall_list.append(result_df['recall_score'][0])
        F1_list.append(result_df['F1'][0])
        test_fetures_list.append(t_test_fetures_list)

    result = pd.DataFrame({'k': k_list, \
                           'accuracy': accuracy_list, \
                           'precision': precision_list, \
                           'recall': recall_list, \
                           'F1': F1_list, \
                           'feature list': test_fetures_list}, \
                          columns=['k', 'accuracy', 'precision', 'recall', 'F1', 'feature list'])

    return result


