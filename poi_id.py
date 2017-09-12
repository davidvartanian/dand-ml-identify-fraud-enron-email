#!/usr/bin/python

import sys
import pickle

from udacity import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.feature_selection import SelectKBest
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import pandas as pd


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi', 'salary_scaled', 'bonus_scaled', 'director_fees_scaled', 
                 'other_scaled', 'exercised_stock_options_scaled', 'expenses_scaled', 
                 'email_addresses_per_poi_scaled', 'poi_mention_rate']

### Load the dictionary containing the dataset
print 'Loading dataset...'
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
data_dict.pop('TOTAL', 0)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
#my_dataset = data_dict
my_df = pd.DataFrame(data_dict.values(), columns=data_dict[data_dict.keys()[0]].keys(), index=data_dict.keys())

print 'Adding new features...'
# Features: 
# - from_poi_to_this_person_ratio
# - from_this_person_to_poi
from feature_engineering import compute_poi_email_ratio

my_df['from_poi_to_this_person_ratio'] = my_df.apply(lambda row: compute_poi_email_ratio(
    row['from_poi_to_this_person'], row['to_messages']), axis=1)
my_df['from_this_person_to_poi_ratio'] = my_df.apply(lambda row: compute_poi_email_ratio(
    row['from_this_person_to_poi'], row['from_messages']), axis=1)


### Task 3: Create new feature(s)
# Features: 
# - email_addresses_per_poi
# - poi_mention_rate
from feature_engineering import poi_email_dict, find_pois_in_data_point, poi_vectorizer, compute_email_addresses_per_poi, compute_poi_mention_rate

vectorizer = poi_vectorizer(poi_email_dict)
n_pois = len(my_df.index)

def compute_email_addresses_per_poi_df(row):
    found_pois, poi_count = find_pois_in_data_point(row, vectorizer, poi_email_dict)
    return compute_email_addresses_per_poi(found_pois, poi_count)

def compute_poi_mention_rate_df(row):
    _, poi_count = find_pois_in_data_point(row, vectorizer, poi_email_dict)
    return compute_poi_mention_rate(poi_count, n_pois)

my_df['email_addresses_per_poi'] = my_df.apply(lambda row: compute_email_addresses_per_poi_df(row), axis=1)
my_df['poi_mention_rate'] = my_df.apply(lambda row: compute_poi_mention_rate_df(row), axis=1)


# Feature scaling
print 'Scaling features...'
from feature_engineering import scale_feature_df

my_df['salary_scaled'] = scale_feature_df(my_df, 'salary')
my_df['bonus_scaled'] = scale_feature_df(my_df, 'bonus')
my_df['director_fees_scaled'] = scale_feature_df(my_df, 'director_fees')
my_df['other_scaled'] = scale_feature_df(my_df, 'other')
my_df['exercised_stock_options_scaled'] = scale_feature_df(my_df, 'exercised_stock_options')
my_df['expenses_scaled'] = scale_feature_df(my_df, 'expenses')
my_df['email_addresses_per_poi_scaled'] = scale_feature_df(my_df, 'email_addresses_per_poi')

# convert data frame back to dict, so that I can use featureFormat and targetFeatureSplit functions
my_dataset = my_df.to_dict(orient='index')

print 'Reformating data...'
### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
steps = [
    ('feature_selection', SelectKBest(k=2)), 
    ('reduce_dim', PCA(random_state=42, n_components=2)), 
    ('clf', KNeighborsClassifier(algorithm='auto', leaf_size=4, n_neighbors=4, p=2, weights='distance'))
]
clf = Pipeline(steps)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

PERF_FORMAT_STRING = "\
\tAccuracy: {:>0.{display_precision}f}\tPrecision: {:>0.{display_precision}f}\t\
Recall: {:>0.{display_precision}f}\tF1: {:>0.{display_precision}f}\tF2: {:>0.{display_precision}f}"
RESULTS_FORMAT_STRING = "\tTotal predictions: {:4d}\tTrue positives: {:4d}\tFalse positives: {:4d}\
\tFalse negatives: {:4d}\tTrue negatives: {:4d}"

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)
print 'Training classifier...'
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
cv = StratifiedShuffleSplit(labels, 1000, random_state = 42)
true_negatives = 0
false_negatives = 0
true_positives = 0
false_positives = 0
for train_idx, test_idx in cv:
    features_train = []
    features_test  = []
    labels_train   = []
    labels_test    = []
    for ii in train_idx:
        features_train.append( features[ii] )
        labels_train.append( labels[ii] )
    for jj in test_idx:
        features_test.append( features[jj] )
        labels_test.append( labels[jj] )
    clf.fit(features_train, labels_train)
    predictions = clf.predict(features_test)
    print predictions
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
            print "Warning: Found a predicted label not == 0 or 1."
            print "All predictions should take value 0 or 1."
            print "Evaluating performance for processed predictions:"
            break

try:
    print 'Evaluating...'
    total_predictions = true_negatives + false_negatives + false_positives + true_positives
    accuracy = 1.0*(true_positives + true_negatives)/total_predictions
    precision = 1.0*true_positives/(true_positives+false_positives)
    recall = 1.0*true_positives/(true_positives+false_negatives)
    f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
    f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
    print clf
    print PERF_FORMAT_STRING.format(accuracy, precision, recall, f1, f2, display_precision = 5)
    print RESULTS_FORMAT_STRING.format(total_predictions, true_positives, false_positives, false_negatives, true_negatives)
    print ""
except Exception as e:
    print 'Exception thrown while computing training/prediction results'
    print e

### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

print 'Dumping classifier, dataset and features list...'
dump_classifier_and_data(clf, my_dataset, features_list)

print 'Done.'