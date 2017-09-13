# Udacity Data Analyst Nanodegree
## Machine Learning - Identify Fraud from Enron Email

## Abstract
The scripts asume that *maildir* and *emails_by_address* (from Udacity's ud120-projects repository) directories are in the root of the project (which is not included). You can create a symbolic link to the actual path:
```
$ cd dand-ml-identify-fraud-enron-email
$ ln -s /actual/path/to/maildir .
$ ln -s /actual/path/to/ud120-projects/final_project/emails\_by\_address .
```

## Development Process & Documentation
I've been working on a [Jupyter Notebook](Enron%20Fraud%20Lab.ipynb). It includes the whole process of data exploration, feature selection, feature engineering, classifier testing, parameter tunning, validation and evaluation, as well as the answers of questions related (underneath).

For the HTML version please go [here](Enron+Fraud+Lab.html).

## Structure
I used the following files to organise the code:

* visualisation.py (plotting functions)
* outlier.py (outlier review/removal logic, not used to complete the project)
* feature_engineering.py (functions and POI data, functions with prefix *compute* are used to create new features)
* udacity.py (code provided by Udacity in the original [project](https://github.com/udacity/ud120-projects))