import numpy as np
from sklearn.linear_model import LinearRegression
from visualisation import scatter_plot


class OutlierCleaner:
    
    def __init__(self, features, labels):
        self.reg = None
        features = np.reshape(features, (len(features), 1))
        labels = np.reshape(labels, (len(labels), 1))
        self.set_data(features, labels)
        
    def set_data(self, features, labels):
        """
        Set data and split it into train/test data sets
        """
        self.features = features
        self.labels = labels
        #self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(
        #    self.features, self.labels, test_size=0.1, random_state=42)
        
    def fit_regressor(self):
        reg = LinearRegression()
        reg.fit(self.features, self.labels)
        return reg
    
    def clean(self, visualise=False):
        """
        Clean 10% or data points with worst residual error
        Keep cleaned training data set for further cleaning

        Return
            list: errors
        """
        reg = self.fit_regressor()
        pred = reg.predict(self.features)
        features, labels, errors, predictions = self.clean_outliers(
            predictions=pred, features=self.features, labels=self.labels)
        self.set_data(features, labels)
        if visualise:
            self.visualise(features, labels, predictions)
        return self.get_data()
    
    def get_data(self):
        """
        Get features and labels (use after calling clean method)
        """
        return self.features, self.labels
    
    def visualise(self, features, labels, predictions):
        scatter_plot(X=features, Y=labels, xplot=features, yplot=predictions)
    
    def clean_outliers(self, predictions, features, labels):
        """
            Clean away the 10% of points that have the largest
            residual errors (difference between the prediction
            and the actual net worth).

            Return a list of tuples named cleaned_data where 
            each tuple is of the form (age, net_worth, error).
        """
        cleaned_data = []
        data = [(f, l, (p - l)**2, p) for p, f, l in zip(predictions, features, labels)]
        data.sort(key=lambda x: x[2], reverse=True)
        for i, d in enumerate(data):
            if i >= len(data) / 10:
                cleaned_data.append(d)
        
        features, labels, errors, predictions = zip(*cleaned_data)
        features = np.reshape(features, (len(features), 1))
        labels = np.reshape(labels, (len(labels), 1))
        return features, labels, errors, predictions