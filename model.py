import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.externals import joblib


class SVM():

    # Trains model
    def train(self, X, Y):
        if X.shape[-1] == Y.shape[0]:
            clf = svm.NuSVC(gamma="auto")  
            clf.fit(X_train, Y_train)
            return clf
        return None

    # Tests model
    def test(self, clf, X):
        Y = clf.predict(X)
        return Y

    def loss(self):
        pass

    def evaluate(self):
        pass

    # Function to save the parameters of the model
    def save_parameters(self, params):
        try:
            joblib.dump(params,"/models/svm_parameters.pkl")
        except:
            print('Error occured when saving model')

    # loads presaved model
    def load_parameters(self):
        try:
            params = joblib.load("/models/svm_parameters.pkl")
            return params
        except:
            print('Error loading saved model. Train first')
            quit()

    def load_data(self):
        pass

    # Saves the predicted results to a new file
    def save_results(self, X, Y):
        pass

    # Saves the training vector to a csv file
    def save_train_vector(self, train_data):
        try:
            df = pd.DataFrame.from_dict(train_Data)
            df.to_csv('train_vector.csv', index=True)
        except:
            print('Error occured when saving data')
