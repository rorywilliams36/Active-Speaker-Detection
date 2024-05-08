import joblib
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import log_loss, hinge_loss, PrecisionRecallDisplay, classification_report
from sklearn.pipeline import make_pipeline

from scipy.stats import expon

class SVM():
    def __init__(self, load):
        # If training skips to last clause, for testing a pre-saved model is used
        if load:
            self.model = self.load_parameters(None)
        else:
            self.model = svm.NuSVC(gamma=0.02925, nu=0.38, probability=True)

    # Trains model
    def train(self, X, Y):
        # Checks data in correct shape
        if X.shape[0] == Y.shape[0]:
            print('\nTraining Starting...')
            self.model.fit(X, Y)
            print('Training Completed')
            return self.model

        print('Error during training. Data constructed incorrectly')
        quit()

    # Tests model
    def test(self, X):
        print('Testing')
        Y = self.model.predict(X)
        return Y

    def evaluate(self, pred_y, test_y):
        return classification_report(pred_y, test_y)

    # Function to save the parameters of the model
    def save_parameters(self, params):
        try:
            with open("svm_parameters.pkl", 'wb') as file:
                joblib.dump(params, file)
                print('Model Saved')
        except:
            print('Error occured when saving model')

    # loads presaved model
    def load_parameters(self, path):
        try:
            if path is not None:
                with open(path, 'rb') as file:
                    params = joblib.load(file)
                    print('Model Loaded Successfully')
            else:
                with open("svm_parameters.pkl", 'rb') as file:
                    params = joblib.load(file)
                    print('Model Loaded Successfully')

            return params
        except:
            print('Error loading saved model. Train first or check path')
            quit()

    # Saves the training vector to a csv file
    def save_train_vector(self, train_data):
        try:
            df = pd.DataFrame.from_dict(train_Data)
            with open('train_vector.csv', 'wb') as file:
                df.to_csv(file, index=True)
            print('Training Vector Saved')
        except:
            print('Error occured when saving training data')
