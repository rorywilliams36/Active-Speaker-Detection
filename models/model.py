import joblib
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import log_loss, hinge_loss, PrecisionRecallDisplay, classification_report

from scipy.stats import expon

class SVM():
    def __init__(self, load, model_path):
        # If training skips to last clause, for testing a pre-saved model is used
        if load:
            self.model = self.load_parameters(None)
        elif model_path is not None:
            self.model = self.load_parameters(model_path)
        else:
            nus = [0.38, 0.39,0.3975, 0.399, 0.42,0.45]
            gammas = [0.02925, 0.0295, 0.035, 0.04, 0.05, 'auto']
            # Gets random values for gamma and nu based on the exponential distribution in the range (0-1)
            # param_grid = {'gamma' : expon(scale=.1), 'nu' : expon(scale=.1), 'kernel' : ['rbf'], 
            # 'class_weight' : [{0:0.7, 1:0.3}, {0:0.75, 1:0.25}],'probability' : [True]}
            # self.model = RandomizedSearchCV(svm.NuSVC(), param_distributions=param_grid, n_iter=50, refit=True, verbose=3)

            param_grid = {'gamma' : gammas, 'nu' : nus, 'kernel' : ['rbf'], 
            'class_weight' : [{0:0.7, 1:0.3}], 'probability' : [True]}
            self.model = GridSearchCV(svm.NuSVC(), param_grid, refit=True, verbose=3)

            # ### Best found:
            # self.model = svm.NuSVC(gamma=0.035, nu=0.3975, class_weight = {0:0.7, 1:0.3}, probability=True)

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

    def loss(self, X, y):
        pred_decision = self.model.decision_function(X)
        hinge = hinge_loss(y, pred_decision)

        # pred_probs = self.model.predict_proba(X)
        # lg_loss = log_loss(y, pred_probs)
        # print(f'Log: {lg_loss}')
        return hinge

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
