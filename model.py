import joblib
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import log_loss, hinge_loss
from scipy.stats import expon

class SVM():
    def __init__(self, load, model_path):
        if load:
            self.model = self.load_parameters(None)
        elif model_path is not None:
            self.model = self.load_parameters(model_path)
        else:
            # self.model = svm.NuSVC(gamma="auto", kernel='rbf', probability=True, class_weight={0 : 0.6})
            param_grid = {'gamma' : expon(scale=0.1), 'nu' : expon(scale=0.1), 'kernel' : ['rbf']}
            # self.model = GridSearchCV(svm.NuSVC(), param_grid, refit=True, verbose=3)
            self.model = RandomizedSearchCV(svm.NuSVC(), param_distributions=param_grid, n_iter=100, refit=True, verbose=3)

    # Trains model
    def train(self, X, Y):
        if X.shape[0] == Y.shape[0]:
            print('Training Starting...')
            self.model.fit(X, Y)
            print('Training Completed')
            return self.model

        print('Error during training. Data constructed incorrectly')
        return None

    # Tests model
    def test(self, X):
        Y = self.model.predict(X)
        return Y

    def loss(self, X, y):
        pred_decision = self.model.decision_function(X)
        hinge = hinge_loss(y, pred_decision)

        pred_probs = self.model.predict_proba(X)
        lg_loss = log_loss(y, pred_probs)
        print(f'Log: {lg_loss}')
        return hinge

    def evaluate(self):
        pass

    # Function to save the parameters of the model
    def save_parameters(self, params):
        try:
            with open('wb', "models/svm_parameters.pkl") as file:
                joblib.dump(params, file)
        except:
            print('Error occured when saving model')

    # loads presaved model
    def load_parameters(self, path):
        try:
            if path is not None:
                with open('rb', path) as file:
                    return joblib.load(file)
            else:
                with open('rb', "models/svm_parameters.pkl") as file:
                    return joblib.load(file)
        except:
            print('Error loading saved model. Train first')
            quit()

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
