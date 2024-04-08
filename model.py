import joblib
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.metrics import log_loss, hinge_loss

class SVM():
    def __init__(self, load, model_path):
        if load:
            self.model = self.load_parameters(None, load_previous)
        elif model_path is not None:
            self.model = self.load_parameters(model_path, False)
        else:
            self.model = svm.NuSVC(gamma="auto", kernel='rbf', probability=True, class_weight={0 : 0.6})

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
            joblib.dump(params,"models/svm_parameters.pkl")
        except:
            print('Error occured when saving model')

    # loads presaved model
    def load_parameters(self, path, prev):
        try:
            if (path is not None) and (not prev):
                params = joblib.load(path)
            else:
                params = joblib.load("models/svm_parameters.pkl")
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
