import joblib
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import log_loss, hinge_loss, PrecisionRecallDisplay, classification_report
from sklearn.pipeline import make_pipeline

from scipy.stats import expon

PATH = 'models/parameter_files'

class SVM():
    def __init__(self, load):
        # If training skips to last clause, for testing a pre-saved model is used
        if load:
            self.model = self.load_parameters()
        else:
            self.model = svm.NuSVC(gamma=0.02925, nu=0.38, probability=True)

    def train(self, X, Y):
        ''' 
        Train model 
        
        Args:
            X: predictions
            Y: Correct Labels
        '''
        # Checks data in correct shape
        if X.shape[0] == Y.shape[0]:
            print('\nTraining Starting...')
            self.model.fit(X, Y)
            print('Training Completed')
            return self.model

        print('Error during training. Data constructed incorrectly')
        quit()

    def test(self, X):
        ''' Test Model '''
        print('Testing')
        Y = self.model.predict(X)
        return Y

    def evaluate(self, pred_y, test_y):
        return classification_report(pred_y, test_y)

    def save_parameters(self, params):
        ''' Saves parameters of a model'''
        try:
            with open(f"{PATH}/svm_parameters.pkl", 'wb') as file:
                joblib.dump(params, file)
                print('Model Saved')
        except FileNotFoundError:
            print('File not found')
        except Exception as e:
            print(f'Error Saving Model: \n{e}')

    def load_parameters(self):
        ''' Loads the parameters of a presaved model '''
        try:
            with open(f"{PATH}/svm_parameters2.pkl", 'rb') as file:
                params = joblib.load(file)
                print('Model Loaded Successfully')
                file.close()
            return params
        except FileNotFoundError:
            print('File not found')
        except Exception as e:
            print(f'Error loading saved model. Train first or check path: \n{e}')

    def save_train_vector(self, train_data):
        ''' Saves the training vector to a csv file '''
        try:
            df = pd.DataFrame.from_dict(train_Data)
            with open('train_vector.csv', 'wb') as file:
                df.to_csv(file, index=True)
                file.close()
            print('Training Vector Saved')
        except FileNotFoundError:
            print('File not found')
        except Exception as e:
            print(f'Error Saving Vector: \n{e}')
