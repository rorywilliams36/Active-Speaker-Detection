
import os, argparse, cv2
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

from train import feature_extract, save_results
from dataLoader import Train_Loader, Test_Loader, extract_labels
from asd import ActiveSpeaker
from models.support_vec import SVM
from models.mobilenet import MobileNet
from models.shuffle import ShuffleNet
from models.train_vectors import train_model, train_validation, test_model

from evaluation import roc, svm_roc, conf_matrix
from utils import tools

MODEL_PATH = '/parameter_files'

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = "Active Speaker Detection Program")
    parser.add_argument('--train', action='store_true', help="Perform training")
    parser.add_argument('--validate', action='store_true', help='Train with Cross-Validation')
    parser.add_argument('--test', action='store_true', help="Perform testing")
    parser.add_argument('--confMatrix', action='store_true',  required=False, help="Plot Confusion Matrix from testing")
    parser.add_argument('--roc', action='store_true',  required=False, help="Plot ROC curve from testing")
    parser.add_argument('--trainDataPath', type=str, default='train', required=False, help="Data path for the training dataset")
    parser.add_argument('--testDataPath', type=str, default='test', required=False, help="Data path for the testing dataset")
    parser.add_argument('--saveResults',  action='store_true', required=False, help='Save results from testing')

    parser.add_argument('--SVM', action='store_true', required=False, help='Selects Support Vector Machine to be used as classifer')
    parser.add_argument('--MobileNet', action='store_true', required=False, help='Selects MobileNetV3 Small to be used as classifer')
    parser.add_argument('--ShuffleNet', action='store_true', required=False, help='Selects ShuffleNetV2 to be used as classifer')
    parser.add_argument('--mobileThresh', type=float, default=0.25, required=False, help='Threshold value for MobileNet classification')
    parser.add_argument('--shuffleThresh', type=float, default=0.12, required=False, help='Threshold value for ShuffleNet classification')
    parser.add_argument('--epochs', type=int, default=50, required=False, help='Select the number of epochs to train for (int)')
    parser.add_argument('--lr', type=float, default=0.003, required=False, help='Select the learing rate for training (float)')
    parser.add_argument('--Loss', action='store_true', required=False, help='Plots loss graph')
    parser.add_argument('--valLoss', action='store_true', required=False, help='Plots validation and training loss graph')

    args = parser.parse_args()

    # Training
    if args.train or args.validate:
        # Get features and store them in dictionary
        data = feature_extract(ids=train_ids, root_dir=args.trainDataPath, train=True, svm_check=args.SVM)
        data['Label'] = np.array(data['Label']).flatten().astype(np.int64)
        X_train = np.array(data['Flow'])
        Y_train = data['Label']

        # Train relevant Model
        if args.SVM:
            svm = SVM(False)
            model = svm.train(X_train, Y_train)
            svm.save_parameters(model)        

        if args.MobileNet:
            model = MobileNet()
            mobile_model_file = f'{MODEL_PATH}/mobilenet_model.pth'
            if args.validate:
                pred_probs, train_loss, valid_loss, valid_accuracies = train_validation(data, model, mobile_model_file, args.epochs, args.lr, threshold=args.mobileThresh)
                if args.valLoss:
                    tools.plot_cross_validation(train_loss, valid_loss, args.epochs)
                    tools.plot_valid_acc(valid_accuracies, args.epochs)
            else:
                pred_probs, loss = train_model(data, model, mobile_model_file, args.epochs, args.lr)
            
        if args.ShuffleNet:
            model = ShuffleNet()            
            shuffle_model_file = f'{MODEL_PATH}/shufflenet_model.pth'
            if args.validate:
                pred_probs, train_loss, valid_loss, valid_accuracies = train_validation(data, model, shuffle_model_file, args.epochs, args.lr, threshold=args.shuffleThresh)
                if args.valLoss:
                    tools.plot_cross_validation(train_loss, valid_loss, args.epochs)
                    tools.plot_valid_acc(valid_accuracies, args.epochs)
            else:
                pred_probs, loss = train_model(data, model, shuffle_model_file, args.epochs, args.lr)


        if (args.ShuffleNet or args.MobileNet) and args.Loss:
            tools.plot_loss(loss, args.epochs)


    # Testing
    if args.test:
        # Feature Extraction
        data = feature_extract(ids=test_ids, root_dir=args.testDataPath, train=False, svm_check=args.SVM)
        X = np.array(data['Flow'])
        data['Label'] = np.array(data['Label']).flatten()
        test_y = data['Label'].astype(np.int64)

        # Use relevant model for classification
        if args.SVM:
            svm = SVM(True)
            predictions = svm.test(X)

        if args.MobileNet:
            model = MobileNet()
            mobile_model_file = f'{MODEL_PATH}/mobilenet_model.pth'
            predictions, pred_probs = test_model(data['Flow'], model, load_path=mobile_model_file, threshold=args.mobileThresh)
            
        if args.ShuffleNet:
            model = ShuffleNet()
            shuffle_model_file = f'{MODEL_PATH}/shufflenet_model.pth'
            predictions, pred_probs = test_model(data['Flow'], model, load_path=shuffle_model_file, threshold=args.shuffleThresh)
        
        # Print Evaluations
        data['Pred'] = predictions
        print(classification_report(predictions, test_y))

        if args.saveResults:
            save_results(data)

        if args.confMatrix:
            conf_matrix(predictions, test_y)

        if args.roc:
            if args.SVM:
                svm_roc(X, test_y, predictions, svm.model)

            if args.MobileNet or args.ShuffleNet:
                roc(X, test_y, pred_probs)