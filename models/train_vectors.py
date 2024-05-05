import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small
import torch.optim as optim

from models.vectorLoader import Vector_Loader, Test_Vector_Loader


def train_model(data, model, save_path, epoch: int = 50, lr: float = 0.001):
    '''
    Function to train model to classify feature vectors

    Args:
        data: dictionary containing feature vectors and labels
        model: PyTorch object either MobileNetV3 Small or ShuffleNetV2
        save_path: Path to file to save model parameters
        epoch: Number of epochs to train for 
        lr: Learning Rate for training
    '''

    dataLoader = Vector_Loader(data)
    train_loader = DataLoader(dataLoader, batch_size=64, num_workers=0, shuffle=False)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    print("\nTraining")
    for e in range(epochs):
        model.train()
        running_loss = 0
        for vector, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(vector).squeeze()
            loss = criterion(outputs.double(), labels.double())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * vector.size(0)

        epoch_loss = running_loss / dataLoader.__len__()
        print(f'Epoch {e}: {epoch_loss} loss')

    try:
        torch.save(model.state_dict(), save_path)
        print('Model Saved')
    except:
        print('Error occured when saving model')


def test_model(data, model, load_path, threshold: float = 0.3):
    '''
    Function to test model

    Args:
        data: dictionary containing feature vectors
        model: PyTorch object either MobileNetV3 Small or ShuffleNetV2
        load_path: Path ot file containing model parameters
        threshold: threshold value to append labels using predicted probabilites

    Returns:
        preds: array of predicted labels (0,1)
    '''
    dataLoader = Test_Vector_Loader(data)
    test_loader = DataLoader(dataLoader, batch_size=64, num_workers=0, shuffle=False)

    model.load_state_dict(torch.load(load_path))
    model.eval()

    preds = np.array([])
    with torch.no_grad():
        for vector in test_loader:
            y_pred = model(vector).squeeze()
            y_pred = y_pred.numpy()
            preds = np.concatenate((preds, y_pred), axis=None)

    # Since sigmoid returns range of values between 0 - 1
    # Apply threshold to append labels
    for i in range(len(preds)):
        if preds[i] > threshold:
            preds[i] = 1
        else:
            preds[i] = 0

    return preds
            
