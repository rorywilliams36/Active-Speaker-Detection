import torch, copy
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import mobilenet_v3_small
import torch.optim as optim

from models.vectorLoader import Vector_Loader, Test_Vector_Loader

def train_model(data, model, save_path, epochs: int = 50, lr: float = 0.003):
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
    train_loader = DataLoader(dataLoader, batch_size=32, num_workers=0, shuffle=False)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    predicted_probs = []

    losses = []
    print("\nTraining")
    for e in range(epochs):
        model.train()
        running_loss = 0
        for vector, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(vector).squeeze()
            predicted_probs.append(outputs)
            loss = criterion(outputs.double(), labels.double())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * vector.size(0)

        epoch_loss = running_loss / dataLoader.__len__()
        losses.append(epoch_loss)
            
        print(f'Epoch {e}: {epoch_loss:.4f} loss')

    try:
        torch.save(model.state_dict(), save_path)
        print('Model Saved')
    except:
        print('Error occured when saving model')

    return predicted_probs, losses


def train_validation(data, model, save_path, epochs: int=50, lr: float = 0.003, threshold: float = 0.25):
    data_size = int(round(len(data['Flow'])*0.2))-1
    valid_data = {'Flow' : [], 'Label' : []}
    valid_data['Flow'] = data['Flow'][-data_size:]
    valid_data['Label'] = data['Label'][-data_size:]
    del data['Flow'][-data_size:]
    del list(data['Label'])[-data_size:]

    dataLoader = Vector_Loader(data)
    valid_dataLoader = Vector_Loader(valid_data)

    train_loader = DataLoader(dataLoader, batch_size=32, num_workers=0, shuffle=False)
    valid_loader = DataLoader(valid_dataLoader, batch_size=32, num_workers=0, shuffle=False)

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    predicted_probs = []

    losses = []
    val_losses = []
    val_accuaracies = []
    best_acc = 0
    min_valid_loss = np.inf
    print("\nTraining")
    for e in range(epochs):
        model.train()
        running_loss = 0
        for vector, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(vector).squeeze()
            predicted_probs.append(outputs)
            loss = criterion(outputs.double(), labels.double())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * vector.size(0)

        epoch_loss = running_loss / dataLoader.__len__()
        losses.append(epoch_loss)

        valid_loss = 0
        acc = 0
        model.eval()
        preds = np.array([])
        with torch.no_grad():
            for vector, labels in valid_loader:
                pred_y = model(vector).squeeze()
                loss = criterion(pred_y.double(), labels.double())
                valid_loss += loss.item() * vector.size(0)
                preds = np.concatenate((preds, pred_y.numpy()), axis=None)

            valid_epoch_loss = valid_loss / valid_dataLoader.__len__()
            val_losses.append(valid_epoch_loss)

            pred_labels = []
            for i in preds:
                if i > 0.375:
                    pred_labels.append(1)
                else:
                    pred_labels.append(0)

            total = 0
            for i in range(len(pred_labels)):
                if pred_labels[i] == valid_data['Label'][i]:
                    total +=1

            acc = total / valid_dataLoader.__len__()
            print(acc)



        print(f'\nEpoch {e}: Training Loss {epoch_loss:.4f} loss \t\t Validation Loss: {valid_epoch_loss:.4f}')
        # if valid_epoch_loss > min_valid_loss:
        #     print(f'Valid loss increased')
        #     # print(f'Copying Model')
        #     best_weights = copy.deepcopy(model.state_dict())
        
        # min_valid_loss = valid_epoch_loss


        # if acc > best_acc:
        #     print('New Best')
        #     best_acc = acc

        val_accuaracies.append(acc)

    try:
        # torch.save(model.state_dict(), save_path)
        print('Model Saved')
    except:
        print('Error occured when saving model')

    return predicted_probs, losses, val_losses, val_accuaracies


def test_model(data, model, load_path, threshold: float = 0.25):
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
    test_loader = DataLoader(dataLoader, batch_size=32, num_workers=0, shuffle=False)

    model.load_state_dict(torch.load(load_path))
    model.eval()

    preds = np.array([])
    with torch.no_grad():
        for vector in test_loader:
            outputs = model(vector).squeeze()
            outputs = outputs.numpy()
            preds = np.concatenate((preds, outputs), axis=None)

    # Since sigmoid returns range of values between 0 - 1
    # Apply threshold to append labels
    pred_labels = []
    for i in range(len(preds)):
        if preds[i] > threshold:
            pred_labels.append(1)
        else:
            pred_labels.append(0)

    return pred_labels, preds
            
