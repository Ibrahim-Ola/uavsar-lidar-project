
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

class PyTorchDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        data = torch.tensor(self.features.iloc[index].values, dtype=torch.float)
        label = torch.tensor(self.labels.iloc[index], dtype=torch.float)

        return data, label
    

def create_dataset_for_dnn(split, columns_of_interest, batch_size=128):

    """
    A function that creates a PyTorch dataset for the DNN model.

    Parameters:
    -----------
    split : pandas DataFrame
        A pandas DataFrame containing the data to split.
    
    columns_of_interest : list
        A list of columns to use for the model.

    batch_size : int
        The batch size to use for the model.

    Returns:
    --------
    A dictionary containing the training, testing and validation dataloaders.
    """

    ## scale varibles
    scaler = StandardScaler()
    scaler.fit(split['X_train'][columns_of_interest])

    X_train = scaler.transform(split['X_train'][columns_of_interest])
    X_test = scaler.transform(split['X_test'][columns_of_interest])
    X_val = scaler.transform(split['X_val'][columns_of_interest])

    train_dataset = PyTorchDataset(
        features=X_train,
        labels=split['y_train']
    )

    test_dataset = PyTorchDataset(
        features=X_test,
        labels=split['y_test']
    )

    val_dataset = PyTorchDataset(
        features=X_val,
        labels=split['y_val']
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )

    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    return {
        'train_dataloader': train_dataloader,
        'test_dataloader': test_dataloader,
        'val_dataloader': val_dataloader,
    }