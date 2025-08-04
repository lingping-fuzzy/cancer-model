import os.path
import os
import torch
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import pandas as pd
from torchvision import models
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset
lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'Melanoma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }
IMG_SIZE = 256
transform_train = transforms.Compose(
[
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
transform_val = transforms.Compose(
[
        transforms.Resize((IMG_SIZE,IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _get_df_data_(source='D:/workspace/CombineModel/'):
    train_df = pd.read_csv('../data/train_df.csv').drop(columns=['Unnamed: 0'])
    val_df = pd.read_csv('../data/val_df.csv').drop(columns=['Unnamed: 0'])
    test_df = pd.read_csv('../data/test_df.csv').drop(columns=['Unnamed: 0'])

    train_df['image'] = train_df['path'].map(
        lambda x: transform_val(Image.open(os.path.join(source, x)).convert("RGB")))
    val_df['image'] = val_df['path'].map(
        lambda x: transform_val(Image.open(os.path.join(source, x)).convert("RGB")))
    test_df['image'] = test_df['path'].map(
        lambda x: transform_val(Image.open(os.path.join(source, x)).convert("RGB")))

    return train_df, val_df, test_df

def _get_df_cutdata_(source= '../data/cutimage/'):
    train_df = pd.read_csv('../data/train_df.csv').drop(columns=['Unnamed: 0'])
    val_df = pd.read_csv('../data/val_df.csv').drop(columns=['Unnamed: 0'])
    test_df = pd.read_csv('../data/test_df.csv').drop(columns=['Unnamed: 0'])

    train_df['cutpath'] = train_df['image_id'].map(
        lambda x: (os.path.join(source, (x + '_cut.png'))))

    val_df['cutpath'] = val_df['image_id'].map(
        lambda x: (os.path.join(source, (x + '_cut.png'))))
    test_df['cutpath'] = test_df['image_id'].map(
        lambda x: (os.path.join(source, (x + '_cut.png'))))

    train_df['image'] = train_df['cutpath'].map(
        lambda x: transform_val(Image.open(x).convert("RGB")))
    val_df['image'] = val_df['cutpath'].map(
        lambda x: transform_val(Image.open(x).convert("RGB")))
    test_df['image'] = test_df['cutpath'].map(
        lambda x: transform_val(Image.open(x).convert("RGB")))

    return train_df, val_df, test_df


# Define the MLP
class ImageResnet(nn.Module):
    def __init__(self,  hidden_size, output_size):
        super(ImageResnet, self).__init__()
        self.net = models.resnet18()
        num_features = self.net.fc.in_features

        self.net.fc = nn.Linear(num_features, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x

class ImageEffNet(nn.Module):
    def __init__(self, output_size, num_classes):
        super(ImageEffNet, self).__init__()
        self.net = models.efficientnet_b0(weights='DEFAULT')
        self.net.classifier[1] = nn.Linear(self.net.classifier[1].in_features, output_size)
        self.fc = nn.Linear(output_size, num_classes)

    def forward(self, x):
        x = self.net(x)
        x = self.fc(x)
        return x

def _get_model_(modelname='resnet'):
    # https://github.com/lukemelas/EfficientNet-PyTorch
    output_size = 64  # Example output size of the MLP
    num_classes = 7
    if modelname == 'resnet':
        model = ImageResnet(output_size, num_classes)
    elif modelname == 'effnet':
        model = ImageEffNet(output_size, num_classes)
    else:
        model = None
    return model

# Function to load the trained model
def load_model(model_path, modelname, output_size=64, num_classes=7):
    if modelname == 'resnet':
        model = ImageResnet(output_size, num_classes)
    elif modelname == 'effnet':
        model = ImageEffNet(output_size, num_classes)
    else:
        model = None
    model.load_state_dict(torch.load(model_path,map_location='cpu'))
    model.eval()
    return model

# Function to register hooks
def register_hooks(model, layers):
    features = {}

    def get_hook(name):
        def hook(module, input, output):
            features[name] = output.detach()
        return hook

    for name, layer in layers.items():
        layer.register_forward_hook(get_hook(name))

    return features


import numpy as np
from scipy.linalg import orth


def generate_binary_nearly_orthogonal_matrix(cnm, nm, c, seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Generate a random matrix
    binary_matrix = np.random.randn(cnm, nm)

    # # Apply orthogonalization (optional: to get closer to orthogonality)
    # # Q, _ = np.linalg.qr(random_matrix)
    # num_ones_per_column = cnm // c
    # # Convert the orthogonal matrix to a binary matrix (0 or 1)
    # # binary_matrix = (Q > 0).astype(int)
    # for col in range(nm):
    #     ones_indices = np.random.choice(cnm, size= num_ones_per_column, replace=False)
    #     binary_matrix[ones_indices, col] = 1
    total_positions = cnm * nm
    num_ones = total_positions //2
    ones_indices = np.random.choice(total_positions, size=num_ones, replace=False)
    binary_matrix.flat[ones_indices]=1

    return binary_matrix


def process_resnet_features(data, seed=None):
    b, c, n, m = data.shape
    cnm = c * n * m
    nm = n * m

    # Reshape the data
    reshaped_data = data.reshape(b, cnm)

    # Generate the binary nearly orthogonal matrix
    binary_matrix = generate_binary_nearly_orthogonal_matrix(cnm, nm, c,seed=seed)

    # Perform matrix multiplication
    result = np.dot(reshaped_data, binary_matrix)

    return result


# # Example usage
# b, c, n, m = 10, 64, 7, 7  # Example dimensions
# data = np.random.randn(b, c, n, m)  # Example data
# seed = 42  # Ensure reproducibility
#
# processed_data = process_resnet_features(data, seed=seed)
# print(processed_data.shape)  # Should be (b, nm)


# Function to extract features and predictions
def extract_features_and_predictions(data_loader, model, features, down=False):
    all_features = []
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            # Flatten and concatenate the features
            layer_features = []
            for key in features:

                if down == True and len(features[key].shape)>2:
                    temp = process_resnet_features(features[key], seed=42)
                    temp = torch.from_numpy(temp)
                else:
                    temp = features[key].view(inputs.size(0), -1)
                layer_features.append(temp)
            concatenated_features = torch.cat(layer_features, dim=1)
            print('done-')
            all_features.append(concatenated_features)
            all_labels.append(labels)
            all_predictions.append(preds)
    return torch.cat(all_features), torch.cat(all_labels), torch.cat(all_predictions)


# Function to process the data and extract features
def extract_features(data_loader, model, features):
    all_features = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            _ = model(inputs)
            # Flatten and concatenate the features
            layer_features = []
            for key in features:
                layer_features.append(features[key].view(inputs.size(0), -1))
            concatenated_features = torch.cat(layer_features, dim=1)
            all_features.append(concatenated_features)
            all_labels.append(labels)
    return torch.cat(all_features), torch.cat(all_labels)


class dfImageDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        image = self.df.iloc[idx]['image']
        label = self.df.iloc[idx]['cell_type_idx']

        return image, label


def get_fetures_eff(train_loader, model_path = 'trained_models/eff_combine_model_tune.pt'):
    # Load the model and data

    model = load_model(model_path, modelname='effnet')
    # Define the layers to hook
    layers = {
        'layer1': model.net.features[3][0].block[3][1],
        'layer2': model.net.features[5][0].block[3][1],
        'layer3': model.net.features[7][0].block[3][1],
        'classifier': model.net.classifier[1]
    }

    # Register hooks
    features = register_hooks(model, layers)


    # Extract features
    train_features, train_labels, predict_labels = extract_features_and_predictions(train_loader, model, features)
    # train_features, train_labels = extract_features(train_loader, model, features)
    return train_features, train_labels, predict_labels


def get_fetures_res(train_loader,model_path = 'trained_models/res_combine_model_tune.pt'):
    # Load the model and data

    model = load_model(model_path, modelname='resnet')
    # Define the layers to hook
    layers = {
        'layer1': model.net.layer1[1].bn2,
        'layer2': model.net.layer2[1].bn2,
        'layer3': model.net.layer3[1].bn2,
        'classifier': model.net.fc
    }

    # Register hooks
    features = register_hooks(model, layers)


    train_features, train_labels, predict_labels = extract_features_and_predictions(train_loader, model, features, down=True)
    # train_features, train_labels = extract_features(train_loader, model, features)
    return train_features, train_labels, predict_labels


def get_fetures_prob(train_loader, model_path = 'trained_models/eff_combine_model_tune.pt', modelname='effnet'):
    # Load the model and data

    model = load_model(model_path, modelname=modelname)
    # Define the layers to hook
    layers = {
        'prob': model.fc
    }
    # Register hooks
    features = register_hooks(model, layers)

    # Extract features
    train_features, train_labels, predict_labels = extract_features_and_predictions(train_loader, model, features)
    # train_features, train_labels = extract_features(train_loader, model, features)
    return train_features, train_labels, predict_labels

