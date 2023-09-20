#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import os
import argparse
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True # using this setting to solve a problem with truncated images during training

def train(model, train_loader, valid_loader, epochs, loss_criterion, optimizer, device):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    for epoch in range(epochs):
        for ds_version in ['train', 'valid']:
            running_loss = 0
            running_correct = 0

            if ds_version == 'train':
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = valid_loader
            
            for data, target in dataloader:
                data, target = data.to(device), target.to(device)
                outputs = model(data)
                loss = loss_criterion(outputs, target)
                _, preds = torch.max(outputs, 1)
                running_loss += loss.item()

                # acumulating number of correct predictions so we can
                # compute accuracy by the end of epoch
                with torch.no_grad():
                    running_correct += torch.sum(preds == target).item()
                
                # perform optimization only with training set
                if ds_version == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_correct / len(dataloader.dataset)
        
            print(f'Epoch : {ds_version}-{epoch}, loss = {epoch_loss}, acc = {epoch_acc}')

def test(model, test_loader, loss_criterion, device):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    model.eval()
    running_loss = 0
    running_correct = 0

    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        outputs = model(data)
        loss = loss_criterion(outputs, target)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item()
        running_correct += torch.sum(preds == target.data).item()

    total_loss = running_loss / len(test_loader.dataset)
    total_acc = running_correct/ len(test_loader.dataset)
    print(f"Test Accuracy: {100 * total_acc}, Test Loss: {total_loss}")

def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    num_classes = 133
    model = models.resnet101(pretrained=True)

    # freeze pretrained model parameters, so we can perform transfer learning
    for param in model.parameters():
        param.requires_grad = False
    # get the number of input features of the final layer of the original network
    num_inputs = model.fc.in_features
    # create a new output layer that will output the class probabilities
    model.fc = nn.Linear(num_inputs, num_classes)
    
    return model

def create_data_loaders(data, batch_size):
    '''
    Utility function that builds dataloaders for each dataset version
    '''

    dataloaders = {version: torch.utils.data.DataLoader(data[version], batch_size, shuffle=True) for version in ['train', 'valid', 'test']}
    return dataloaders

def main(args):
    print(f"Hyperparameters used: learning_rate={args.lr}\tbatch_size={args.batch_size}\tepochs={args.epochs}")
    
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), args.lr)
    
    # storing number of epochs in a variable 
    epochs = args.epochs

    # importing training data
    # defining data augumentation and transformations for each dataset version
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    datasets = {version: torchvision.datasets.ImageFolder(os.path.join(args.data_dir, version), transform=data_transforms[version]) for version in ['train', 'valid', 'test']}
    
    # creating data loaders
    dataloaders = create_data_loaders(datasets , args.batch_size)
    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']
    test_loader = dataloaders['test']

    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    # getting gpu info
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    train(model, train_loader, valid_loader, epochs, loss_criterion, optimizer, device)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_loader, loss_criterion, device)
    
    '''
    TODO: Save the trained model
    '''
    path = os.path.join(args.model_dir, 'model.pth')
    torch.save(model, path)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify all the hyperparameters you need to use to train your model.
    '''
    # learning rate
    parser.add_argument(
        "--lr", type=float, default=0.001, metavar="LEARNING_RATE", help="learning rate. default: 0.001"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="EPOCHS",
        help="number of epochs. default: 5",
    )
    # batch_size
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="BATCH_SIZE",
        help="batch size. default: 128",
    )
    
    parser.add_argument(
        "--model-dir",
        type=str, 
        default=os.environ["SM_MODEL_DIR"]
        )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.environ["SM_CHANNEL_DATA"]
        )
    parser.add_argument(
        '--output-dir',
        type=str, 
        default=os.environ['SM_OUTPUT_DATA_DIR']
        )

    
    args=parser.parse_args()
    
    main(args)
