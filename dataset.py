import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_data(root_dir,device, batch_size):
    # Define appropriate input size based on the model being used
    input_size=(224,224)

    transform = transforms.Compose([
        transforms.Resize(input_size),  
        transforms.ToTensor(),           
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(os.path.join(root_dir, "train"), transform=transform)
    val_dataset = datasets.ImageFolder(os.path.join(root_dir, "val"), transform=transform)
    test_dataset = datasets.ImageFolder(os.path.join(root_dir, "test"), transform=transform)
    print("Train dataset has", len(train_dataset), "samples")
    print("Validation dataset has", len(val_dataset), "samples")
    print("Test dataset has", len(test_dataset), "samples")
    #print(len(train_dataset), len(val_dataset), len(test_dataset))
          
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # for inputs, labels in train_loader:
    #     inputs, labels = inputs.to(device), labels.to(device)

    # for inputs, labels in val_loader:
    #     inputs, labels = inputs.to(device), labels.to(device)

    # for inputs, labels in test_loader:
    #     inputs, labels = inputs.to(device), labels.to(device)

    return train_loader, val_loader, test_loader, train_dataset, test_dataset
