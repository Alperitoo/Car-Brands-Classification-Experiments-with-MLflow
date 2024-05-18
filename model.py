import torch
import torch.nn as nn
import mlflow
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np 
class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, 1024)
        self.fc2 = nn.Linear(1024, 19)  # Output layer with single neuron for binary classification
        self.softmax = nn.Softmax()     # Sigmoid activation for binary classification

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.conv4(x))
        x = self.pool(x)
        x = torch.relu(self.conv5(x))
        x = torch.relu(self.conv6(x))
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)  # Apply sigmoid activation for binary classification
        return x



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define models
class VGG16(nn.Module):
    def __init__(self, num_classes,dropout_rate):
        super(VGG16, self).__init__()
        self.base_model = models.vgg16(pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.dropout_rate=dropout_rate
        num_features = self.base_model.classifier[6].in_features
        self.base_model.classifier[6] = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, 19),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.base_model(x)

class ResNet18(nn.Module):
    def __init__(self, num_classes,dropout_rate):
        super(ResNet18, self).__init__()
        self.base_model = models.resnet18(pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.dropout_rate=dropout_rate
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 19),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.base_model(x)

class InceptionV3(nn.Module):
    def __init__(self, num_classes,dropout_rate):
        super(InceptionV3, self).__init__()
        self.base_model = models.inception_v3(pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.dropout_rate=dropout_rate
        num_features = self.base_model.fc.in_features
        self.base_model.fc = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, 19),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.base_model(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes,dropout_rate):
        super(MobileNetV2, self).__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        for param in self.base_model.parameters():
            param.requires_grad = False
        self.dropout_rate=dropout_rate
        
        num_features = self.base_model.classifier[1].in_features
        self.base_model.classifier[1] = nn.Sequential(
            nn.Linear(num_features, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(1024, 19),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.base_model(x)

# # Example usage:
# num_classes = 2  # Number of classes (cats and dogs)
# vgg16_model = CustomVGG16(num_classes).to(device)
# resnet50_model = CustomResNet50(num_classes).to(device)
# inceptionv3_model = CustomInceptionV3(num_classes).to(device)
# mobilenetv2_model = CustomMobileNetV2(num_classes).to(device)

def get_model(model_name, dropout_rate):
    if model_name == "VGG16":
        model = VGG16(num_classes=19, dropout_rate=dropout_rate).to(device)
    elif model_name == "ResNet18":
        model = ResNet18(num_classes=19, dropout_rate=dropout_rate).to(device)
    elif model_name == "InceptionV3":
        model = InceptionV3(num_classes=19, dropout_rate=dropout_rate).to(device)
    elif model_name == "MobileNetV2":
        model = MobileNetV2(num_classes=19, dropout_rate=dropout_rate).to(device)
    elif model_name == "CustomCNN":
        model = CustomCNN().to(device)
    return model



def train_model(model, train_loader, criterion,optimizer, metric_fn, epoch):
    #optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    correct=0
    total=0
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)  # BCELoss expects float labels
        loss.backward()
        optimizer.step()
        running_loss += loss.item()*inputs.size(0)
        _,predicted=torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    accuracy=metric_fn(outputs,labels,total,correct)
    running_loss /= len(train_loader.dataset)
    mlflow.log_metric("train_loss",running_loss,step=epoch)
    mlflow.log_metric("train_accuracy",accuracy,step=epoch)
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, Accuracy: {accuracy}")
    return accuracy,running_loss


# Evaluation function
def evaluate_model(model,val_loader,criterion,metric_fn,epoch):
    model.eval()
    correct = 0
    total = 0
    running_loss=0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _,predicted = torch.max(outputs,1)
            eval_loss=criterion(outputs,labels)
            running_loss+=eval_loss.item()*inputs.size(0)
            #eval_accuracy=metric_fn(predicted,labels.unsqueeze(1).float())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    running_loss /= len(val_loader.dataset)        
    eval_accuracy = metric_fn(outputs,labels,total,correct)
    mlflow.log_metric("eval_loss",running_loss,step=epoch)
    mlflow.log_metric("eval_accuracy",eval_accuracy,step=epoch)
    print(f"Accuracy on validation set: {(100*eval_accuracy):.2f}%")
    return eval_accuracy,running_loss
import torch

def metric_fn(outputs, labels,total,correct):
    _, predicted = torch.max(outputs, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()
    accuracy=correct / total
    return accuracy


def test_model(model,test_loader,criterion,metric_fn):
    model.eval()
    correct = 0
    total = 0
    running_loss = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _,predicted = torch.max(outputs,1)
            eval_loss=criterion(outputs,labels)
            running_loss+=eval_loss.item()*inputs.size(0)
            #eval_accuracy=metric_fn(predicted,labels.unsqueeze(1).float())
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = metric_fn(outputs,labels,total,correct)
    running_loss /= len(test_loader.dataset)
    print(f"Accuracy on test set: {(100*accuracy):.2f}%")
    mlflow.log_metric("test_accuracy",accuracy)
    mlflow.log_metric("test_loss",running_loss)
    return running_loss,accuracy

def save_model(model, path):
    torch.save(model.state_dict(), path)

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies,exp_id,run_name,run_id,gun_saat ):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))
    #train_losses = [loss.item() for loss in train_losses]
    # Plotting losses
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    plt.title('Training and Validation Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.savefig(f"./mlruns/{exp_id}/{run_id}/artifacts/Loss_graph_{run_name}.png")
    mlflow.log_artifact(f"./mlruns/{exp_id}/{run_id}/artifacts/Loss_graph_{run_name}.png")
    plt.legend()
    #plt.show()
    # Plotting accuracies
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'bo-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracies')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.savefig(f"./mlruns/{exp_id}/{run_id}/artifacts/Accuracy_graph_{run_name}.png")
    mlflow.log_artifact(f"./mlruns/{exp_id}/{run_id}/artifacts/Accuracy_graph_{run_name}.png")
    plt.legend()

    plt.tight_layout()
    #plt.show()

def display_images_predictions(model, test_loader,exp_id,run_name,run_id):
    model.eval()
    
    images, labels = next(iter(test_loader))
    images,labels = images.to(device),labels.to(device)
    # Predict
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    classes={v: k for k, v in test_loader.dataset.class_to_idx.items()}
    images,labels = images.to("cpu"),labels.to("cpu")
    # Plot
    fig = plt.figure(figsize=(10,10))
    for i in range(9):
        ax = fig.add_subplot(3, 3, i+1, xticks=[], yticks=[])
        ax.imshow(np.transpose(images[i], (1, 2, 0)))  # transpose to go from torch to numpy image
        ax.set_title(f"Predicted label:{classes[predicted[i].item()]}\n Actual Label:({classes[labels[i].item()]})",
                     color=("green" if predicted[i].item()==labels[i].item() else "red"))
    plt.savefig(f"./mlruns/{exp_id}/{run_id}/artifacts/PredictionDisplay_{run_name}.png")
    mlflow.log_artifact(f"./mlruns/{exp_id}/{run_id}/artifacts/PredictionDisplay_{run_name}.png")
    #plt.show()