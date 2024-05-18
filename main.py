import torch
from torch import nn
from torch.utils.data import DataLoader
from torchinfo import summary
from torchmetrics.classification import Accuracy
from torchvision import datasets,transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import mlflow
from dataset import load_data
from model import get_model,train_model,evaluate_model,metric_fn,test_model,plot_training_history,metric_fn,display_images_predictions
from utils import create_experiment,get_mlflow_experiment,delete_mlflow_experiment
from datetime import datetime
import os
import warnings
warnings.filterwarnings("ignore")
import time
start_time = time.time()
current_datetime = datetime.now()
gun_saat = current_datetime.strftime("%Y_%m_%d-%H_%M_%S")

#Create a new MLflow Experiment
experiment_id=create_experiment(name="Car-Brands-Classification",artifact_location="CarBrandsArtifacts",tags={"env":"dev","version":"1.0.0"})
experiment=get_mlflow_experiment(experiment_id=experiment_id)
experiment_name=experiment.name

print("Name: {}".format(experiment.name))
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
print("Creation timestamp: {}".format(experiment.creation_time))

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

#MODELS=["VGG16","ResNet18","MobileNetV2","InceptionV3","CustomCNN"]

root_dir="./car_data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#train_loader,val_loader,test_loader,train_dataset,test_dataset=load_data(root_dir,device)
# Define hyperparameters

hyperparameters = {
    'learning_rate': [1e-3,1e-4], # [1e-3, 1e-4, 1e-5]
    'optimizer': ['Adam'],  #['Adam', 'SGD']
    'dropout_rate': [0.5,0.8], # [0.5, 0.6, 0.7, 0.8] 0.5 is been used
}
batch_size = 32
metric_fn=metric_fn
models = ["MobileNetV2",'ResNet18']
train_loader,val_loader,test_loader,train_dataset,test_dataset=load_data(root_dir,device,batch_size)
for model_name in models:
    #model = get_model(model_name)
    for lr in hyperparameters['learning_rate']:
        for optimizer_name in hyperparameters['optimizer']:
            for dropout_rate in hyperparameters['dropout_rate']:
                run_name = f"{model_name}_{optimizer_name}_lr={lr}_batch_size={batch_size}_dropout={dropout_rate}_{gun_saat}"
                with mlflow.start_run(run_name=run_name, experiment_id=experiment_id) as run:
                    run_id = run.info.run_id
                    model=get_model(model_name,dropout_rate)
                    params = {
                        "experiment_name": experiment_name,
                        "run_name": run_name,
                        "run_id": run_id,
                        "model_name": model_name,
                        "learning_rate": lr,
                        "batch_size": batch_size,
                        "optimizer": optimizer_name,
                        "dropout_rate": dropout_rate,
                        "model_summary": summary(model),
                    }
                    # Log training parameters.
                    mlflow.log_params(params)
                    #model=get_model(model_name,dropout_rate)
                    criterion = nn.CrossEntropyLoss()
                    optimizer=torch.optim.Adam(model.parameters(), lr=lr) 
                    # Log model summary.
                    with open(f"model_summary_{run_name}.txt", "w", encoding="utf-8") as f:
                        f.write(str(summary(model)))
                    mlflow.log_artifact(f"model_summary_{run_name}.txt")
                    train_accuracies = []
                    val_accuracies = []
                    train_losses = []
                    val_losses = []
                    epochs = 30
                    for t in range(epochs):
                        print(f"Epoch {t+1}\n-------------------------------")
                        tr_acc, tr_loss = train_model(model, train_loader, criterion,optimizer, metric_fn, epoch=t)
                        val_acc, val_loss = evaluate_model(model, val_loader, criterion, metric_fn, epoch=t)
                        train_accuracies.append(tr_acc)
                        val_accuracies.append(val_acc)
                        train_losses.append(tr_loss)
                        val_losses.append(val_loss)
                    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, experiment_id,run_name,run_id, gun_saat)
                    display_images_predictions(model,test_loader,experiment_id,run_name,run_id)
                    loss, accuracy = test_model(model, test_loader, criterion, metric_fn)
                    print(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
                    # Save the trained model to MLflow.
                    mlflow.pytorch.log_model(model, "model")
                    mlflow.pytorch.save_model(model, f"{experiment_id}/{run_name}.pth")
                    #mlflow.log_artifact(f"{model_name}_summary.txt")
                    mlflow.log_artifact(f"model_summary_{run_name}.txt")
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print(f"Elapsed time: {elapsed_time:.2f} seconds")
                    
                    mlflow.log_metric(f"elapsed_time_in_minutes", (elapsed_time/60))
                mlflow.end_run()