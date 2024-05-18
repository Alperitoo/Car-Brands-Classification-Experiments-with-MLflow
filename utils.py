import torch

def metric_fn(outputs, labels,total,correct):
    predicted = torch.max(outputs)
    total += labels.size(0)
    correct += (predicted == labels.unsqueeze(1).float()).sum().item()
    accuracy=correct / total
    return accuracy
import mlflow
from typing import Any 

def create_experiment(name:str,artifact_location:str,tags:dict[str,Any]) ->str:
    try:
        experiment_id = mlflow.create_experiment(name, artifact_location,tags={"env":"dev","version":"1.0.0"})
    except:
        print(f"Experiment {name} already exists")
        experiment_id = mlflow.get_experiment_by_name(name).experiment_id
    return experiment_id

#experiment_id=create_experiment("CatsAndDogs","artifacts",{"env":"dev","version":"1.0.0"})
def get_mlflow_experiment(
    experiment_id: str = None, experiment_name: str = None
) -> mlflow.entities.Experiment:
    """
    Retrieve the mlflow experiment with the given id or name.

    Parameters:
    ----------
    experiment_id: str
        The id of the experiment to retrieve.
    experiment_name: str
        The name of the experiment to retrieve.

    Returns:
    -------
    experiment: mlflow.entities.Experiment
        The mlflow experiment with the given id or name.
    """
    if experiment_id is not None:
        experiment = mlflow.get_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
    else:
        raise ValueError("Either experiment_id or experiment_name must be provided.")
    return experiment

"""#experiment=get_mlflow_experiment(experiment_id=experiment_id)
print("Name: {}".format(experiment.name))
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
print("Creation timestamp: {}".format(experiment.creation_time))"""

def delete_mlflow_experiment(
    experiment_id: str = None, experiment_name: str = None
) -> None:
    """
    Delete the mlflow experiment with the given id or name.

    Parameters:
    ----------
    experiment_id: str
        The id of the experiment to delete.
    experiment_name: str
        The name of the experiment to delete.
    """
    if experiment_id is not None:
        mlflow.delete_experiment(experiment_id)
    elif experiment_name is not None:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        experiment_id = experiment.experiment_id
        mlflow.delete_experiment(experiment_id)
    else:
        raise ValueError("Either experiment_id or experiment_name must be provided.")

#delete_mlflow_experiment(experiment_name="CatsAndDogs")
