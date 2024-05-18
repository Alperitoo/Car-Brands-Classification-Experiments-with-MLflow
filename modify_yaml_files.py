import os
import yaml
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--user_path', type=str, help='Specify the path of the project')
user_path = parser.parse_args().user_path
user_path = user_path.replace("\\", "/")
def update_artifact_uri(yaml_path, new_base_path):
    # Read the YAML file
    with open(yaml_path, 'r',encoding="utf-8") as file:
        data = yaml.safe_load(file)
    
    # Get the old artifact_uri path
    old_artifact_uri = data['artifact_uri']
    
    # Replace the part before /CarBrands
    new_artifact_uri = old_artifact_uri.split('/CarBrands', 1)[-1]
    new_artifact_uri = new_base_path + '/CarBrands' + new_artifact_uri
    
    # Assign the updated value
    data['artifact_uri'] = new_artifact_uri
    
    # Save the updated YAML file
    with open(yaml_path, 'w',encoding="utf-8") as file:
        yaml.safe_dump(data, file,allow_unicode=True)

# Root directory containing the runs
root_dir = "mlruns/321951008137017149"

# Iterate over each run_id in the root directory
for run_id in os.listdir(root_dir):
    # Check if the current item is a directory
    if os.path.isdir(os.path.join(root_dir, run_id)):
        run_path = root_dir + '/' + run_id
        
        # Iterate over each file in the run_path directory
        for file in os.listdir(run_path):
            # Check if the file has a .yaml extension
            if file.endswith('.yaml'):
                yaml_file = run_path + '/' + file
                # Update the artifact_uri in the YAML file
                update_artifact_uri(yaml_file, user_path)
