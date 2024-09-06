import torch
from model import LightXML

from dataset import MDataset, createDataCSV
# Function to get model stats
def get_model_stats(model):
    num_params = sum(p.numel() for p in model.parameters())
    model_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    return num_params, model_size_mb

# List of transformer models you want to inspect
berts = ['bert-base', 'roberta', 'distilbert']

# Load dataset for label map
df, label_map = createDataCSV('amazoncat13k')  # Example dataset, you can change this

# Iterate over each model and get stats
for index in range(len(berts)):
    model_name = ['amazoncat13k', '' if berts[index] == 'bert-base' else berts[index]]
    model_name = '_'.join([i for i in model_name if i != ''])
    model = LightXML(n_labels=len(label_map), bert=berts[index])
    num_params, model_size_mb = get_model_stats(model)
    print(f'Model: {berts[index]}')
    print(f'Number of Parameters: {num_params}')
    print(f'Model Size: {model_size_mb:.2f} MB')
    print('-------------------------------------')
