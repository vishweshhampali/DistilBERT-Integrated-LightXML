import torch
import numpy as np
from torch.utils.data import DataLoader
from dataset import MDataset, createDataCSV
from model import LightXML
import argparse
import time
import csv  # Import CSV to save results

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')
args = parser.parse_args()

if __name__ == '__main__':
    # Load dataset
    df, label_map = createDataCSV(args.dataset)
    print(f'Loaded {args.dataset} dataset with '
          f'{len(df[df.dataType == "train"])} train samples and {len(df[df.dataType == "test"])} test samples, with {len(label_map)} labels')

    xmc_models = []
    predicts = []
    berts = ['bert-base', 'roberta', 'xlnet']

    # Prepare CSV file to store results
    csv_file_path = f'./results/{args.dataset}_results.csv'
    with open(csv_file_path, 'w', newline='') as csvfile:
        fieldnames = ['Model', 'Inference Time (s)', 'Top-1 Accuracy', 'Top-3 Accuracy', 'Top-5 Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()  # Write header to CSV file

    # Loop over each transformer model (BERT, RoBERTa, XLNet)
    for index in range(len(berts)):
        model_name = [args.dataset, '' if berts[index] == 'bert-base' else berts[index]]
        model_name = '_'.join([i for i in model_name if i != ''])

        model = LightXML(n_labels=len(label_map), bert=berts[index])

        print(f'Loading model: {model_name}')
        model.load_state_dict(torch.load(f'models/model-{model_name}.bin'))

        tokenizer = model.get_tokenizer()
        test_d = MDataset(df, 'test', tokenizer, label_map, 128 if args.dataset == 'amazoncat13k' and berts[index] == 'xlnent' else 512)
        testloader = DataLoader(test_d, batch_size=16, num_workers=0, shuffle=False)

        model.cuda()  # Move model to GPU if available

        # Measure inference time
        start_time = time.time()

        # Run inference for this model
        predicts.append(torch.Tensor(model.one_epoch(0, testloader, None, mode='test')[0]))

        end_time = time.time()
        inference_time = end_time - start_time
        print(f'{berts[index]} inference time: {inference_time:.2f} seconds')

        # Save the model
        xmc_models.append(model)

    # Filter only test data
    df = df[df.dataType == 'test']
    total = len(df)
    acc1 = [0 for _ in range(len(berts))]
    acc3 = [0 for _ in range(len(berts))]
    acc5 = [0 for _ in range(len(berts))]

    # Evaluate model accuracy for each model (without ensemble)
    for index, true_labels in enumerate(df.label.values):
        true_labels = set([label_map[i] for i in true_labels.split()])

        # Get top predictions for each model individually (no ensemble)
        logits = [torch.sigmoid(predicts[i][index]) for i in range(len(berts))]
        logits = [(-i).argsort()[:10].cpu().numpy() for i in logits]

        # Calculate accuracy for each model
        for i, logit in enumerate(logits):
            acc1[i] += len(set([logit[0]]) & true_labels)
            acc3[i] += len(set(logit[:3]) & true_labels)
            acc5[i] += len(set(logit[:5]) & true_labels)

    # Log results for each model (no ensemble) and save to CSV
    for i, name in enumerate(berts):
        p1 = acc1[i] / total
        p3 = acc3[i] / total / 3
        p5 = acc5[i] / total / 5

        # Print results to console
        print(f'{name} Top-1 Accuracy: {p1:.4f}, Top-3 Accuracy: {p3:.4f}, Top-5 Accuracy: {p5:.4f}')

        # Write results to CSV file
        with open(csv_file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow({
                'Model': name,
                'Inference Time (s)': inference_time,
                'Top-1 Accuracy': p1,
                'Top-3 Accuracy': p3,
                'Top-5 Accuracy': p5
            })

        # Also save results to a text file if needed
        with open(f'./results/{args.dataset}', 'a') as f:
            print(f'{name} {p1:.4f} {p3:.4f} {p5:.4f}', file=f)
