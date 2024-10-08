import sys
import random
import numpy as np
from apex import amp
from model import LightXML
import os

from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

from transformers import AdamW

import torch

from torch.utils.data import DataLoader
from dataset import MDataset, createDataCSV
from log import Logger


import psutil
import time
import subprocess
import csv

def save_final_evaluation_score(dataset, bert_name, final_scores, csv_path='final_evaluation_scores.csv'):
    """
    Save the final evaluation score, BERT model name, and dataset name to a CSV file.

    :param dataset: The name of the dataset used.
    :param bert_name: The name of the BERT model used.
    :param final_scores: A dictionary of evaluation scores (e.g., {'p1': 0.85, 'p3': 0.75, 'p5': 0.65}).
    :param csv_path: The path to the CSV file where the data will be saved.
    """
    fieldnames = ['dataset', 'bert_name', 'p1', 'p3', 'p5', 'g_p1', 'g_p3', 'g_p5']
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            'dataset': dataset,
            'bert_name': bert_name,
            'p1': final_scores.get('p1', 'N/A'),
            'p3': final_scores.get('p3', 'N/A'),
            'p5': final_scores.get('p5', 'N/A'),
            'g_p1': final_scores.get('g_p1', 'N/A'),
            'g_p3': final_scores.get('g_p3', 'N/A'),
            'g_p5': final_scores.get('g_p5', 'N/A'),
        })


def compute_model_stats(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_size = sum(p.element_size() * p.nelement() for p in model.parameters()) / 1024 ** 2

    stats = {
        'model_name': model.__class__.__name__,
        'bert_name': model.bert_name,  # Include BERT model name
        'total_params': total_params,
        'trainable_params': trainable_params,
        'param_size_mb': param_size
    }

    return stats

def save_model_stats_to_csv(stats, csv_path='model_stats.csv'):
    fieldnames = ['model_name', 'bert_name', 'total_params', 'trainable_params', 'param_size_mb']
    file_exists = os.path.isfile(csv_path)

    with open(csv_path, mode='a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(stats)


def log_gpu_usage(useddataset, usedtransformer, gpu_usage_accumulator):
    gpu_stats = subprocess.check_output(
        ['nvidia-smi', '--query-gpu=utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,power.draw',
         '--format=csv,nounits,noheader']
    ).decode('utf-8').strip().split('\n')

    for stat in gpu_stats:
        gpu_util, mem_util, mem_total, mem_used, mem_free, power_draw = map(float, stat.split(','))
        gpu_usage_accumulator.append({
            'gpu_util': gpu_util,
            'mem_util': mem_util,
            'mem_total': mem_total,
            'mem_used': mem_used,
            'mem_free': mem_free,
            'power_draw': power_draw
        })

def aggregate_gpu_usage(gpu_usage_accumulator, dataset, transformer):
    total_entries = len(gpu_usage_accumulator)
    if total_entries == 0:
        return

    avg_gpu_util = sum(entry['gpu_util'] for entry in gpu_usage_accumulator) / total_entries
    avg_mem_util = sum(entry['mem_util'] for entry in gpu_usage_accumulator) / total_entries
    avg_mem_used = sum(entry['mem_used'] for entry in gpu_usage_accumulator) / total_entries
    avg_mem_free = sum(entry['mem_free'] for entry in gpu_usage_accumulator) / total_entries
    avg_power_draw = sum(entry['power_draw'] for entry in gpu_usage_accumulator) / total_entries

    with open('gpu_usage_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([dataset, transformer, avg_gpu_util, avg_mem_util, avg_mem_used, avg_mem_free, avg_power_draw])


def log_cpu_memory_usage(useddataset, usedtransformer):
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    with open('cpu_memory_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([useddataset, usedtransformer, time.time(), cpu_usage, memory_info.percent])

def record_epoch_time(dataset, transformer, epoch_times):
    average_epoch_time = sum(epoch_times) / len(epoch_times)
    with open('epoch_time_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([dataset, transformer, average_epoch_time])
def record_total_training_time(dataset, transformer, total_training_time):
    with open('total_training_time_log.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([dataset, transformer, total_training_time])

def load_group(dataset, group_tree=0):
    if dataset == 'wiki500k':
        return np.load(f'./data/Wiki-500K/label_group{group_tree}.npy', allow_pickle=True)
    elif dataset == 'amazon670k':
        return np.load(f'./data/Amazon-670K/label_group{group_tree}.npy', allow_pickle=True)

def train(model, df, label_map):
    tokenizer = model.get_tokenizer()


    if args.dataset in ['wiki500k', 'amazon670k']:
        group_y = load_group(args.dataset, args.group_y_group)
        train_d = MDataset(df, 'train', tokenizer, label_map, args.max_len, group_y=group_y,
                           candidates_num=args.group_y_candidate_num)#, token_type_ids=token_type_ids)
        test_d = MDataset(df, 'test', tokenizer, label_map, args.max_len, 
                           candidates_num=args.group_y_candidate_num)#, token_type_ids=token_type_ids)

        train_d.tokenizer = model.get_fast_tokenizer()
        test_d.tokenizer = model.get_fast_tokenizer()

        trainloader = DataLoader(train_d, batch_size=args.batch, num_workers=5,
                                 shuffle=True)
        testloader = DataLoader(test_d, batch_size=args.batch, num_workers=5,
                                shuffle=False)
        if args.valid:
            valid_d = MDataset(df, 'valid', tokenizer, label_map, args.max_len, group_y=group_y,
                               candidates_num=args.group_y_candidate_num)
            validloader = DataLoader(valid_d, batch_size=args.batch, num_workers=0, 
                                     shuffle=False)
    else:
        train_d = MDataset(df, 'train', tokenizer, label_map, args.max_len)
        test_d = MDataset(df, 'test', tokenizer, label_map, args.max_len)
        trainloader = DataLoader(train_d, batch_size=args.batch, num_workers=2,
                                 shuffle=True)
        testloader = DataLoader(test_d, batch_size=args.batch, num_workers=1,
                                shuffle=False)

    model.cuda()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)#, eps=1e-8)
    
    # Original Code    
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    
    # Changed code for mobile bert
    #model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
    epoch_times = []
    max_only_p5 = 0
    gpu_usage_accumulator = []

    for epoch in range(0, args.epoch+5):
        log_gpu_usage(args.dataset, args.bert, gpu_usage_accumulator)

        log_cpu_memory_usage(args.dataset, args.bert) # Log CPU and Memory usage

        epoch_start_time = time.time()  # Start timing this epoch


        train_loss = model.one_epoch(epoch, trainloader, optimizer, mode='train',
                                     eval_loader=validloader if args.valid else testloader,
                                     eval_step=args.eval_step, log=LOG)

        
        epoch_end_time = time.time()
        epoch_times.append(epoch_end_time - epoch_start_time)

        if args.valid:
            ev_result = model.one_epoch(epoch, validloader, optimizer, mode='eval')
        else:
            ev_result = model.one_epoch(epoch, testloader, optimizer, mode='eval')

        g_p1, g_p3, g_p5, p1, p3, p5 = ev_result

        log_str = f'{epoch:>2}: {p1:.4f}, {p3:.4f}, {p5:.4f}, train_loss:{train_loss}'
        if args.dataset in ['wiki500k', 'amazon670k']:
            log_str += f' {g_p1:.4f}, {g_p3:.4f}, {g_p5:.4f}'
        if args.valid:
            log_str += ' valid'
        LOG.log(log_str)

        if max_only_p5 < p5:
            max_only_p5 = p5
            model.save_model(f'models/model-{get_exp_name()}.bin')

        if epoch >= args.epoch + 5 and max_only_p5 != p5:
            break

    # Log final evaluation scores
    final_scores = {
        'p1': p1,
        'p3': p3,
        'p5': p5,
        'g_p1': g_p1,
        'g_p3': g_p3,
        'g_p5': g_p5
    }
    save_final_evaluation_score(args.dataset, args.bert, final_scores)


    aggregate_gpu_usage(gpu_usage_accumulator, args.dataset, args.bert)

    record_epoch_time(args.dataset, args.bert, epoch_times)

def get_exp_name():
    name = [args.dataset, '' if args.bert == 'bert-base' else args.bert]
    if args.dataset in ['wiki500k', 'amazon670k']:
        name.append('t'+str(args.group_y_group))

    return '_'.join([i for i in name if i != ''])


def init_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch', type=int, required=False, default=16)
parser.add_argument('--update_count', type=int, required=False, default=1)
parser.add_argument('--lr', type=float, required=False, default=0.0001)
parser.add_argument('--seed', type=int, required=False, default=6088)
parser.add_argument('--epoch', type=int, required=False, default=20)
parser.add_argument('--dataset', type=str, required=False, default='eurlex4k')
parser.add_argument('--bert', type=str, required=False, default='bert-base')

parser.add_argument('--max_len', type=int, required=False, default=512)

parser.add_argument('--valid', action='store_true')

parser.add_argument('--swa', action='store_true')
parser.add_argument('--swa_warmup', type=int, required=False, default=10)
parser.add_argument('--swa_step', type=int, required=False, default=100)

parser.add_argument('--group_y_group', type=int, default=0)
parser.add_argument('--group_y_candidate_num', type=int, required=False, default=3000)
parser.add_argument('--group_y_candidate_topk', type=int, required=False, default=10)

parser.add_argument('--eval_step', type=int, required=False, default=20000)

parser.add_argument('--hidden_dim', type=int, required=False, default=300)

parser.add_argument('--eval_model', action='store_true')

args = parser.parse_args()

if __name__ == '__main__':
    init_seed(args.seed)

    print(get_exp_name())

    LOG = Logger('log_'+get_exp_name())
    
    print(f'load {args.dataset} dataset...')
    df, label_map = createDataCSV(args.dataset)
    if args.valid:
        train_df, valid_df = train_test_split(df[df['dataType'] == 'train'],
                                              test_size=4000,
                                              random_state=1240)
        df.iloc[valid_df.index.values, 2] = 'valid'
        print('valid size', len(df[df['dataType'] == 'valid']))

    print(f'load {args.dataset} dataset with '
          f'{len(df[df.dataType =="train"])} train {len(df[df.dataType =="test"])} test with {len(label_map)} labels done')

    if args.dataset in ['wiki500k', 'amazon670k']:
        group_y = load_group(args.dataset, args.group_y_group)
        _group_y = []
        for idx, labels in enumerate(group_y):
            _group_y.append([])
            for label in labels:
                _group_y[-1].append(label_map[label])
            _group_y[-1] = np.array(_group_y[-1])
        group_y = np.array(_group_y)

        model = LightXML(n_labels=len(label_map), group_y=group_y, bert=args.bert,
                          update_count=args.update_count,
                          use_swa=args.swa, swa_warmup_epoch=args.swa_warmup, swa_update_step=args.swa_step,
                          candidates_topk=args.group_y_candidate_topk,
                          hidden_dim=args.hidden_dim)
    else:
        model = LightXML(n_labels=len(label_map), bert=args.bert,
                         update_count=args.update_count,
                         use_swa=args.swa, swa_warmup_epoch=args.swa_warmup, swa_update_step=args.swa_step)

    if args.eval_model and args.dataset in ['wiki500k', 'amazon670k']:
        print(f'load models/model-{get_exp_name()}.bin')
        testloader = DataLoader(MDataset(df, 'test', model.get_fast_tokenizer(), label_map, args.max_len, 
                                         candidates_num=args.group_y_candidate_num),
                                batch_size=256, num_workers=0, 
                                shuffle=False)

        group_y = load_group(args.dataset, args.group_y_group)
        validloader = DataLoader(MDataset(df, 'valid', model.get_fast_tokenizer(), label_map, args.max_len, group_y=group_y,
                                          candidates_num=args.group_y_candidate_num),
                                 batch_size=256, num_workers=0, 
                            shuffle=False)
        model.load_state_dict(torch.load(f'models/model-{get_exp_name()}.bin'))
        model = model.cuda()

        print(len(df[df.dataType == 'test']))
        model.one_epoch(0, validloader, None, mode='eval')

        pred_scores, pred_labels = model.one_epoch(0, testloader, None, mode='test')
        np.save(f'results/{get_exp_name()}-labels.npy', np.array(pred_labels))
        np.save(f'results/{get_exp_name()}-scores.npy', np.array(pred_scores))
        sys.exit(0)

    start_time = time.time()  # Start total training time tracking

    train(model, df, label_map)

    stats = compute_model_stats(model)
    save_model_stats_to_csv(stats)

    end_time = time.time() # End total training time tracking

    # Calculate the total training time
    total_training_time = end_time - start_time

    # Log the total training time
    record_total_training_time(args.dataset, args.bert, total_training_time)