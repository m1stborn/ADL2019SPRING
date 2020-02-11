from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef, f1_score

# from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear

from run_classifier import BCNProcessor,convert_examples_to_features
parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--data_dir",
                    default="./data/test.csv ",
                    type=str,
                    # required=True,
                    help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
parser.add_argument("--bert_model", default='bert-large-uncased', type=str, #required=True,
                    help="Bert pre-trained model selected in the list: bert-base-uncased, "
                    "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                    "bert-base-multilingual-cased, bert-base-chinese.")
parser.add_argument("--task_name",
                    default='BCN',
                    type=str,
                    # required=True,
                    help="The name of the task to train.")
parser.add_argument("--output_dir",
                    default="./model/model_01.27",
                    type=str,
                    # required=True,
                    help="The output directory where the model predictions and checkpoints will be written.")

## Other parameters
parser.add_argument("--cache_dir",
                    default="./cache/",
                    type=str,
                    help="Where do you want to store the pre-trained models downloaded from s3")
parser.add_argument("--max_seq_length",
                    default=64,
                    type=int,
                    help="The maximum total input sequence length after WordPiece tokenization. \n"
                         "Sequences longer than this will be truncated, and sequences shorter \n"
                         "than this will be padded.")
parser.add_argument("--do_train",
                    default=True,
                    action='store_true',
                    help="Whether to run training.")
parser.add_argument("--do_eval",
                    default=True,
                    action='store_true',
                    help="Whether to run eval on the dev set.")
parser.add_argument("--do_lower_case",
                    action='store_true',
                    help="Set this flag if you are using an uncased model.")

parser.add_argument("--eval_batch_size",
                    default=32,
                    type=int,
                    help="Total batch size for eval.")
parser.add_argument("--learning_rate",
                    default=3e-5,
                    type=float,
                    help="The initial learning rate for Adam.")

parser.add_argument('--seed',
                    type=int,
                    default=9527,
                    help="random seed for initialization")

parser.add_argument('--use_epoch',
                    type=int,
                    default=1,
                    help="which epoch to use")

parser.add_argument('--out',
                    type=str,
                    default="out.csv",
                    help="path to predict")


args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger(__name__)


processor = BCNProcessor()
output_mode = "classification"
label_list = processor.get_labels()
num_labels = len(label_list)
tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

# Load a trained model that you have fine-tuned
output_model_file = os.path.join(args.output_dir, 'weight.{}'.format(args.use_epoch))
model_state_dict = torch.load(output_model_file)
model = BertForSequenceClassification.from_pretrained(args.bert_model,cache_dir=args.cache_dir, state_dict=model_state_dict,num_labels=num_labels)
model.to(device)

test_examples = processor.get_test_examples(args.data_dir)
eval_features = convert_examples_to_features(
    test_examples, label_list, args.max_seq_length, tokenizer, output_mode)
logger.info("***** Running testing *****")
logger.info("  Num examples = %d", len(test_examples))
logger.info("  Batch size = %d", args.eval_batch_size)
all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

if output_mode == "classification":
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
elif output_mode == "regression":
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.float)

eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
# Run prediction for full data
eval_sampler = SequentialSampler(eval_data)
eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

model.eval()
eval_loss = 0
nb_eval_steps = 0
preds = []

for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader, desc="Evaluating",ncols=70):
    input_ids = input_ids.to(device)
    input_mask = input_mask.to(device)
    segment_ids = segment_ids.to(device)
    label_ids = label_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids, segment_ids, input_mask, labels=None)

    # create eval loss and other metric required by the task
    if output_mode == "classification":
        loss_fct = CrossEntropyLoss()
        tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
    elif output_mode == "regression":
        loss_fct = MSELoss()
        tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))

    eval_loss += tmp_eval_loss.mean().item()
    nb_eval_steps += 1
    if len(preds) == 0:
        preds.append(logits.detach().cpu().numpy())
    else:
        preds[0] = np.append(
            preds[0], logits.detach().cpu().numpy(), axis=0)

eval_loss = eval_loss / nb_eval_steps
preds = preds[0]
if output_mode == "classification":
    preds = np.argmax(preds, axis=1)
elif output_mode == "regression":
    preds = np.squeeze(preds)
# output
test_file = os.path.join(args.out)
with open (test_file, "w")as f:
    writer = csv.writer(f)
    writer.writerow(['Id','label'])
    for i,pred in enumerate(preds):
        writer.writerow([i+20001,pred+1])
