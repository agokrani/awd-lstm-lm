import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn

import data
import model

from utils import batchify, get_batch, repackage_hidden

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='data/penn/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (LSTM, QRNN, GRU)')
parser.add_argument('--load', type=str,
                    help='path to load the best model for evaluation')
parser.add_argument('--cuda', action='store_false',
                    help='use CUDA')
parser.add_argument('--bptt', type=int, default=70,
                    help='sequence length')
args = parser.parse_args()

def model_load(fn):
    global model, criterion, optimizer
    with open(fn, 'rb') as f:
        model, criterion, optimizer = torch.load(f)

def evaluate(data_source, batch_size=10):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if args.model == 'QRNN': model.reset()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, args, evaluation=True)
        output, hidden = model(data, hidden)
        total_loss += len(data) * criterion(model.decoder.weight, model.decoder.bias, output, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss.item() / len(data_source)


corpus = data.Corpus(args.data)

test_batch_size = 1		
test_data = batchify(corpus.test, test_batch_size, args)

# Load the best saved model.
model_load(args.load)

# Run on test data.
test_loss = evaluate(test_data, test_batch_size)
print('=' * 89)
print('| Test Results | test loss {:5.2f} | test ppl {:8.2f} | test bpc {:8.3f}'.format(
    test_loss, math.exp(test_loss), test_loss / math.log(2)))
print('=' * 89)
