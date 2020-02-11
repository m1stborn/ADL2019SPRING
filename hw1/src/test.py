from example_predictor import ExamplePredictor
from base_predictor import BasePredictor
from tqdm import tqdm,trange
from time import sleep
import numpy as np
import torch
import torch.utils.data as Data
# PredictorClass = BasePredictor
# predictor = PredictorClass()
# predictor.p(2)
# for i in trange(1000,ncols=80):
#     pass
test = np.array([0,1,2,3,4,5,6,7,8,9,10,11])

inputing = torch.tensor(np.array([test[i:i + 3] for i in range(10)]))
target = torch.tensor(np.array([test[i:i + 1] for i in range(10)]))
# print(inputing)
# torch_dataset = Data.TensorDataset(inputing,target)
# batch = 3
# print(torch_dataset[0][1])
# print(len(torch_dataset))
# print(inputing)
# print(target)
# loader = Data.DataLoader(
#     dataset=torch_dataset,
#     batch_size=batch,
#     shuffle=True,
# )
#
# for epoch in range(3):
#     for step, (batch_x, batch_y) in enumerate(loader):
#         print('Epoch: ', epoch, '| Step: ', step, '| batch x: ',batch_x.numpy(), '| batch y: ', batch_y.numpy())
import argparse
def process_command():
    parser = argparse.ArgumentParser()
    parser.add_argument('--foo', help='foo help')
    parser.add_argument('--text', '-t', type=str, help='Text for program')
    return parser.parse_args()
import re
def _load_embedding(embedding_path="../data/crawl-300d-2M.vec"):
    vectors = []
    with open(embedding_path, encoding="utf-8") as fp:

        row1 = fp.readline()
        # if the first row is not header
        if not re.match('^[0-9]+ [0-9]+$', row1):
            # seek to 0
            fp.seek(0)
        # otherwise ignore the header

        for i, line in enumerate(fp):
            cols = line.rstrip().split(' ')
            word = cols[0]
            if i < 3:
                print(word)
                print(len(cols))

            else:
                break
            # skip word not in words if words are provided
            # if words is not None and word not in words:
            #     continue
            # elif word not in self.word_dict:
            #     self.word_dict[word] = len(self.word_dict)
            #     vectors.append([float(v) for v in cols[1:]])

    # vectors = torch.tensor(vectors)
    # if self.vectors is not None:
    #     self.vectors = torch.cat([self.vectors, vectors], dim=0)
    # else:
    #     self.vectors = vectors



import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import random
import numpy as np
import pickle
from torch.utils.data import DataLoader
from torch.autograd import Variable

def load_pretrained(embedding_vecs):
    rows, cols = embedding_vecs.shape
    embedding = torch.nn.Embedding(num_embeddings=rows, embedding_dim=cols)
    embedding.weight = torch.nn.Parameter(embedding_vecs)
    embedding.weight.requires_grad = False

    return embedding

def _init_hidden(batch_size):
    hidden = torch.zeros(3,batch_size,300)
    return Variable(hidden)
if __name__ == '__main__':
    with open('../data/valid.pkl', 'rb') as file:
        valid = pickle.load(file)
    # with open('../data/test.pkl', 'rb') as file:
    #     test = pickle.load(file)
    with open('../data/train.pkl', 'rb') as file:
        train = pickle.load(file)
    with open('../data/embedding.pkl', 'rb') as file:
        embeddings = pickle.load(file)
    print('train')
    # print(train.collate_fn([train[0],train[1]])['options'].size())
    batch = train.collate_fn([train[0],train[1],train[2],train[3],train[4],train[5],train[6],train[7],train[8],train[9]])
    # print(train.collate_fn([train[0],train[1]])['context'].size())
    # print(train.collate_fn([train[0],train[1]])['labels'].size())
    # # print('valid')
    # print(valid.collate_fn([valid[0],valid[1]])['options'].size())
    # print(valid.collate_fn([valid[0],valid[1]])['context'].size())
    # print(valid.collate_fn([valid[0],valid[1]])['labels'].size())
    # print('test')
    # print(test.collate_fn([test[0],valid[1]])['options'].size())
    # print(test.collate_fn([test[0],valid[1]])['context'].size())
    # print(test.collate_fn([test[0],valid[1]])['labels'].size())
    embedded = load_pretrained(embeddings.vectors)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    embedded = embedded.to(device)
    context = embedded(batch['context'].long().to(device))
    options = embedded(batch['options'].long().to(device))
    # print(context.size())
    # print(options.size())
    # print(options.transpose(1,0).size())
    from modules import ExampleNet
    model = ExampleNet(300)
    model.to(device)

    train_dataloader = DataLoader(train,batch_size=10,collate_fn=train.collate_fn,shuffle=train.shuffle)
    valid_dataloader = DataLoader(valid,batch_size=10,collate_fn=valid.collate_fn,shuffle=valid.shuffle)
    print(valid.shuffle)
    print(train.shuffle)
    gru_context = torch.nn.GRU(300,300,3).to(device)
    gru_options = torch.nn.GRU(300,300,3).to(device)
    w = torch.autograd.Variable(torch.randn(300,300), requires_grad=True).to(device)
    w2 = torch.nn.Sequential(
        torch.nn.Linear(300, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 1),
        torch.nn.ReLU(),
    ).to(device)
    
    # for i,batch in enumerate(train_dataloader):
    #     if i != 0:
    #         break
    #     # context
    #     print(batch['context'].size())
    #     context = embedded(batch['context'].long().to(device))
    #     print(context.size()[0])
    #     print(batch['context_lens'])
    #     context_hidden = _init_hidden(10).to(device)
    #     lengths, idx_sort = torch.sort(torch.Tensor(batch['context_lens']), dim=0, descending=True)
    #     _, idx_unsort = torch.sort(idx_sort, dim=0)
    #     # print(batch['context'])
    #     context = context[idx_sort]
    #     context_packed = torch.nn.utils.rnn.pack_padded_sequence(context, lengths,batch_first=True).to(device)
    #     context_outputs,context_hidden = gru_context(context_packed,context_hidden)
    #     context_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(context_outputs,batch_first=True)
    #     context_outputs = context_outputs[idx_unsort]
    #     print(context_outputs.max(1)[0].size())
    #     context_outputs = context_outputs.max(1)[0]
    #     print(context_outputs.size())
    #     # options
    #     print(batch['options'].size())
    #     options = embedded(batch['options'].long().to(device))
    #     # logits = []
    #     logits = []
    #     print(options.transpose(1,0).size())
    #     for i,option in enumerate(options.transpose(1,0)):
    #         # if i != 0:
    #         #     break
    #         print(option.size())
    #         option_hidden = _init_hidden(10).to(device)
    #         lengths, idx_sort = torch.sort(torch.Tensor(np.asarray(batch['option_lens'])[:,i]), dim=0, descending=True)
    #         _, idx_unsort = torch.sort(idx_sort, dim=0)
    #         option = option[idx_sort]
    #         option_packed = torch.nn.utils.rnn.pack_padded_sequence(option, lengths,batch_first=True).to(device)
    #         option_outputs,option_hidden = gru_options(option_packed,option_hidden)
    #         option_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(option_outputs,batch_first=True)
    #         option_outputs = option_outputs[idx_unsort]
    #         print(option_outputs.size())
    #         print(option_outputs.max(1)[0].size())
    #         option_outputs = option_outputs.max(1)[0]
    #         scores = []
    #         for i in range(10):#range at batch size
    #             # if i != 0:
    #             #     break
    #             # print(context_outputs[i].size())
    #             # print(context_outputs[i].view(1,300).size())
    #             score = torch.mm(context_outputs[i].view(1,300), w)
    #             # print(score.size())
    #             score = torch.mm(score,option_outputs[i].view(300,1))
    #             # print(score.size())
    #             scores.append(score)
    #         # print(scores)
    #         print(torch.stack(scores,-1).view(10).size())
    #         scores = torch.stack(scores,-1).view(10)
    #         print(scores)
    #
    #         # score = torch.mm(context_outputs.view(10,300), w)
    #         # print(score.size())
    #         # score = torch.mm(score,option_outputs.view(300,10))
    #         # print(score.size())
    #         # logits2.append(score)
    #
    #         # logit = w2(context_outputs+option_outputs).sum(-1)
    #         # print(logit.size())
    #         logits.append(scores)
    #         # print(logit)
    #     logits = torch.stack(logits, 1)
    #     print(logits)
    #     print(logits.size())
    #     print(relu(logits))
    #     print(relu(logits).size())
    #     # print(logits)
    #     # logits2 = torch.stack(logits2, 1)
    #     # print(logits2.size())
    #
    #
    #         # output,hidden = gru_context(option,hidden)
    #         # print(hidden.size())
    #         # print(output.size())
    # print('model')
    for i,batch in enumerate(valid_dataloader):
        if i != 0 :
            break
        # print(i)
        # context = embedded(batch['context'].long().to(device))
        # options = embedded(batch['options'].long().to(device))
        # logits = model.forward(
        #     context.to(device),
        #     batch['context_lens'],
        #     options.to(device),
        #     batch['option_lens']
        # )
        # print(batch['context'].size())
        # print(batch['labels'])
        # print(batch['labels'].nonzero())
        # lengths, idx_sort = torch.sort(torch.Tensor(torch.randn(10,100)), dim=0, descending=True)
        # print(idx_sort.size[:][:10])
        # if 0 in idx_sort[:][:10]:
        #     print(yes)
        x = torch.randn(10,100)
        y, i = torch.topk(x, 10)
        print(batch['options'].size())
        print(i[0])
        one = torch.zeros(len(batch['id']), batch['options'].size(1)).scatter_(1,i.resize_(len(batch['id']),10), 1)
        # print(one)
        # i.resize_(len(batch['id']),10)
        # print(x)
        # print(i)
        print(one[0])
        print(batch['labels'].float()[0])
        print(torch.sum(one*batch['labels'].float()).item())
        # print(i.size())
        # print(logits.size())
        # print(logits)
