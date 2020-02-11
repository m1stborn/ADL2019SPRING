import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from char_embedding import CharEmbedding
class elmo_net(nn.Module):
    """
    """
    def __init__(self, num_embeddings, char_padidx,word_padidx,
                                vocab_size,batch_size=64):
        super().__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available()
                                       else 'cpu')

        self.batch_size = batch_size

        filters = [(1, 32), (2, 64), (3, 128), (4, 128), (5, 256), (6, 256), (7, 512)]

        self.vocab_size = vocab_size
        self.num_layer = 1
        self.char_padidx = char_padidx
        self.word_padidx = word_padidx
        # architecture

        self.char_cnn = CharEmbedding(num_embeddings,16,
                                      char_padidx,filters,2,512)
        #forward
        # self.f_char_cnn = CharEmbedding(num_embeddings,16,
        #                               padding_idx,filters,2,512)
        self.f_lstm = nn.LSTM(512, 2048, self.num_layer,batch_first=True)
        self.f_p = nn.Linear(2048,512)

        self.f_lstm2 = nn.LSTM(512, 2048, self.num_layer,batch_first=True)
        self.f_p2 = nn.Linear(2048,512)


        #
        # self.f_embedding = torch.nn.Embedding(vocab_size,
        #                                     300)
        # self.f_embedding.weight = torch.nn.Parameter(embedding)
        #

        #backword
        # self.b_char_cnn = CharEmbedding(num_embeddings,16,
        #                               padding_idx,filters,2,512)
        self.b_lstm = nn.LSTM(512, 2048, self.num_layer,batch_first=True)
        self.b_p = nn.Linear(2048,512)

        self.b_lstm2 = nn.LSTM(512, 2048, self.num_layer,batch_first=True)
        self.b_p2 = nn.Linear(2048,512)

        # softmax
        self.softmax = nn.AdaptiveLogSoftmaxWithLoss(512,self.vocab_size
                                                     ,cutoffs=[100,1000,10000])
        #
        # self.b_embedding = torch.nn.Embedding(vocab_size,
        #                                     300)
        # self.b_embedding.weight = torch.nn.Parameter(embedding)
        #
        #


    def forward(self, x, y):
        """
        x = [batch_szie,seq_len,max_chars]
        y = [batch_size,seq_len]
        """
        b_x = x.flip(1) #[batch_szie,seq_len,max_chars]
        b_y = y.flip(1) #[batch_size,seq_len]

        #forward
        # f_emb = self.f_char_cnn(x)#[batch_szie,seq_len,512]
        f_emb = self.char_cnn(x)#[batch_szie,seq_len,512]
        #1
        f_h_t,f_c_t = self._init_hidden(64)
        f_lstmout , (f_h_t,f_c_t) = self.f_lstm(f_emb)

        f_lstmout = self.f_p(f_lstmout)#[batch_szie,seq_len,512]
        #2
        f_h_t2,f_c_t2 = self._init_hidden(64)
        f_lstmout2 , (f_h_t2,f_c_t2) = self.f_lstm(f_lstmout)
        #
        f_lstmout2 = self.f_p2(f_lstmout2)#[batch_szie,seq_len,512]
        #
        f_out = f_lstmout2[:,:-1,:].contiguous().view(-1,f_lstmout2.size(-1))
                                #[seq_len-1*batch_size,hidden_size]
        f_ans = y[:,1:].contiguous().view(-1)
                                #[seq_len-1*batch_size]
        mask1 = f_ans.ne(self.word_padidx).type(torch.float).view(-1)
        f_out,f_loss = self.softmax(f_out,f_ans)
        f_loss = (-f_out*mask1).sum()/(mask1.sum()+1)
        #backword
        # b_emb = self.b_char_cnn(b_x)#[batch_szie,seq_len,512]
        b_emb = self.char_cnn(b_x)#[batch_szie,seq_len,512]
        #1
        b_h_t,b_c_t = self._init_hidden(64)
        b_lstmout , (b_h_t,b_c_t) = self.b_lstm(b_emb)

        b_lstmout = self.b_p(b_lstmout)#[batch_szie,seq_len,512]
        #2
        b_h_t2,b_c_t2 = self._init_hidden(64)
        b_lstmout2 , (b_h_t2,b_c_t2) = self.b_lstm(b_lstmout)

        b_lstmout2 = self.b_p2(b_lstmout2)#[batch_szie,seq_len,512]
        #
        b_out = b_lstmout2[:,:-1,:].contiguous().view(-1,b_lstmout2.size(-1))
                                #[seq_len-1*batch_size,hidden_size]

        b_ans = b_y[:,1:].contiguous().view(-1)
                                #[seq_len-1*batch_size]
        mask2 = b_ans.ne(self.word_padidx).type(torch.float).view(-1)
        b_out,b_loss = self.softmax(b_out,f_ans)
        b_loss = (-b_out*mask2).sum()/(mask2.sum()+1)

        loss = (f_loss+b_loss)/2

        out_char_cnn = torch.cat((f_emb[:,1:-1,:],b_emb[:,1:-1,:]),-1)
        out_lstm1 = torch.cat((f_lstmout[:,1:-1,:],b_lstmout[:,1:-1,:]),-1)
        out_lstm2 = torch.cat((f_lstmout2[:,1:-1,:],b_lstmout2[:,1:-1,:]),-1)

        ctx_embed= torch.stack((out_char_cnn,out_lstm2,out_lstm1),2)

        return loss, ctx_embed

    #
    def _init_hidden(self,batch_size):
        hidden = torch.zeros(self.num_layer,
                     batch_size,
                     2048,
                    ).to(self.device)
        #
        cell = torch.zeros(self.num_layer,
                     batch_size,
                     2048,
                    ).to(self.device)

        return Variable(hidden),Variable(cell)
