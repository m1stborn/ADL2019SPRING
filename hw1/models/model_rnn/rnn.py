import torch
import numpy as np
from torch.autograd import Variable

class ExampleNet(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings,
                 similarity='inner_product',
                 n_layers=3
                ):
        super(ExampleNet, self).__init__()
        #ExampleNet
        # self.mlp = torch.nn.Sequential(
        #     torch.nn.Linear(dim_embeddings, 256),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(256, 256)
        # )
        #---------------------
        #RNN
        self.dim_embeddings = dim_embeddings
        self.n_layers = n_layers

        self.d = torch.device('cuda:0' if torch.cuda.is_available()
                                       else 'cpu')

        #architecuture
        self.gru_context = torch.nn.GRU(dim_embeddings,dim_embeddings,self.n_layers,batch_first=True)
        self.gru_options = torch.nn.GRU(dim_embeddings,dim_embeddings,self.n_layers,batch_first=True)
        self.w = torch.autograd.Variable(torch.randn(dim_embeddings,dim_embeddings)
                                         , requires_grad=True).to(self.d)
        # self.relu = torch.nn.ReLU()
        self.b = torch.nn.Bilinear(dim_embeddings, dim_embeddings, 1)
    # def forward(self, context, context_lens, options, option_lens):
    #     B_size = context.size(0)
    #     #ExampleNet
    #     # context = self.mlp(context).max(1)[0]
    #     # print(context.size())
    #     # context = self.mlp(context)
    #     # print(context.size())
    #     # context = context.max(1)[0]
    #     # logits = []
    #     # print(context.size())
    #     # print(options.transpose(1, 0).size())
    #     # for i, option in enumerate(options.transpose(1, 0)):
    #     #     option = self.mlp(option).max(1)[0]
    #     #     if i == 0:
    #     #         print(option.size())
    #     #     logit = ((context - option) ** 2).sum(-1)
    #     #     # print(logit.size())
    #     #     logits.append(logit)
    #     # logits = torch.stack(logits, 1)
    #     # return logits
    #     #---------------------
    #     ctx_hidden = self._init_hidden(B_size)
    #     lengths, idx_sort = torch.sort(torch.Tensor(context_lens), dim=0, descending=True)
    #     _,idx_unsort = torch.sort(idx_sort, dim=0)
    #     context = context[idx_sort]
    #     ctx_packed = torch.nn.utils.rnn.pack_padded_sequence(context, lengths,batch_first=True)
    #     ctx_outputs,ctx_hidden = self.gru_context(ctx_packed,ctx_hidden)
    #     ctx_outputs,_ = torch.nn.utils.rnn.pad_packed_sequence(ctx_outputs,batch_first=True)
    #     ctx_outputs = ctx_outputs[idx_unsort]
    #     ctx_outputs = ctx_outputs.max(1)[0]
    #
    #     logits = []
    #     for i,option in enumerate(options.transpose(1,0)):
    #         opt_hidden = self._init_hidden(B_size)
    #         lengths, idx_sort = torch.sort(torch.Tensor(np.asarray(option_lens)[:,i]), dim=0, descending=True)
    #         _,idx_unsort = torch.sort(idx_sort, dim=0)
    #         option = option[idx_sort]
    #         opt_packed = torch.nn.utils.rnn.pack_padded_sequence(option, lengths,batch_first=True)
    #         opt_outputs,opt_hidden = self.gru_options(opt_packed,opt_hidden)
    #         opt_outputs,_ = torch.nn.utils.rnn.pad_packed_sequence(opt_outputs,batch_first=True)
    #         opt_outputs = opt_outputs[idx_unsort]
    #         opt_outputs = opt_outputs.max(1)[0]
    #         scores = []
    #
    #         for j in range(B_size):
    #             score = torch.mm(ctx_outputs[j].view(1,self.dim_embeddings),self.w)
    #             score = torch.mm(score,opt_outputs[j].view(self.dim_embeddings,1))
    #             scores.append(score)
    #
    #         scores = torch.stack(scores,-1).view(B_size)
    #         logits.append(scores)
    #     logits = torch.stack(logits, 1)
    #     # logits = self.relu(logits)
    #
    #     return logits
    #     # pass
    def forward(self, context, context_lens, options, option_lens):
        B_size = context.size(0)
        ctx_hidden = self._init_hidden(B_size)
        lengths, idx_sort = torch.sort(torch.Tensor(context_lens), dim=0, descending=True)
        _,idx_unsort = torch.sort(idx_sort, dim=0)
        context = context[idx_sort]
        # ctx_packed = torch.nn.utils.rnn.pack_padded_sequence(context, lengths,batch_first=True)
        # ctx_outputs,ctx_hidden = self.gru_context(ctx_packed,ctx_hidden)
        # ctx_outputs,_ = torch.nn.utils.rnn.pad_packed_sequence(ctx_outputs,batch_first=True)
        ctx_outputs = self._encoder(context,lengths,ctx_hidden,True)
        ctx_outputs = ctx_outputs[idx_unsort]
        ctx_outputs = ctx_outputs.max(1)[0]

        logits = []
        for i,option in enumerate(options.transpose(1,0)):
            # if i != 0:
            #     break
            opt_hidden = self._init_hidden(B_size)
            lengths, idx_sort = torch.sort(torch.Tensor(np.asarray(option_lens)[:,i]), dim=0, descending=True)
            _,idx_unsort = torch.sort(idx_sort, dim=0)
            option = option[idx_sort]
            # opt_packed = torch.nn.utils.rnn.pack_padded_sequence(option, lengths,batch_first=True)
            # opt_outputs,opt_hidden = self.gru_options(opt_packed,opt_hidden)
            # opt_outputs,_ = torch.nn.utils.rnn.pad_packed_sequence(opt_outputs,batch_first=True)
            opt_outputs = self._encoder(option,lengths,opt_hidden,False)
            opt_outputs = opt_outputs[idx_unsort]
            opt_outputs = opt_outputs.max(1)[0]
            scores = []
            score = self.b(ctx_outputs,opt_outputs).squeeze(-1)
            # print(score.squeeze(-1).size())
            logits.append(score)
            # scores = torch.stack(scores,-1).view(B_size)
            # logits.append(scores)
        # print(logits.size())
        logits = torch.stack(logits, 1)

        return logits
    def _init_hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers,
                             batch_size,
                             self.dim_embeddings
                            ).to(self.d)
        return Variable(hidden)
    def _encoder(self,sequences,hidden,lengths,context=True):
        packed = torch.nn.utils.rnn.pack_padded_sequence(sequences,lengths,batch_first=True)
        if context:
            outputs,hidden = self.gru_context(packed,hidden)
        else:
            outputs,hidden = self.gru_options(packed,hidden)
        outputs,_ = torch.nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)

        return outputs
