import torch
import numpy as np
from torch.autograd import Variable

class Rnn_net(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings,
                 similarity='inner_product',
                 n_layers=3
                ):
        super(Rnn_net, self).__init__()
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
        self.b = torch.nn.Bilinear(dim_embeddings, dim_embeddings, 1)

    def forward(self, context, context_lens, options, option_lens):
        B_size = context.size(0)
        ctx_hidden = self._init_hidden(B_size)
        lengths, idx_sort = torch.sort(torch.Tensor(context_lens), dim=0, descending=True)
        _,idx_unsort = torch.sort(idx_sort, dim=0)
        context = context[idx_sort]
        ctx_outputs = self._encoder(context,lengths,ctx_hidden,True)
        ctx_outputs = ctx_outputs[idx_unsort]
        ctx_outputs = ctx_outputs.max(1)[0]

        logits = []
        for i,option in enumerate(options.transpose(1,0)):
            opt_hidden = self._init_hidden(B_size)
            lengths, idx_sort = torch.sort(torch.Tensor(np.asarray(option_lens)[:,i]), dim=0, descending=True)
            _,idx_unsort = torch.sort(idx_sort, dim=0)
            option = option[idx_sort]
            opt_outputs = self._encoder(option,lengths,opt_hidden,False)
            opt_outputs = opt_outputs[idx_unsort]
            opt_outputs = opt_outputs.max(1)[0]
            scores = []
            score = self.b(ctx_outputs,opt_outputs).squeeze(-1)
            logits.append(score)
        logits = torch.stack(logits, 1)

        return logits
    def _init_hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers,
                             batch_size,
                             self.dim_embeddings
                            ).to(self.d)
        return Variable(hidden)
    def _encoder(self,sequences,lengths,hidden,context=True):
        packed = torch.nn.utils.rnn.pack_padded_sequence(sequences, lengths,batch_first=True)
        if context:
            outputs,hidden = self.gru_context(packed,hidden)
        else:
            outputs,hidden = self.gru_options(packed,hidden)
        outputs,_ = torch.nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)

        return outputs
