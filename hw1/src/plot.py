import torch
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class rnn_attn_net(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings,
                 similarity='inner_product',
                 n_layers=3
                ):
        super(rnn_attn_net, self).__init__()
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
        # self.w = torch.autograd.Variable(torch.randn(dim_embeddings,dim_embeddings)
        #                                  , requires_grad=True).to(self.d)
        self.b = torch.nn.Bilinear(dim_embeddings, dim_embeddings, 1)
        self.attn = Attn('general',self.dim_embeddings)
        self.gru_attn = torch.nn.GRU(dim_embeddings,dim_embeddings,self.n_layers,batch_first=True)
    def forward(self, context, context_lens, options, option_lens):
        B_size = context.size(0)
        ctx_hidden = self._init_hidden(B_size)
        lengths, idx_sort = torch.sort(torch.Tensor(context_lens), dim=0, descending=True)
        _,idx_unsort = torch.sort(idx_sort, dim=0)
        context = context[idx_sort]
        ctx_outputs,ctx_hidden = self._encoder(context,lengths,ctx_hidden,'context')
        ctx_outputs = ctx_outputs[idx_unsort]
        # attn_weights = self.attn(ctx_outputs.transpose(1,0))
        ctx_pooling = ctx_outputs.max(1)[0]
        logits = []
        for i,option in enumerate(options.transpose(1,0)):
            if i != 0:
                break
            opt_hidden = self._init_hidden(B_size)
            opt_outputs,opt_hidden = self.gru_options(option,opt_hidden)

            ctxs = []
            weights = []
            for last_hidden in opt_outputs.transpose(1,0):
                weight = self.attn(last_hidden,ctx_outputs).unsqueeze(-1)
                # print(weight.size())
                weights.append(weight.squeeze(-1))
                ctx = ctx_outputs.transpose(2,1).bmm(weight).squeeze(-1)
                ctxs.append(ctx)
            #
            weights = torch.stack(weights,1)


            ax = plot_heatmap(0,0,weights[0,:,:].detach().cpu().numpy())

            ctxs = torch.stack(ctxs, 1)
            ctxs_cat_opt_outputs = torch.cat((ctxs,opt_outputs),1)
            att_hidden = self._init_hidden(B_size)

            lengths, idx_sort = torch.sort(torch.Tensor(np.asarray(option_lens)[:,i]), dim=0, descending=True)
            _,idx_unsort = torch.sort(idx_sort, dim=0)
            ctxs_cat_opt_outputs = ctxs_cat_opt_outputs[idx_sort]

            attn_outputs,attn_hidden = self._encoder(ctxs_cat_opt_outputs,lengths,att_hidden,'attn')
            attn_outputs = attn_outputs[idx_unsort]
            # print(attn_outputs.size())
            attn_outputs = attn_outputs.max(1)[0]
            # print(attn_outputs.size())

            score = self.b(ctx_pooling,attn_outputs).squeeze(-1)
            # print(score.size())

            logits.append(score)
        #
        logits = torch.stack(logits, 1)
        # print(logits.size())
        return logits,ax
    def _init_hidden(self,batch_size):
        hidden = torch.zeros(self.n_layers,
                             batch_size,
                             self.dim_embeddings
                            ).to(self.d)
        return Variable(hidden)
    def _encoder(self,sequences,lengths,hidden,layer):
        packed = torch.nn.utils.rnn.pack_padded_sequence(sequences, lengths,batch_first=True)
        # print(lengths)
        if layer == 'context' :
            outputs,hidden = self.gru_context(packed,hidden)
        elif layer == 'options':
            outputs,hidden = self.gru_options(packed,hidden)
        elif layer == 'attn':
            outputs,hidden = self.gru_attn(packed,hidden)
        outputs,hidden = torch.nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)
        # print(hidden)
        return outputs,hidden
#
import matplotlib.pyplot as plt
def plot_heatmap(src, trg, scores):

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(scores, cmap='viridis')

    # ax.set_xticklabels(trg, minor=False, rotation='vertical')
    # ax.set_yticklabels(src, minor=False)

    # put the major ticks at the middle of each cell
    # and the x-ticks on top
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(scores.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(scores.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()

    plt.colorbar(heatmap)
    # plt.show()
    return ax

class Attn(torch.nn.Module):
    def __init__(self, method='concat', hidden_size=300,mlp=False):
        super(Attn, self).__init__()
        self.d = torch.device('cuda:0' if torch.cuda.is_available()
                                       else 'cpu')
        self.method = method

        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")

        self.hidden_size = hidden_size
        #
        if method == 'dot':
          pass
        if self.method == 'general':
            self.Wa = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        elif self.method == 'concat':
            self.Wa = torch.nn.Linear(hidden_size, hidden_size, bias=False)
            self.Va = torch.nn.Parameter(torch.FloatTensor(batch_size, hidden_size))
        #
        self.mlp = mlp
        if mlp:
            self.phi = torch.nn.Linear(hidden_size, hidden_size, bias=False)
            self.psi = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, last_hidden, encoder_outputs, seq_len=None):
        #last_hidden = (batch_size,dim_embeddings),encoder_outputs = (batch_size,seq_len,dim_embeddings)
        if self.mlp:
            last_hidden = self.phi(last_hidden)
            encoder_outputs = self.psi(encoder_outputs)

        batch_size, seq_lens, _ = encoder_outputs.size()
        attention_energies = self._score(last_hidden, encoder_outputs, self.method)

        return F.softmax(attention_energies, -1)
    def _score(self, last_hidden, encoder_outputs, method):
        """
        Computes an attention score
        :param last_hidden: (batch_size, hidden_dim)
        :param encoder_outputs: (batch_size, max_time, hidden_dim)
        :param method: str (`dot`, `general`, `concat`, `bahdanau`)
        :return: a score (batch_size, max_time)
        """

        assert encoder_outputs.size()[-1] == self.hidden_size

        if method == 'dot':
            last_hidden = last_hidden.unsqueeze(-1)
            return encoder_outputs.bmm(last_hidden).squeeze(-1)

        elif method == 'general':
            x = self.Wa(last_hidden)
            x = x.unsqueeze(-1)
            return encoder_outputs.bmm(x).squeeze(-1)

        elif method == "concat":
            x = last_hidden.unsqueeze(1)
            x = F.tanh(self.Wa(torch.cat((x, encoder_outputs), 1)))
            return x.bmm(self.va.unsqueeze(2)).squeeze(-1)

        # elif method == "bahdanau":
        #     x = last_hidden.unsqueeze(1)
        #     out = F.tanh(self.Wa(x) + self.Ua(encoder_outputs))
        #     return out.bmm(self.va.unsqueeze(2)).squeeze(-1)
