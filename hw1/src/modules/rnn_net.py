import torch

def load_pretrained(embedding_vecs):
    rows, cols = embedding_vecs.shape
    embedding = torch.nn.Embedding(num_embeddings=rows, embedding_dim=cols)
    embedding.weight = torch.nn.Parameter(embedding_vecs)
    embedding.weight.requires_grad = False
    return embedding


# class EncoderRNN(torch.nn.Module):
#     """
#
#     Args:
#
#     """
#
#     def __init__(self,embedding_vectors,device=None):
#         super(EncoderRNN, self).__init__()
#
#         if device is not None:
#             self.device = torch.device(device)
#         else:
#             self.device = torch.device('cuda:0' if torch.cuda.is_available()
#                                        else 'cpu')
#
#         #architecuture
#         self.embedding = load_pretrained(embedding_vectors)
#         self.embedding = self.embedding.to(self.device)
#         # self.rnn = nn.GRU()
#     def forward(self, context, context_lens, options, option_lens):
