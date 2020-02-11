import numpy as np
import pickle
import torch
# from preprocessor import create_one_batch
# from elmo_net import elmo_net
class Embedder:
    """
    The class responsible for loading a pre-trained ELMo model and provide the ``embed``
    functionality for downstream BCN model.

    You can modify this class however you want, but do not alter the class name and the
    signature of the ``embed`` function. Also, ``__init__`` function should always have
    the ``ctx_emb_dim`` parameter.
    """

    def __init__(self, n_ctx_embs, ctx_emb_dim):
        """
        The value of the parameters should also be specified in the BCN model config.
        """
        self.n_ctx_embs = n_ctx_embs
        self.ctx_emb_dim = ctx_emb_dim
        # TODO
        model_path = './Elmo/model/model_17.36/model.pkl.1.10000'

        with open('./Elmo/word_lexicon', 'rb') as f:
            word_lexicon = pickle.load(f)
        with open('./Elmo/char_lexicon', 'rb') as f:
            char_lexicon = pickle.load(f)

        self.word_lexicon = word_lexicon
        self.char_lexicon = char_lexicon
        # torch.nn.Module.dump_patches = True
        self.model = torch.load(model_path)
        self.drop = torch.nn.Dropout(0.1)
    def __call__(self, sentences, max_sent_len):
        """
        Generate the contextualized embedding of tokens in ``sentences``.

        Parameters
        ----------
        sentences : ``List[List[str]]``
            A batch of tokenized sentences.
        max_sent_len : ``int``
            All sentences must be truncated to this length.

        Returns
        -------
        ``np.ndarray``
            The contextualized embedding of the sentence tokens.

            The ndarray shape must be
            ``(len(sentences), min(max(map(len, sentences)), max_sent_len), self.n_ctx_embs, self.ctx_emb_dim)``
            and dtype must be ``np.float32``.
        """
        # TODO
        data = []
        for i in range(len(sentences)):
            line = sentences[i][:]
            # if len(sentences[i]) > max_sent_len-2:
            #     sentences[i] = sentences[i][:max_sent_len-2]
            # sentences[i].insert(0,'<BOS>')
            # sentences[i].append('<EOS>')
            line.insert(0,'<BOS>')
            line.append('<EOS>')
            data.append(line)

        batch = create_one_batch(data,self.word_lexicon,self.char_lexicon,
                             sort=False,for_embed=True)
        bw,bc,blens,bmasks = batch

        self.model.eval()

        _ , output = self.model.forward(bc.to('cuda:0'),
                                    bw.to('cuda:0'))
        # output = self.model.contex_embedding(bc.to('cuda:0'),
        #                             bw.to('cuda:0'))
        # print(min(max(map(len, sentences)))
        # output = self.drop(output)

        return output

        # return np.zeros(
        #     (len(sentences), min(max(map(len, sentences)), max_sent_len), self.n_ctx_embs, self.ctx_emb_dim), dtype=np.float32)
#
def create_one_batch(x, word2id, char2id, oov='<UNK>', pad='<PAD>', sort=True,for_embed = False):

    batch_size = len(x)
    lst = list(range(batch_size))
    if sort:
        lst.sort(key=lambda l: -len(x[l]))

    x = [x[i] for i in lst]
    lens = [len(x[i]) for i in lst]
    if for_embed:
        max_len = min(max(lens),64)
        # print(max_len)
    else:
        max_len = 64

    if word2id is not None:
        oov_id, pad_id = word2id.get(oov, None), word2id.get(pad, None)
        assert oov_id is not None and pad_id is not None
        batch_w = torch.LongTensor(batch_size, max_len).fill_(pad_id)
        for i, x_i in enumerate(x):
            for j, x_ij in enumerate(x_i):
                batch_w[i][j] = word2id.get(x_ij, oov_id)
    else:
        batch_w = None

    if char2id is not None:
        bow_id, eow_id, oov_id, pad_id = char2id.get('<eow>', None), char2id.get('<bow>', None), char2id.get(oov, None), char2id.get(pad, None)

        assert bow_id is not None and eow_id is not None and oov_id is not None and pad_id is not None

        # if config['token_embedder']['name'].lower() == 'cnn':
        max_chars = 16
        # assert max([len(w) for i in lst for w in x[i]]) + 2 <= max_chars
        # elif config['token_embedder']['name'].lower() == 'lstm':
            # max_chars = max([len(w) for i in lst for w in x[i]]) + 2  # counting the <bow> and <eow>

        batch_c = torch.LongTensor(batch_size, max_len, max_chars).fill_(pad_id)

        for i, x_i in enumerate(x):
            for j, x_ij in enumerate(x_i):
                batch_c[i][j][0] = bow_id
                if x_ij == '<BOS>' or x_ij == '<EOS>':
                    batch_c[i][j][1] = char2id.get(x_ij)
                    # batch_c[i][j][2] = eow_id
                else:
                    for k, c in enumerate(x_ij):
                        if k+1>=max_chars:
                            break
                        batch_c[i][j][k + 1] = char2id.get(c, oov_id)
                        # batch_c[i][j][len(x_ij) + 1] = eow_id
    else:
        batch_c = None

    masks = [torch.LongTensor(batch_size, max_len).fill_(0), [], []]

    for i, x_i in enumerate(x):
        for j in range(len(x_i)):
            masks[0][i][j] = 1
            if j + 1 < len(x_i):
                masks[1].append(i * max_len + j)
            if j > 0:
                masks[2].append(i * max_len + j)

    assert len(masks[1]) <= batch_size * max_len
    assert len(masks[2]) <= batch_size * max_len

    masks[1] = torch.LongTensor(masks[1])
    masks[2] = torch.LongTensor(masks[2])

    return batch_w, batch_c, lens, masks
