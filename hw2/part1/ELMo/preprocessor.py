import nltk
import pickle
import torch
import random
import torch.nn as nn
from collections import Counter
from tqdm import tqdm

from sklearn.externals import joblib #need to delete

class Preprocessor:
    """

    Args:
        embedding_path (str): Path to the embedding to use.
    """
    def __init__(self, embedding):
        self.word_lexicon = {}
        self.char_lexicon = {}


    def tokenize(self, sentence):
        """ Tokenize a sentence.
        Args:
            sentence (str): One string.
        Return:
            indices (list of str): List of tokens in a sentence.
        """
        # TODO
        # pass

        token = word_tokenize(sentence)
        return token


#
def break_sent(sentence, max_sent_len):
    ret = []
    cur = 0
    l = len(sentence)
    while cur < l:
        if cur + max_sent_len + 5 >= l:
            ret.append(sentence[cur: l])
            break
        ret.append(sentence[cur: min(l, cur + max_sent_len)])
        cur += max_sent_len
    return ret

#
def chunks(l, n):
    n = max(1, n)
    return list(l[i:i+n] for i in range(0, len(l), n))

def get_vocab(dataset,min_count=3):
    word_count = Counter()
    char_count = Counter()
    for sentence in dataset:
        word_count.update(sentence)
        for word in sentence:
            char_count.update(list(word))

    word_count = list(word_count.items())
    word_count.sort(key=lambda x: x[1], reverse=True)

    for i, (word, count) in enumerate(word_count):
        if count < min_count:
            break
    word_count = word_count[:i]

    char_count = list(char_count.items())
    char_count.sort(key=lambda x: x[1], reverse=True)

    for i, (char, count) in enumerate(char_count):
        if count < 1000:
            break
    char_count = char_count[:i]
    return word_count,char_count

#
def create_one_batch(x, word2id, char2id, oov='<UNK>', pad='<PAD>', sort=True,for_embed = False):

    batch_size = len(x)
    lst = list(range(batch_size))
    if sort:
        lst.sort(key=lambda l: -len(x[l]))

    x = [x[i] for i in lst]
    lens = [len(x[i]) for i in lst]
    if for_embed:
        max_len = max(lens)
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

    # masks = [torch.LongTensor(batch_size, max_len).fill_(0), [], []]
    #
    # for i, x_i in enumerate(x):
    #     for j in range(len(x_i)):
    #         masks[0][i][j] = 1
    #         if j + 1 < len(x_i):
    #             masks[1].append(i * max_len + j)
    #         if j > 0:
    #             masks[2].append(i * max_len + j)
    #
    # assert len(masks[1]) <= batch_size * max_len
    # assert len(masks[2]) <= batch_size * max_len
    #
    # masks[1] = torch.LongTensor(masks[1])
    # masks[2] = torch.LongTensor(masks[2])

    return batch_w, batch_c, lens

#
def create_batches(x, batch_size, word2id, char2id, perm=None, shuffle=True, sort=False, use_cuda=False):

    lst = perm or list(range(len(x)))
    if shuffle:
        random.shuffle(lst)
    if sort:
        lst.sort(key=lambda l: -len(x[l]))

    x = [x[i] for i in lst]

    sum_len = 0.0
    batches_w, batches_c, batches_lens, batches_masks = [], [], [], []
    size = batch_size
    nbatch = (len(x) - 1) // size + 1
    for i in tqdm(range(nbatch),ncols=70):
        start_id, end_id = i * size, (i + 1) * size
        bw, bc, blens = create_one_batch(x[start_id: end_id], word2id, char2id, sort=sort)
        sum_len += sum(blens)
        batches_w.append(bw)
        batches_c.append(bc)
        batches_lens.append(blens)
        # batches_masks.append(bmasks)

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_w = [batches_w[i] for i in perm]
        batches_c = [batches_c[i] for i in perm]
        batches_lens = [batches_lens[i] for i in perm]
        # batches_masks = [batches_masks[i] for i in perm]

  #logging.info("{} batches, avg len: {:.1f}".format(nbatch, sum_len / len(x)))
    return batches_w, batches_c, batches_lens



if __name__ == '__main__':

    corpus_path = '../data/language_model/corpus_tokenized.txt'

    with open(corpus_path,encoding="utf-8") as f:
        rawdata = f.readlines()

    dataset = []
    rawdata = rawdata[:500000]
    for i in range(5):
        data = []
        # for line in tqdm(rawdata[:500000],ncols=70):
        for line in tqdm(rawdata[i*100000:(i+1)*100000],ncols=70):
            line = line.strip().split()
            line.insert(0,'<BOS>')
            line.append('<EOS>')
            cut = chunks(line,64)
            data = data + cut
        dataset = dataset + data
    # print(map(len,dataset[:20]))
    print(len(dataset))
    vocab,char = get_vocab(dataset)


    #word_lexicon
    word_lexicon = {}
    for special_word in ['<UNK>','<BOS>','<EOS>','<PAD>']:
        if special_word not in word_lexicon:
            word_lexicon[special_word] = len(word_lexicon)
    for word,c in vocab:
        if word not in word_lexicon:
            word_lexicon[word] = len(word_lexicon)
    with open('word_lexicon', 'wb') as f:
        pickle.dump(word_lexicon, f)
    #char_lexicon
    char_lexicon = {}
    for special_word in ['<UNK>','<BOS>','<EOS>','<PAD>','<eow>','<bow>']:
        if special_word not in char_lexicon:
            char_lexicon[special_word] = len(char_lexicon)
    for c,_ in char:
        if c not in char_lexicon:
            char_lexicon[c] = len(char_lexicon)

    with open('char_lexicon', 'wb') as f:
        pickle.dump(char_lexicon, f)

    train = create_batches(dataset, 64, word_lexicon, char_lexicon, use_cuda=True)

    #need to delete
    joblib.dump(train,'train_data')

    # with open('train_data50w','rb') as f:
    #     train = pickle.load(f)

    # train_w, train_c, train_lens, train_masks = train
    # print(len(train_w))
    # with open('train_data50w','wb') as f:
    #     pickle.dump(train, f)

    # train_w, train_c, train_lens, train_masks = train
    # print(train_w[0].size())
    # print(train_w[0][:5])
    # print(train_c[0].size())
    # print(train_c[0][:5])
