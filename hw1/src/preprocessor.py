import json
import logging
import multiprocessing
from multiprocessing import Pool
from dataset import DialogDataset
from tqdm import tqdm,trange

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

class Preprocessor:
    """

    Args:
        embedding_path (str): Path to the embedding to use.
    """
    def __init__(self, embedding):
        self.embedding = embedding
        self.logging = logging.getLogger(name=__name__)

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

    def sentence_to_indices(self, sentence):
        """ Convert sentence to its word indices.
        Args:
            sentence (str): One string.
        Return:
            indices (list of int): List of word indices.
        """
        # TODO
        # Hint: You can use `self.embedding`
        # pass

        tokens = self.tokenize(sentence)
        # print(tokens)
        words_idxs = [
                        self.embedding.to_index(word)
                        for word in tokens
        ]
        return  words_idxs
    def collect_words(self, data_path, n_workers=4):
        with open(data_path) as f:
            data = json.load(f)

        utterances = []
        for sample in data:
            utterances += (
                [message['utterance']
                 for message in sample['messages-so-far']]
                + [option['utterance']
                   for option in sample['options-for-next']]
            )
        utterances = list(set(utterances))
        chunks = [
            ' '.join(utterances[i:i + len(utterances) // n_workers])
            for i in range(0, len(utterances), len(utterances) // n_workers)
        ]
        # multiprocessing
        # with Pool(n_workers) as pool:
        #     chunks = pool.map_async(self.tokenize, chunks)
        #     words = set(sum(chunks.get(), []))
        # ----------------------------
        words = set(self.tokenize(chunks[0]))
        return words

    def get_dataset(self, data_path, n_workers=4, dataset_args={}):
        """ Load data and return Dataset objects for training and validating.

        Args:
            data_path (str): Path to the data.
            valid_ratio (float): Ratio of the data to used as valid data.
        """
        self.logging.info('loading dataset...')
        with open(data_path) as f:
            dataset = json.load(f)

        self.logging.info('preprocessing data...')

        #multiprocessing
        # results = [None] * n_workers
        # with Pool(processes=n_workers) as pool:
        #     for i in range(n_workers):
        #         batch_start = (len(dataset) // n_workers) * i
        #         if i == n_workers - 1:
        #             batch_end = len(dataset)
        #         else:
        #             batch_end = (len(dataset) // n_workers) * (i + 1)
        #
        #         batch = dataset[batch_start: batch_end]
        #         results[i] = pool.apply_async(self.preprocess_samples, [batch])
        #
        #         # When debugging, you'd better not use multi-thread.
        #         # results[i] = self.preprocess_dataset(batch, preprocess_args)
        #
        #     pool.close()
        #     pool.join()
        #
        # processed = []
        # for result in results:
        #     processed += result.get()
        #----------------------------
        processed = self.preprocess_samples(dataset)
        #----------------------------

        padding = self.embedding.to_index('</s>')
        return DialogDataset(processed, padding=padding, **dataset_args)

    def preprocess_samples(self, dataset):
        """ Worker function.

        Args:
            dataset (list of dict)
        Returns:
            list of processed dict.
        """
        processed = []
        for sample in tqdm(dataset,ncols=70):
            processed.append(self.preprocess_sample(sample))

        return processed

    def preprocess_sample(self, data):
        """
        Args:
            data (dict)
        Returns:
            dict
        """
        processed = {}
        processed['id'] = data['example-id']

        # process messages-so-far
        processed['context'] = []
        processed['speaker'] = []
        for message in data['messages-so-far']:
            processed['context'].append(
                self.sentence_to_indices(message['utterance'].lower())
            )

        # process options
        processed['options'] = []
        processed['option_ids'] = []

        # process correct options
        if 'options-for-correct-answers' in data:
            processed['n_corrects'] = len(data['options-for-correct-answers'])
            for option in data['options-for-correct-answers']:
                processed['options'].append(
                    self.sentence_to_indices(option['utterance'].lower())
                )
                processed['option_ids'].append(option['candidate-id'])
        else:
            processed['n_corrects'] = 0

        # process the other options
        for option in data['options-for-next']:
            if option['candidate-id'] in processed['option_ids']:
                continue

            processed['options'].append(
                self.sentence_to_indices(option['utterance'].lower())
            )
            processed['option_ids'].append(option['candidate-id'])

        return processed

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess and generate preprocessed pickle.")
    parser.add_argument('dest_dir', type=str,
                        help='[input] Path to the directory that .')
    parser.add_argument('--n_workers', type=int, default=4)
    args = parser.parse_args()
    return args

#for testing
if __name__ == '__main__':

    from embedding import Embedding

    words = {"I","have","apple","fish","banana","learing","adl"}
    print("start embedding")
    embedding = Embedding("../data/crawl-300d-2M.vec", words)
    preprocessor = Preprocessor(None)
    preprocessor.embedding = embedding
    sentence = "I have a apple ."
    print("end embedding")
    print(preprocessor.sentence_to_indices(sentence))
    print(embedding.word_dict['<unk>'])
    # print(config['embedding_vec_path'])
    print(embedding.word_dict)
