import torch


class Metrics:
    def __init__(self):
        self.name = 'Metric Name'

    def reset(self):
        pass

    def update(self, predicts, batch):
        pass

    def get_score(self):
        pass


class Recall(Metrics):
    """
    Args:
         ats (int): @ to eval.
         rank_na (bool): whether to consider no answer.
    """
    def __init__(self, at=10):
        self.at = at
        self.n = 0
        self.n_correct = 0
        self.name = 'Recall@{}'.format(at)

    def reset(self):
        self.n = 0
        self.n_correct = 0

    def update(self, predicts, batch):
        """
        Args:
            predicts (FloatTensor): with size (batch, n_samples).
            batch (dict): batch.
        """
        predicts = predicts.cpu()
        # TODO
        # This method will be called for each batch.
        # You need to
        # - increase self.n, which implies the total number of samples.
        # - increase self.n_corrects based on the prediction and labels
        #   of the batch.
        k = min(self.at,batch['options'].size()[1])
        B_size = batch['options'].size()[0]
        _, i = torch.topk(predicts, k)
        one = torch.zeros(B_size, batch['options'].size()[1]).scatter_(1,i.resize_(B_size,k), 1)
        self.n += B_size
        self.n_correct += torch.sum(one*batch['labels'].float()).item()
        # print(self.n_correct/self.n)
    def get_score(self):
        return self.n_correct / self.n

    def print_score(self):
        score = self.get_score()
        return '{:.2f}'.format(score)
