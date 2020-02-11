import pickle
import torch
from tqdm import tqdm

class Predictor():
    def __init__(self,
                 model,
                 batch_size=64,
                 max_epochs=3,
                 device=None,
                 learning_rate=1e-3,
                 max_iters_in_epoch=1e20,
                 grad_accumulate_steps=1):
        #hyper parameter
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.max_iters_in_epoch = max_iters_in_epoch
        self.grad_accumulate_steps = grad_accumulate_steps

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available()
                                       else 'cpu')

        self.epoch = 0
        self.train_loss = []
        self.valid_loss = []
        #model
        self.model = model
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)
    #

    def fit_epoch(self,x,y,save_path,training):
        if training:
            print('training %i' % self.epoch)
        else:
            print('evaluating %i' % self.epoch)

        self.model.train(training)

        trange = tqdm(enumerate(zip(x,y)),
                  total=len(x),
                  ncols=70,
                 )
        loss = 0

        for i,(batch_x,batch_y) in trange:

            batch_loss,_ = self.model.forward(batch_x.to(self.device),
                                            batch_y.to(self.device)
                                            )
            if training:
                self.optimizer.zero_grad()
                batch_loss.backward()
                self.optimizer.step()

            loss+=batch_loss.item()
            trange.set_postfix(
                loss=loss / (i + 1),
                )

            if i%5000==0 and training:
                self.save('{}.{}.{}'.format(save_path,self.epoch,i))
            if training:
                self.train_loss.append(loss/(i+1))
            else:
                self.valid_loss.append(loss/(i+1))
    
        if training:
            self.epoch+=1

    #
    def save(self, path):
        torch.save(self.model,path)

    #
    def load(self, path):
        torch.nn.Module.dump_patches = True
        self.model = torch.load(path)
        self.model.eval()
