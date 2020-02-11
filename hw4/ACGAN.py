import torch, time, os, pickle,argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utlis import *
from torch.nn.utils import spectral_norm
import torchvision
from torch import Tensor
import torch.autograd as autograd
import scipy.misc
class generator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, input_dim=100, output_dim=1, input_size=128, class_num=15):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.up_num = 32 #2^up num

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim + self.class_num, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // self.up_num) * (self.input_size // self.up_num)),
            nn.BatchNorm1d(128 * (self.input_size // self.up_num) * (self.input_size // self.up_num)),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(128, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            #
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            #
            nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
            nn.Tanh(),
        )

        initialize_weights(self)

    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // self.up_num), (self.input_size // self.up_num))
        # print(self.deconv2(x).size())
        x = self.deconv(x)
        # print(x.size())

        return x

class discriminator(nn.Module):
    # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
    # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
    def __init__(self, input_dim=1, output_dim=1, input_size=128, class_num=15):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size
        self.class_num = class_num

        self.up_num = 16 #2^up num

        self.conv = nn.Sequential(
            spectral_norm(nn.Conv2d(self.input_dim, 64, 4, 2, 1)),
            # nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            #
            spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            #
            spectral_norm(nn.Conv2d(256, 128, 4, 2, 1)),
            # nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.fc1 =nn.Sequential(
            spectral_norm(nn.Linear(128 * (self.input_size // self.up_num) * (self.input_size // self.up_num), 1024)),
            # nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
        )
        self.dc = nn.Sequential(
            spectral_norm(nn.Linear(1024, self.output_dim)),
            # nn.Linear(1024, self.output_dim),
            # nn.Sigmoid(),
        )
        self.cl = nn.Sequential(
            spectral_norm(nn.Linear(1024, self.class_num)),
            # nn.Linear(1024, self.class_num),
        )
        self.sf = nn.Softmax(dim=1)
        #
        self.hair = nn.Sequential(
            spectral_norm(nn.Linear(1024, 6)),
            nn.Softmax(dim=1),
        )
        self.eyes = nn.Sequential(
            spectral_norm(nn.Linear(1024, 4)),
            nn.Softmax(dim=1),
        )
        self.face = nn.Sequential(
            spectral_norm(nn.Linear(1024, 3)),
            nn.Softmax(dim=1),
        )
        self.glass = nn.Sequential(
            spectral_norm(nn.Linear(1024, 2)),
            nn.Softmax(dim=1),
        )
        initialize_weights(self)

    def forward(self, input):
        x = self.conv(input)
        x = x.view(-1, 128 * (self.input_size // self.up_num) * (self.input_size // self.up_num))
        # print(x.size())
        x = self.fc1(x)
        d = self.dc(x)
        # c = self.cl(x)
        #
        # c[:,0:6] = self.sf(c[:,0:6].clone())
        # c[:,6:10] = self.sf(c[:,6:10].clone())
        # c[:,10:13] = self.sf(c[:,10:13].clone())
        # c[:,13:] = self.sf(c[:,13:].clone())
        h = self.hair(x)
        e = self.eyes(x)
        f = self.face(x)
        g = self.glass(x)
        a = torch.cat([h,e,f,g],1)
        # print(a.size())
        return d, a
#

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
#


class ACGAN(object):
    """docstring for ACGAN"""
    def __init__(self,epoch=1000,batch_size=128,dataset=None,model_dir='2340',data_loader=None,
                label_file = None,
                ):
        # pass

        self.epoch = epoch
        self.batch_size = batch_size
        self.data = dataset
        # self.data_loader =DataLoader(dataset, batch_size=self.batch_size,
        #                                      shuffle=True, num_workers=1,drop_last=True)
        self.data_loader = data_loader
        self.lrG = 0.0002
        self.lrD = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        self.device = torch.device('cuda:0' if torch.cuda.is_available()
                                   else 'cpu')

        self.z_dim = 100
        self.output_dim = 3     #RGB channels
        self.input_size = 128   # 3*128*128
        self.class_num = 15     #label attr num

        self.G = generator(input_dim=self.z_dim, output_dim=self.output_dim, input_size=self.input_size).to(self.device)
        self.D = discriminator(input_dim=3, output_dim=1, input_size=self.input_size).to(self.device)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=self.lrG, betas=(self.beta1, self.beta2))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=self.lrD, betas=(self.beta1, self.beta2))

        self.BCE_loss = nn.BCELoss().to(self.device)
        self.CE_loss = nn.CrossEntropyLoss().to(self.device)

        self.criterion = torch.nn.BCEWithLogitsLoss()
        # create samples
        self.sample_num = self.class_num**2

        z =  torch.rand((self.sample_num,self.z_dim)).to(self.device)
        z.data.normal_(0,1)
        # ber = np.random.binomial(1,0.5,(self.sample_num, self.z_dim))
        # z = torch.from_numpy(ber).to(self.device,dtype=torch.float)
        self.sample_z_ = z

        sampels = self.read_samples(txt=label_file).to(self.device)
        self.sample_y_ = sampels[:self.sample_num]

        self.base_dir = model_dir
        self.model_dir = os.path.join('./model/',model_dir)
        self.seed = 60229

    def create_gen_labels(self):

        label1 = torch.LongTensor(self.batch_size, 1).random_() % 6
        label2 = torch.LongTensor(self.batch_size, 1).random_() % 4
        label3 = torch.LongTensor(self.batch_size, 1).random_() % 3
        label4 = torch.LongTensor(self.batch_size, 1).random_() % 2

        a1 = torch.zeros(self.batch_size, 6).scatter_(1, label1, 1)
        a2 = torch.zeros(self.batch_size, 4).scatter_(1, label2, 1)
        a3 = torch.zeros(self.batch_size, 3).scatter_(1, label3, 1)
        a4 = torch.zeros(self.batch_size, 2).scatter_(1, label4, 1)
        gen_labels = torch.cat([a1,a2,a3,a4], 1)

        return gen_labels.to(self.device)

    def train(self,debug=0):
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []

        self.y_real_, self.y_fake_ = torch.ones(self.batch_size, 1).to(self.device), torch.zeros(self.batch_size, 1).to(self.device)


        for epoch in range(self.epoch):
            if epoch != 0 and debug:
                break
            trange = tqdm(enumerate(self.data_loader),
                          total=len(self.data_loader),
                          desc="epoch {}".format(epoch),
                          ncols=70,
                         )
            for i,(batch_x, batch_y) in trange:

                batch_x,batch_y = batch_x.to(self.device) , batch_y.to(self.device)
                gen_labels = self.create_gen_labels()
                # gen_labels= batch_y.clone()

                z_ = torch.rand((self.batch_size, self.z_dim)).to(self.device)
                # ber = np.random.binomial(1,0.5,(self.batch_size, self.z_dim))
                # z_ = torch.from_numpy(ber).to(self.device,dtype=torch.float)


                if i != 0 and debug:
                    break
                #Discriminator
                # train with real
                self.D_optimizer.zero_grad()

                D_real, C_real = self.D(batch_x)

                # D_real_loss_label = self.criterion(D_real, self.y_real_)
                D_real_aux = self.catgorical_loss(C_real,batch_y)
                # D_real_loss = (D_real_loss_label + D_real_aux)/2
                # D_real_loss = D_real_loss_label + D_real_aux
                D_real_loss = -torch.mean(D_real)


                # train with fake
                z_.data.normal_(0,1)
                # ber = np.random.binomial(1,0.5,(self.batch_size, self.z_dim))
                # z_ = torch.from_numpy(ber).to(self.device,dtype=torch.float)
                # batch_y.data.uniform_(to=1)

                # G_ = self.G(z_,gen_labels)
                G_ = self.G(z_,batch_y)
                D_fake, C_fake = self.D(G_)

                # D_fake_loss_label = self.criterion(D_fake,self.y_fake_)
                # D_fake_aux = self.catgorical_loss(C_fake,gen_labels)
                D_fake_aux = self.catgorical_loss(C_fake,batch_y)

                # D_fake_loss = (D_fake_loss_label + D_fake_aux)/2
                # D_fake_loss = D_fake_loss_label + D_fake_aux
                D_fake_loss = torch.mean(D_fake)
                # TO DO:gradient penalty

                # D_loss = (D_real_loss + D_fake_loss)/2
                # D_loss = D_real_loss + D_fake_loss

                D_loss = D_real_loss + D_fake_loss + D_real_aux

                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                #Generator
                z_.data.normal_(0,1)
                # ber = np.random.binomial(1,0.5,(self.batch_size, self.z_dim))
                # z_ = torch.from_numpy(ber).to(self.device,dtype=torch.float)
                # batch_y.data.uniform_(to=1)
                self.G_optimizer.zero_grad()


                # G_ = self.G(z_, gen_labels)
                G_ = self.G(z_, batch_y)
                D_fake_g, C_fake_c = self.D(G_)

                # G_loss_label = self.criterion(D_fake,self.y_real_)
                # G_loss_aux = self.catgorical_loss(C_fake_c,gen_labels)
                G_loss_aux = self.catgorical_loss(C_fake_c,batch_y)

                # G_loss = G_loss_label*0.5 + G_loss_aux
                # G_loss = G_loss_label + G_loss_aux
                G_loss = -torch.mean(D_fake_g) + G_loss_aux

                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()


                trange.set_postfix(
                    G_loss = self.train_hist['G_loss'][-1],
                    D_loss = self.train_hist['D_loss'][-1],
                    G_label = G_loss.item(),
                    G_tag = G_loss_aux.item()
                )

            with torch.no_grad():
                self.visualize_result((epoch+1),self.model_dir)
            if epoch%1 ==0:
                self.save(self.model_dir,(epoch+1))
    def catgorical_loss(self,pred,target):
        #4 attrs
        hair = self.CE_loss(pred[:,0:6],torch.max(target[:,0:6], 1)[1])
        eyes = self.CE_loss(pred[:,6:10],torch.max(target[:,6:10], 1)[1])
        face = self.CE_loss(pred[:,10:13],torch.max(target[:,10:12], 1)[1])
        glass = self.CE_loss(pred[:,13:],torch.max(target[:,13:], 1)[1])

        loss = hair/6 + eyes/4 + face/3 + glass/2
        return loss
    def save(self,path,epoch):
        # print(path)
        if not os.path.exists(path):
            os.makedirs(path)

        with open(os.path.join(path,'log.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

        torch.save(self.G.state_dict(), os.path.join(path, 'ACGAN_G.{}.ckpt'.format(epoch)))
        torch.save(self.D.state_dict(), os.path.join(path, 'ACGAN_D.{}.ckpt'.format(epoch)))

        # pass
    def load(self,path=None,epoch=1):
        # path = self.model_dir
        # self.G.load_state_dict(torch.load(os.path.join(path, 'ACGAN_G.{}.ckpt'.format(epoch))))
        # self.D.load_state_dict(torch.load(os.path.join(path, 'ACGAN_D.{}.ckpt'.format(epoch))))
        self.G.load_state_dict(torch.load('ACGAN_G.{}.ckpt'.format(epoch)))
        self.D.load_state_dict(torch.load('ACGAN_D.{}.ckpt'.format(epoch)))
        # pass
    def visualize_result(self,epoch,path=None):
        # print(self.sample_z_.size())
        self.G.eval()

        if not os.path.exists(path):
            os.makedirs(path)
        #
        image_frame_dim = int(np.floor(np.sqrt(self.sample_num)))
        #
        samples = self.G(self.sample_z_, self.sample_y_)

        #denorm
        # transform = transforms.Normalize(mean=(-0.5/0.5, -0.5/0.5, -0.5/0.5), std=(1/0.5, 1/0.5, 1/0.5))
        # for i in range(samples.size(0)):
        #     samples[i] = transform(samples[i])

        if torch.cuda.is_available():
            samples = samples.cpu().data.permute(0, 2, 3, 1).numpy()
        else:
            samples = samples.data.permute(0, 2, 3, 1).numpy()

        save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim]
                          ,path + '/_epoch%03d' % epoch + '.png')
    def read_samples(self,txt="./sample_test/sample_human_testing_labels.txt"):
        imgs = []
        with open(txt) as f:
            lines = f.readlines()[2:]
        #
        for line in lines:
            attr = line.split()
            imgs.append(list(map(int, attr)) )
        samples = torch.FloatTensor(imgs)

        return torch.FloatTensor(imgs)
    def generate_samples(self,save_dir="./test_result",epoch=None,label_file="./sample_test/sample_fid_testing_labels.txt"):
        samples = self.read_samples(label_file).to(self.device)
        # z =  torch.rand(self.z_dim).to(self.device)
        # z.data.normal_(0,1)
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        self.G.eval()
        self.D.eval()
        # save_dir = os.path.join(save_dir,self.base_dir+"_{}".format(epoch))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for i in range(samples.size(0)):
            z =  torch.rand(self.z_dim).to(self.device)
            z.data.normal_(0,1)
            imgs = self.G(z.unsqueeze(0), samples[i].unsqueeze(0))
            # os.path.join(save_dir,'{}.png'.format(i))
            # torchvision.utils.save_image(imgs[0],os.path.join(save_dir,'{}.png'.format(i)))
            scipy.misc.imsave(save_dir+'/{}.png'.format(i), imgs[0].permute(1,2,0).cpu().data.numpy())
# def parse_args():
#     desc = "Pytorch implementation of GAN collections"
#     parser = argparse.ArgumentParser(description=desc)
#
#     parser.add_argument('--save_dir', type=str, default='models',
#                         help='Directory name to save the model')
#
#     return parser.parse_args()
if __name__ == '__main__':
    from cartoonDataset import cartoonDataset
    cartoon_data = cartoonDataset(is_train=1)
    gan = ACGAN(1000,64,cartoon_data)
    gan.train()
    # gan.load()
