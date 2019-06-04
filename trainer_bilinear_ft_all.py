import torch
import torch.nn as nn

import torchvision.models as models
from utils.Config import Config
from utils.weight_init import weight_init_kaiming
from torchvision import models
import os
import numpy as np
from torchvision import transforms
import matplotlib.pyplot as plt
import shutil
from cub import cub200
import argparse
from utils.bilinear_layers import *
import time
import torchvision

stage = "train_all"

class BCNN(torch.nn.Module):
    def __init__(self, is_all=True, num_classes=200):
        torch.nn.Module.__init__(self)
        self._is_all = is_all

        if self._is_all:
            self.features = torchvision.models.vgg16(pretrained=True).features
            self.features = torch.nn.Sequential(*list(self.features.children())
                                                [:-2])

        self.relu5_3 = torch.nn.ReLU(inplace=False)
        self.sign_sqrt = sign_sqrt.apply

        self.classifier = torch.nn.Linear(
            in_features=512 * 512, out_features=num_classes, bias=True)

        torch.nn.init.xavier_uniform_(self.classifier.weight)

    def forward(self, X):
        N = X.size()[0]
        if self._is_all:
            assert X.size() == (N, 3, 448, 448)
            X = self.features(X)
        assert X.size() == (N, 512, 28, 28)
        X = self.relu5_3(X)
        assert X.size() == (N, 512, 28, 28)
        X = torch.reshape(X, (N, 512, 28 * 28))
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (28 * 28)
        assert X.size() == (N, 512, 512)
        X = self.sign_sqrt(X)
        X = torch.reshape(X, (N, 512 * 512))
        X = torch.nn.functional.normalize(X)
        X = self.classifier(X)
        return X

class NetworkManager(object):
    def __init__(self, options, path):
        self.options = options
        self.path = path
        self.device = options['device']

        print('Starting to prepare network and data...')
        net_params = BCNN(is_all=True, num_classes=200)
        self.net = nn.DataParallel(net_params).to(self.device)
        if stage == "train_all":
            self.net.load_state_dict(torch.load('./models/saved/VGG/VGG_bilinear_wonorm_train_last_layer.pkl'))
            1
        else:
            for param in net_params.parameters():
                param.requires_grad = False
            for param in net_params.classifier.parameters():
                param.requires_grad = True
        print('Network is as follows:')
        print(self.net)
        #print(self.net.state_dict())
        self.criterion = nn.CrossEntropyLoss()
        if stage == "train_all":
            self.solver = torch.optim.SGD(self.net.parameters(), lr=self.options['base_lr'], momentum=self.options['momentum'], weight_decay=self.options['weight_decay'])
        else:
            self.solver = torch.optim.SGD(filter(lambda p: p.requires_grad, self.net.parameters()), lr=self.options['base_lr'], momentum=self.options['momentum'], weight_decay=self.options['weight_decay'])
        # self.schedule = torch.optim.lr_scheduler.StepLR(self.solver, step_size=50, gamma=0.25)
        self.schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.solver, mode='max', factor=0.25, patience=5, verbose=True,
            threshold=1e-4)
        if stage == "train_all":
            self.schedule = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.solver, mode='max', factor=0.1, patience=5, verbose=True,
                threshold=1e-4)

        train_transform_list = [
            transforms.Resize(512),
            transforms.RandomCrop(448),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ]
        test_transforms_list = [
            transforms.Resize(512),
            transforms.CenterCrop(self.options['img_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))
        ]
        train_data = cub200(self.path['data'], train=True, transform=transforms.Compose(train_transform_list))
        test_data = cub200(self.path['data'], train=False, transform=transforms.Compose(test_transforms_list))
        self.train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self.options['batch_size'], shuffle=True, num_workers=4, pin_memory=True
        )
        self.test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=16, shuffle=False, num_workers=4, pin_memory=True
        )

    def train(self):
        epochs  = np.arange(1, self.options['epochs']+1)
        test_acc = list()
        train_acc = list()
        print('Training process starts:...')
        if torch.cuda.device_count() > 1:
            print('More than one GPU are used...')
        print('Epoch\tTrainLoss\tTrainAcc\tTestAcc\tLearningRate')
        print('-'*50)
        best_acc = 0.0
        best_epoch = 0
        self.net.train(True)
        for epoch in range(self.options['epochs']):
            num_correct = 0
            train_loss_epoch = list()
            num_total = 0
            for imgs, labels in self.train_loader:
                self.solver.zero_grad()
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                output = self.net(imgs)
                loss = self.criterion(output, labels)
                _, pred = torch.max(output, 1)
                num_correct += torch.sum(pred == labels.detach_())
                num_total += labels.size(0)
                train_loss_epoch.append(loss.item())
                loss.backward()
                #nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                self.solver.step()

            if epoch % 1 == 0:
                test_acc_epoch = self._accuracy()
            train_acc_epoch = num_correct.detach().cpu().numpy()*100. / num_total
            avg_train_loss_epoch  = sum(train_loss_epoch)/len(train_loss_epoch)
            test_acc.append(test_acc_epoch)
            train_acc.append(train_acc_epoch)
            save_flg = ""
            if test_acc_epoch>best_acc:
                best_acc = test_acc_epoch
                best_epoch = epoch+1
                torch.save(self.net.state_dict(), os.path.join(self.path['model_save'], self.options['net_choice'], "VGG_bilinear_wonorm_"+stage+'.pkl'))
                save_flg = "model saved!"
            print('{}\t{:.4f}\t{:.2f}%\t{:.2f}%\t{:f}\t'.format(epoch, avg_train_loss_epoch, train_acc_epoch, test_acc_epoch, self.solver.param_groups[0]['lr'])+time.strftime('%H:%M:%S\t',time.localtime(time.time()))+save_flg)
            self.schedule.step(test_acc_epoch)
            plt.figure()
        plt.plot(epochs, test_acc, color='r', label='Test Acc')
        plt.plot(epochs, train_acc, color='b', label='Train Acc')

        plt.xlabel('epochs')
        plt.ylabel('Acc')
        plt.legend()
        plt.title(self.options['net_choice']+str(self.options['model_choice']))
        plt.savefig(self.options['net_choice']+str(self.options['model_choice'])+'.png')

    def _accuracy(self):
        self.net.eval()
        num_total = 0
        num_acc = 0
        with torch.no_grad():
            for imgs, labels in self.test_loader:
                imgs = imgs.to(self.device)
                labels = labels.to(self.device)
                output = self.net(imgs)
                _, pred = torch.max(output, 1)
                num_acc += torch.sum(pred==labels.detach_())
                num_total += labels.size(0)
        return num_acc.detach().cpu().numpy()*100./num_total

    def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, 'model_best.pth.tar')

    def adjust_learning_rate(optimizer, epoch, args):
        print("here")
        print(args.lr)
        lr = args.lr * (0.1 ** (epoch // 50))
        if stage == "train_last_layer":
            lr = args.lr * (0.25 ** (epoch // 50))

        for param_group in optimizer.param_groups:
            print(param_group['lr'])
            param_group['lr'] = lr

def main():
    parser = argparse.ArgumentParser(
        description='Options for base model finetuning on CUB_200_2011 datasets'
    )
    if stage == "train_last_layer":
        parser.add_argument('--base_lr', type=float, default=1.)
        parser.add_argument('--epochs', type=int, default=100)
    else:
        parser.add_argument('--base_lr', type=float, default=0.01)
        parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--net_choice', type=str, default="VGG")
    parser.add_argument('--model_choice', type=int, default="16")
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-6)
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--img_size', type=int, default=448)
    args = parser.parse_args()
    assert args.gpu_id.__class__ == int


    options = {
        'net_choice': args.net_choice,
        'model_choice': args.model_choice,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'base_lr': args.base_lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum,
        'img_size': args.img_size,
        'device': torch.device('cuda:'+str(args.gpu_id) if torch.cuda.is_available() else 'cpu')
    }

    path = {
        'data': Config.data_path,
        'model_save': Config.model_save_path
    }

    for p in path:
        print(p)
        print(path[p])
        assert os.path.isdir(path[p])

    manager = NetworkManager(options, path)
    manager.train()


if __name__ == '__main__':
    main()