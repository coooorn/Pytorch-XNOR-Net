import argparse
import itertools
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import datasets, transforms

from tqdm import tqdm

from sklearn.metrics import classification_report, confusion_matrix

sys.path.append('../')

import Cifar10.models as models

from util import BinopTrain, bin_save_state

tqdm.monitor_interval = 0


class RunningMean:
    def __init__(self, value=0, count=0):
        self.total_value = value
        self.count = count

    def update(self, value, count=1):
        self.total_value += value
        self.count += count

    @property
    def value(self):
        if self.count:
            return self.total_value / self.count
        else:
            return float("inf")

    def __str__(self):
        return str(self.value)


def plot_confusion_matrix(cm, classes, normalize=False,
                          title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def eval_test(y_pred, y_true):
    cnf_matrix = confusion_matrix(y_true, y_pred)
    np.set_printoptions(precision=6)

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=target_name,
                          title='Confusion matrix')

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=target_name, normalize=True,
                          title='Normalized confusion matrix')

    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=target_name, digits=6))
    plt.show()


def save_state(model, mname=""):
    print('==> Saving model ...')
    torch.save(model.state_dict(), 'models/' + args.arch + mname + '.pth')


def train(model, binop_model, epoch):
    running_loss = RunningMean()
    running_score = RunningMean()
    model.train()
    pbar = tqdm(train_loader, total=len(train_loader), )

    for data, target in pbar:
        batch_size = data.size(0)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)

        optimizer.zero_grad()
        if binop_model is not None:
            binop_model.binarization()

        output = model(data)
        _, preds = torch.max(output.data, dim=1)
        loss = criterion(output, target)
        running_loss.update(loss.data[0], 1)
        running_score.update(torch.sum(preds != target.data), batch_size)
        loss.backward()

        if binop_model is not None:
            # restore weights
            binop_model.restore()
            # update
            binop_model.update_binary_grad_weight()

        optimizer.step()

        pbar.set_description('%.6f %.6f' % (running_loss.value, running_score.value))
    print('[+] epoch %d: \nTraining: Average loss: %.6f, Average error: %.6f' %
          (epoch, running_loss.value, running_score.value))

    if binop_model is not None:
        bin_save_state(args, model)
    else:
        save_state(model)


def test(model, evaluate=False):
    global best_acc
    test_loss = 0
    correct = 0
    model.eval()
    if evaluate:
        model.load_state_dict(torch.load(args.pretrained))
    else:
        model.load_state_dict(torch.load('models/' + args.arch + '.pth'))
    pbar = tqdm(test_loader, total=len(test_loader))
    if evaluate:
        pred = torch.LongTensor()
        true = torch.LongTensor()
        if args.cuda:
            pred = pred.cuda()
            true = true.cuda()
    for data, target in pbar:
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        data, target = Variable(data), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0]
        if evaluate:
            pred = torch.cat((pred, output.data.max(1, keepdim=False)[1]))
            true = torch.cat((true, target.data))
        else:
            pred = output.data.max(1, keepdim=False)[1]
            correct += pred.eq(target.data).cpu().sum()
    if evaluate:
        correct = pred.eq(true).cpu().sum()
    acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * args.batch_size, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    if not evaluate:
        if acc > best_acc:
            best_acc = acc
            os.rename('models/' + args.arch + '.pth', 'models/' + args.arch + '.best.pth')
            if model_train is not None:
                save_state(model_train, "bak")
        else:
            os.remove('models/' + args.arch + '.pth')
        print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    else:
        eval_test(y_pred=pred.tolist(), y_true=true.tolist())


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 15 epochs"""
    update_list = [150, 200, 250]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    print('Learning Rate: {}'.format(optimizer.param_groups[0]['lr']))


if __name__ == '__main__':
    # Training settings

    parser = argparse.ArgumentParser(description='PyTorch Cifar-10')

    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 128)')
    parser.add_argument('--epochs', type=int, default=300, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr-epochs', type=int, default=100, metavar='N',
                        help='number of epochs to decay the lr (default: 20)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float,
                        metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--arch', action='store', default='Bin_VGG16',
                        help='the MNIST network structure: Bin_VGG19')
    parser.add_argument('--pretrained', action='store', default=None,
                        help='pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='whether to run evaluation')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    target_name = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(args)
    cnt = 0
    lr_cnt = 0

    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    else:
        torch.manual_seed(args.seed)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # load data
    kwargs = {'num_workers': 1, 'pin_memory': True}

    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]), download=True), batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root='./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    best_acc = 0.0
    binop_model_train = None
    model_train = None
    model_test = None

    # generate the model

    model_names = ['VGG11', 'VGG13', 'VGG16', 'VGG19']  # ['RESNET18', 'NIN',
    args.arch = args.arch.upper()
    name = args.arch.split('_')[-1] if '_' in args.arch else args.arch

    if name not in model_names:
        print('ERROR: specified arch is not suppported')
        exit()

    if 'GROUPED_BIN_VGG' in args.arch:
        model_train = models.GroupedBinVGG(name)
        model_test = models.GroupedBinVGG(name, is_train=False)
    elif 'BIN_VGG' in args.arch:
        model_train = models.BinVGG(name)
        model_test = models.BinVGG(name, is_train=False)
    elif 'GROUPED_VGG' in args.arch:
        model_train = models.GroupedVGG(name)
        model_test = models.GroupedVGG(name)
    elif 'VGG' in name:
        model_train = models.VGG(name)
        model_test = models.VGG(name)
    else:
        print('ERROR: specified arch is not suppported')
        exit()

    if args.cuda:
        model_train.cuda()
        model_test.cuda()

    if 'BIN' in args.arch:
        if args.pretrained:
            if args.evaluate:
                model_test.load_state_dict(torch.load(args.pretrained))
            else:
                model_train.load_state_dict(torch.load(args.pretrained))
                binop_model_train = BinopTrain(model_train)
        else:
            binop_model_train = BinopTrain(model_train)
    elif args.pretrained:
        model_test.load_state_dict(torch.load(args.pretrained))

    param_dict = dict(model_train.named_parameters()) if model_train is None else dict(model_train.named_parameters())
    params = []

    for key, value in param_dict.items():
        if value.requires_grad:
            params += [{
                'params': [value],
                'lr': args.lr,
                'key': key
            }]

    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum)

    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion.cuda()

    if args.evaluate:
        if model_train is None:
            test(model_test, evaluate=True)
        else:
            test(model_train, evaluate=True)
        exit()

    print('training parameters: %d' % sum([np.prod(p.size()) for p in model_train.parameters() if p.requires_grad]))
    print('testing parameters: %d' % sum([np.prod(p.size()) for p in model_test.parameters() if p.requires_grad]))

    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)
        train(model_train, binop_model_train, epoch)
        test(model_test)
