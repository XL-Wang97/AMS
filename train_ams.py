from __future__ import print_function
import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import copy
from models.wideresnet import *
import models
from utils import Bar, Logger, AverageMeter, accuracy, LogIt

parser = argparse.ArgumentParser(description='TRADES Adversarial Training with AMS')
parser.add_argument('--arch', type=str, default='PreActResNet')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='retrain from which epoch')
parser.add_argument('--data', type=str, default='CIFAR10', choices=['CIFAR10', 'CIFAR100', 'SVHN', 'TinyImageNet'])
parser.add_argument('--data-path', type=str, default='../data',
                    help='where is the dataset CIFAR-10')
parser.add_argument('--weight-decay', '--wd', default=5e-4,
                    type=float, metavar='W')
parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--norm', default='l_inf', type=str, choices=['l_inf', 'l_2'],
                    help='The threat model')
parser.add_argument('--epsilon', default=8, type=float,
                    help='perturbation')
parser.add_argument('--num-steps', default=10, type=int,
                    help='perturb number of steps')
parser.add_argument('--step-size', default=2, type=float,
                    help='perturb step size')
parser.add_argument('--beta', default=6.0, type=float,
                    help='regularization, i.e., 1/lambda in TRADES')
parser.add_argument('--model-dir', default='./model-cifar100-s/lastandbest',
                    help='directory of model for saving checkpoint')
parser.add_argument('--resume-dir', default='./model-cifar100-s/resume',
                    help='directory of model for saving checkpoint')
parser.add_argument('--resume-model', default='', type=str,
                    help='directory of model for retraining')
parser.add_argument('--resume-optim', default='', type=str,
                    help='directory of optimizer for retraining')
parser.add_argument('--save-freq', '-s', default=1, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--alpha', default=0.5,type=float,
                    help='regularization')
parser.add_argument('--cuda-visible-divice', default=0, type=int,
                    help='cuda visible divice number')
parser.add_argument('--m', default=20, type=int,
                    help='save frequency')

args = parser.parse_args()
epsilon = args.epsilon / 255
step_size = args.step_size / 255
if args.data == 'CIFAR100':
    NUM_CLASSES = 100
else:
    NUM_CLASSES = 10

model_dir = args.model_dir
resume_dir = args.resume_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(resume_dir):
    os.makedirs(resume_dir)
use_cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
trainset = getattr(datasets, args.data)(
    root=args.data_path, train=True, download=True, transform=transform_train)
testset = getattr(datasets, args.data)(
    root=args.data_path, train=False, download=True, transform=transform_test)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, **kwargs)

def perturb_input(model,
                  x_natural,
                  step_size=0.003,
                  epsilon=0.031,
                  perturb_steps=10,
                  distance='l_inf'):
    model.eval()
    batch_size = len(x_natural)
    if distance == 'l_inf':
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss_kl = F.kl_div(F.log_softmax(model(x_adv), dim=1),
                                   F.softmax(model(x_natural), dim=1),
                                   reduction='sum')
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'l_2':
        delta = 0.001 * torch.randn(x_natural.shape).to(device).detach()
        delta = Variable(delta.data, requires_grad=True)
        optimizer_delta = optim.SGD([delta], lr=epsilon / perturb_steps * 2)

        for _ in range(perturb_steps):
            adv = x_natural + delta
            optimizer_delta.zero_grad()
            with torch.enable_grad():
                loss = (-1) * F.kl_div(F.log_softmax(model(adv), dim=1),
                                       F.softmax(model(x_natural), dim=1),
                                       reduction='sum')
            loss.backward()
            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            optimizer_delta.step()

            # projection
            delta.data.add_(x_natural)
            delta.data.clamp_(0, 1).sub_(x_natural)
            delta.data.renorm_(p=2, dim=0, maxnorm=epsilon)
        x_adv = Variable(x_natural + delta, requires_grad=False)
    else:
        x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv

def train(model, train_loader, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    bar = Bar('Processing', max=len(train_loader))
    criterion_kl = nn.KLDivLoss(size_average=False)
    
    for batch_idx, (data, target) in enumerate(train_loader):
        x_natural, target = data.to(device), target.to(device)
        
        batch_size = len(x_natural)
        # craft adversarial examples
        x_adv = perturb_input(model=model,
                              x_natural=x_natural,
                              step_size=step_size,
                              epsilon=epsilon,
                              perturb_steps=args.num_steps,
                              distance=args.norm)

        model.train()

        optimizer.zero_grad()
        output = model(x_natural)
        output_adv = model(x_adv)
        loss_natural = F.cross_entropy(output, target)
        loss_natural.backward(retain_graph=True)

        optimizer.zero_grad()
        loss_adv = F.cross_entropy(output_adv, target)
        loss_adv.backward(retain_graph=True)

        optimizer.zero_grad()
        kl_loss =0
        sum_probs =0
        alpha = 0
        for filename in os.listdir(resume_dir):
            net = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).to(device)
            path = os.path.join(resume_dir, filename)
            net.load_state_dict(torch.load(path))
            old_model = copy.deepcopy(net)
            old_model.eval()
            old_outputs = old_model(x_adv).detach()  # Ensure no gradient tracking
            kl_div = F.kl_div(F.log_softmax(old_outputs, dim=1),
                               F.softmax(output_adv, dim=1),
                               reduction='batchmean')
            prob = F.softmax(old_outputs, dim=1)
            avg_probs = prob[torch.arange(target.size(0)), target].mean()
            sum_probs += avg_probs
            kl_loss = kl_loss + avg_probs * kl_div
            alpha = args.alpha / sum_probs   

        loss_robust = F.kl_div(F.log_softmax(output_adv, dim=1),
                               F.softmax(output, dim=1),
                               reduction='batchmean')

        loss = loss_natural + args.beta*loss_robust + alpha * kl_loss

        prec1, prec5 = accuracy(output_adv, target, topk=(1, 5))
        losses.update(loss.item(), x_natural.size(0))
        top1.update(prec1.item(), x_natural.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        batch_time.update(time.time() - end)
        end = time.time()

        bar.suffix = '({batch}/{size}) Batch: {bt:.3f}s| Total:{total:}| ETA:{eta:}| Loss:{loss:.4f}| top1:{top1:.2f}'.format(
            batch=batch_idx + 1,
            size=len(train_loader),
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            top1=top1.avg)
        bar.next()
    bar.finish()
    return losses.avg, top1.avg

def _pgd_whitebox(model,
                  X,
                  y,
                  epsilon=epsilon,
                  num_steps=args.num_steps,
                  step_size=step_size):
    model.eval()
    out = model(X)

    X_pgd = Variable(X.data, requires_grad=True)
    random_noise = torch.FloatTensor(*X_pgd.shape).uniform_(-epsilon, epsilon).to(device)
    X_pgd = Variable(X_pgd.data + random_noise, requires_grad=True)

    for _ in range(num_steps):
        opt = optim.SGD([X_pgd], lr=1e-3)
        opt.zero_grad()

        with torch.enable_grad():
            loss2 = nn.CrossEntropyLoss()(model(X_pgd), y)
        loss2.backward()
        eta = step_size * X_pgd.grad.data.sign()
        X_pgd = Variable(X_pgd.data + eta, requires_grad=True)
        eta = torch.clamp(X_pgd.data - X.data, -epsilon, epsilon)
        X_pgd = Variable(X.data + eta, requires_grad=True)
        X_pgd = Variable(torch.clamp(X_pgd, 0, 1.0), requires_grad=True)
    return X_pgd

def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def eval_adv_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            with torch.no_grad():
                X, y = Variable(data, requires_grad=True), Variable(target)
                data_adv = _pgd_whitebox(copy.deepcopy(model), X, y)
            output = model(data_adv)
            test_loss += F.cross_entropy(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy

def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate"""
    lr = args.lr
    if epoch >= 100:
        lr = args.lr * 0.1
    if epoch >= 150:
        lr = args.lr * 0.01
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def main():
    model = nn.DataParallel(getattr(models, args.arch)(num_classes=NUM_CLASSES)).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    criterion = nn.CrossEntropyLoss()

    logger = Logger(os.path.join(model_dir, 'log.txt'), title=args.arch)
    logger.set_names(['epoch', 'train_time',
                      'benign_accuarcy', 'adv_accuarcy', 'benign test loss',
                      'adversarial test loss'])
    log_dir = os.path.join(model_dir, 'log.txt')
    loger = LogIt(log_dir)

    if args.resume_model:
        model.load_state_dict(torch.load(args.resume_model, map_location=device))
    if args.resume_optim:
        optimizer.load_state_dict(torch.load(args.resume_optim, map_location=device))

    tstt = []
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs + 1):
        adjust_learning_rate(optimizer, epoch)

        train(model, train_loader, optimizer, epoch)
        end_time = time.time()
        train_time = end_time - start_time
        tstloss, tstacc = eval_test(model, device, test_loader)
        advtstloss, advtstacc = eval_adv_test(model, device, test_loader)
        print('Epoch '+str(epoch)+': '+str(int(time.time()-start_time))+'s', end=', ')
        print('test_loss: {:.4f}, test_acc: {:.2f}%'.format(tstloss, 100. * tstacc), end=', ')
        print('adv_test_loss: {:.4f}, adv_test_acc: {:.2f}%'.format(advtstloss, 100. * advtstacc))
        loger.log_iter(epoch, {'train_time': train_time, 'benign_accuarcy' : tstacc, 'adv_accuarcy' : advtstacc, 'benign test loss' : tstloss, 'adversarial test loss' : advtstloss})

        tstt.append(advtstacc)
        # save checkpoint
        torch.save(model.state_dict(), os.path.join(model_dir, 'lastepoch.pt'))
        if epoch>99 and advtstacc==max(tstt):
            torch.save(model.state_dict(), os.path.join(model_dir, 'bestepoch.pt'))
        if epoch > 0 and epoch % args.m==0:
            torch.save(model.state_dict(), os.path.join(resume_dir, 'epoch{}.pt'.format(epoch)))


if __name__ == '__main__':
    main()
