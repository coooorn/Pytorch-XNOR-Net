import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, Variable

import binop


def bin_save_state(args, model):
    print('==> Binarizing and Saving model ...')
    state = model.state_dict()
    weight_ = []
    for key in state.keys():
        if 'weight' in key and 'bn' not in key:
            weight_.append((key, state.get(key)))

    # except the first and last layer
    weight_.pop(0)
    weight_.pop()

    for key, weight in weight_:
        s = weight.size()
        if len(s) == 4:
            weight = weight.view(s[0], s[1] * s[2] * s[3])

        if args.cuda:
            bin_weight = torch.cuda.IntTensor()
            binop.encode_rows(weight, bin_weight)
        else:
            bin_weight = torch.IntTensor()
            binop.encode_rows_cpu(weight, bin_weight)

        state[key] = bin_weight

    torch.save(state, 'models/' + args.arch + '.pth')


def bin_conv2d(input, weight, bias, alpha, kernel_size, stride, padding, groups=1):
    out_tensor = torch.FloatTensor()
    use_cuda = input.is_cuda
    if use_cuda:
        out_tensor = out_tensor.cuda()
    output = Variable(out_tensor, requires_grad=False)
    if bias is None:
        if use_cuda:
            bias = Variable(torch.cuda.FloatTensor(), requires_grad=False)
        else:
            bias = Variable(torch.FloatTensor(), requires_grad=False)
    if use_cuda:
        binop.update_conv_output_gpu(input.data, output.data, weight.data, bias.data, alpha.data,
                                     kernel_size[0], kernel_size[1], stride[0], stride[1], padding[0], padding[1],
                                     groups)
    else:
        binop.update_conv_output_cpu(input.data, output.data, weight.data, bias.data, alpha.data,
                                     kernel_size[0], kernel_size[1], stride[0], stride[1], padding[0], padding[1],
                                     groups)
    return output


def bin_linear(input, weight, bias, alpha):
    m = input.data.shape[0]
    n = input.data.shape[1]
    k = weight.data.shape[0]
    out_tensor = torch.FloatTensor()
    bin_input = torch.IntTensor()
    use_cuda = input.is_cuda

    if use_cuda:
        bin_input = bin_input.cuda()
        out_tensor = out_tensor.cuda()

    output = Variable(out_tensor, requires_grad=False)
    if use_cuda:
        binop.encode_rows(input.data, bin_input)
        binop.binary_gemm(bin_input, weight.data, output.data, m, n, k, 1, 0, 0, alpha.data)
    else:
        binop.encode_rows_cpu(input.data, bin_input)
        binop.binary_gemm_cpu(bin_input, weight.data, output.data, m, n, k, 1, 0, 0, alpha.data)
    output.data.mul_(alpha.data.t().expand(output.shape))
    if bias is not None:
        output.data.add_(bias.data.expand(output.shape))
    return output


class BinActive(Function):
    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        input = input.sign()
        return input

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input


class BinConv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, is_train=True, drop=0,
                 groups=1):
        if is_train:
            super(BinConv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, bias=bias, groups=groups)
        else:
            super(BinConv2d, self).__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            padding=padding, bias=bias)
            self.groups = groups
        self.alpha = nn.Parameter(torch.FloatTensor(out_channels, 1, 1, 1), requires_grad=False)
        self.is_train = is_train
        self.bn = nn.BatchNorm2d(in_channels)
        self.dropout_ratio = drop

        if drop != 0:
            self.drop = nn.Dropout(drop)

        if not is_train:
            h_weight = out_channels
            w_weight = (1 + (in_channels * self.kernel_size[0] * self.kernel_size[1] - 1) // 32) // self.groups
            self.weight = nn.Parameter(torch.IntTensor(h_weight, w_weight))

    def forward(self, input):
        input = self.bn(input)

        if self.is_train:
            input = BinActive.apply(input)
            if self.dropout_ratio != 0:
                input = self.drop(input)
            input = F.conv2d(input, weight=self.weight, bias=self.bias, stride=self.stride, padding=self.padding,
                             groups=self.groups)
        else:
            input = bin_conv2d(input, self.weight, self.bias, self.alpha, self.kernel_size, self.stride, self.padding,
                               groups=self.groups)
        return input


class BinLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, is_train=True, drop=0):
        super().__init__(in_features, out_features, bias)

        self.alpha = nn.Parameter(torch.FloatTensor(out_features, 1), requires_grad=False)
        self.istrain = is_train
        self.bn = nn.BatchNorm1d(in_features)
        self.dropout_ratio = drop
        if drop != 0:
            self.drop = nn.Dropout(drop)
        if not is_train:
            self.weight = nn.Parameter(torch.IntTensor(out_features, 1 + (in_features - 1) // 32))

    def forward(self, input):
        input = self.bn(input)
        if self.istrain:
            input = BinActive.apply(input)
            if self.dropout_ratio != 0:
                input = self.drop(input)
            input = F.linear(input, weight=self.weight, bias=self.bias)
        else:
            input = bin_linear(input, weight=self.weight, bias=self.bias, alpha=self.alpha)
        return input


class BinopTrain:
    def __init__(self, model):
        self.alpha_to_save = []
        self.saved_params = []
        self.target_modules = []
        for m in model.modules():
            if type(m).__name__ in ['BinConv2d', 'BinLinear']:
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp)
                self.target_modules.append(m.weight)
                self.alpha_to_save.append(m.alpha)
        self.num_of_params = len(self.target_modules)

    def binarization(self):
        for index in range(self.num_of_params):
            n = self.target_modules[index].data[0].nelement()
            s = self.target_modules[index].data.size()

            # meancenter
            negMean = self.target_modules[index].data.mean(1, keepdim=True).mul(-1).expand_as(
                self.target_modules[index].data)
            self.target_modules[index].data.add_(negMean)
            # clamp
            self.target_modules[index].data.clamp_(-1.0, 1.0)
            # save param
            self.saved_params[index].copy_(self.target_modules[index].data)

            # get alpha, binarize weight and mutiply alpha
            if len(s) == 4:
                self.alpha_to_save[index].data = self.target_modules[index].data.norm(1, 3, keepdim=True) \
                    .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
            elif len(s) == 2:
                self.alpha_to_save[index].data = self.target_modules[index].data.norm(1, 1, keepdim=True).div(n)
            self.target_modules[index].data.sign().mul(self.alpha_to_save[index].data.expand(s),
                                                       out=self.target_modules[index].data)

    def restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])

    def update_binary_grad_weight(self):
        for index in range(self.num_of_params):
            weight = self.target_modules[index].data
            alpha = self.alpha_to_save[index].data.clone()
            n = weight[0].nelement()
            s = weight.size()
            alpha = alpha.expand(s)
            alpha[weight.le(-1.0)] = 0
            alpha[weight.ge(1.0)] = 0
            alpha = alpha.mul(self.target_modules[index].grad.data)
            add = weight.sign().mul(self.target_modules[index].grad.data)
            if len(s) == 4:
                add = add.sum(3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
            elif len(s) == 2:
                add = add.sum(1, keepdim=True).div(n).expand(s)
            add = add.mul(weight.sign())
            self.target_modules[index].grad.data = alpha.add(add).mul(1.0 - 1.0 / s[1]).mul(n)
