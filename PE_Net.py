import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def conv_block(in_channel, out_channel, kernel_size=3, stride=(2, 2), padding=1, normalization=False, pooling=False):
    layers = [
        nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding),
        nn.LeakyReLU(0.2),
    ]
    if normalization:
        layers.append(nn.InstanceNorm2d(out_channel, affine=True))

    if pooling:
        layers.append(nn.AdaptiveAvgPool2d(1))
    return layers


class Exp_pre(nn.Module):
    def __init__(self,inchannel, filter_nums):
        super(Exp_pre, self).__init__()
        self.conv1 = nn.Sequential(
            *conv_block(inchannel, 16, normalization=True)
        )
        self.conv2 = nn.Sequential(
            *conv_block(16, 32, normalization=True)
        )
        self.conv3 = nn.Sequential(
            *conv_block(32, 64, normalization=True)
        )
        self.conv4 = nn.Sequential(
            *conv_block(64, 64, normalization=True)
        )
        self.fc11 = nn.Sequential(
            nn.Linear(1, 16),
            nn.Softmax(dim=1)
        )
        self.fc12 = nn.Sequential(
            nn.Linear(1, 16),
            nn.Softmax(dim=1)
        )
        self.fc21 = nn.Sequential(
            nn.Linear(1, 32),
            nn.Softmax(dim=1)
        )
        self.fc22 = nn.Sequential(
            nn.Linear(1, 32),
            nn.Softmax(dim=1)
        )
        self.fc31 = nn.Sequential(
            nn.Linear(1, 64),
            nn.Softmax(dim=1)
        )
        self.fc32 = nn.Sequential(
            nn.Linear(1, 64),
            nn.Softmax(dim=1)
        )
        self.fc41 = nn.Sequential(
            nn.Linear(1, 64),
            nn.Softmax(dim=1)
        )
        self.fc42 = nn.Sequential(
            nn.Linear(1, 64),
            nn.Softmax(dim=1)
        )
        self.pre_fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, filter_nums),
            nn.Tanh()
        )

    def forward(self, x, y):
        y = torch.mean(y, dim=(1, 2, 3)).unsqueeze(1)
        x = self.conv1(x)
        scale1 = self.fc11(y)
        shift1 = self.fc12(y)
        x = x * scale1.unsqueeze(2).unsqueeze(3) + shift1.unsqueeze(2).unsqueeze(3)

        x = self.conv2(x)
        scale2 = self.fc21(y)
        shift2 = self.fc22(y)
        x = x * scale2.unsqueeze(2).unsqueeze(3) + shift2.unsqueeze(2).unsqueeze(3)

        x = self.conv3(x)
        scale3 = self.fc31(y)
        shift3 = self.fc32(y)
        x = x * scale3.unsqueeze(2).unsqueeze(3) + shift3.unsqueeze(2).unsqueeze(3)

        x = self.conv4(x)
        scale4 = self.fc41(y)
        shift4 = self.fc42(y)
        x = x * scale4.unsqueeze(2).unsqueeze(3) + shift4.unsqueeze(2).unsqueeze(3)

        x = x.view(-1, 64 * 8 * 8)
        x = self.pre_fc(x)
        return x


class Con_pre(nn.Module):
    def __init__(self, inchannel, filter_nums):
        super(Con_pre, self).__init__()
        self.conv1 = nn.Sequential(
            *conv_block(inchannel, 16, normalization=True)
        )
        self.conv2 = nn.Sequential(
            *conv_block(16, 32, normalization=True)
        )
        self.conv3 = nn.Sequential(
            *conv_block(32, 64, normalization=True)
        )
        self.conv4 = nn.Sequential(
            *conv_block(64, 64, normalization=True)
        )
        self.fc11 = nn.Sequential(
            nn.Linear(256, 16),
            nn.Softmax(dim=1)
        )
        self.fc12 = nn.Sequential(
            nn.Linear(256, 16),
            nn.Softmax(dim=1)
        )
        self.fc21 = nn.Sequential(
            nn.Linear(256, 32),
            nn.Softmax(dim=1)
        )
        self.fc22 = nn.Sequential(
            nn.Linear(256, 32),
            nn.Softmax(dim=1)
        )
        self.fc31 = nn.Sequential(
            nn.Linear(256, 64),
            nn.Softmax(dim=1)
        )
        self.fc32 = nn.Sequential(
            nn.Linear(256, 64),
            nn.Softmax(dim=1)
        )
        self.fc41 = nn.Sequential(
            nn.Linear(256, 64),
            nn.Softmax(dim=1)
        )
        self.fc42 = nn.Sequential(
            nn.Linear(256, 64),
            nn.Softmax(dim=1)
        )
        self.pre_fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, filter_nums),
            nn.Softmax(dim=1)
        )

    def forward(self, x, y):
        x = self.conv1(x)
        scale1 = self.fc11(y)
        shift1 = self.fc12(y)
        x = x * scale1.unsqueeze(2).unsqueeze(3) + shift1.unsqueeze(2).unsqueeze(3)

        x = self.conv2(x)
        scale2 = self.fc21(y)
        shift2 = self.fc22(y)
        x = x * scale2.unsqueeze(2).unsqueeze(3) + shift2.unsqueeze(2).unsqueeze(3)

        x = self.conv3(x)
        scale3 = self.fc31(y)
        shift3 = self.fc32(y)
        x = x * scale3.unsqueeze(2).unsqueeze(3) + shift3.unsqueeze(2).unsqueeze(3)

        x = self.conv4(x)
        scale4 = self.fc41(y)
        shift4 = self.fc42(y)
        x = x * scale4.unsqueeze(2).unsqueeze(3) + shift4.unsqueeze(2).unsqueeze(3)

        x = x.view(-1, 64 * 8 * 8)
        x = self.pre_fc(x)
        return x


class Sat_pre(nn.Module):
    def __init__(self, inchannel, filter_nums):
        super(Sat_pre, self).__init__()
        self.layers = [16, 32, 64]
        self.conv1 = nn.Sequential(
            *conv_block(inchannel, 16, normalization=True)
        )
        self.conv2 = nn.Sequential(
            *conv_block(16, 32, normalization=True)
        )
        self.conv3 = nn.Sequential(
            *conv_block(32, 64, normalization=True)
        )
        self.conv4 = nn.Sequential(
            *conv_block(64, 64, normalization=True)
        )
        self.cond1 = nn.Sequential(
            *conv_block(inchannel*2, 16, normalization=True)
        )
        self.cond2 = nn.Sequential(
            *conv_block(16, 32, normalization=True)
        )
        self.cond3 = nn.Sequential(
            *conv_block(32, 64, normalization=True)
        )
        self.cond11 = nn.Sequential(
            *conv_block(inchannel*2, 16, normalization=True)
        )
        self.cond22 = nn.Sequential(
            *conv_block(16, 32, normalization=True)
        )
        self.cond33 = nn.Sequential(
            *conv_block(32, 64, normalization=True)
        )

        self.pre_fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, 512),
            nn.PReLU(),
            nn.Linear(512, 512),
            nn.PReLU(),
            nn.Linear(512, filter_nums),
            nn.Tanh()
        )

    def forward(self, x_, y_):
        x = self.conv1(x_)

        scale1 = self.cond1(torch.cat([x_,y_], dim=1))
        shift1 = self.cond11(torch.cat([x_, y_], dim=1))

        x = x * scale1 + shift1


        x = self.conv2(x)
        scale1 = self.cond2(scale1)
        shift1 = self.cond22(shift1)
        x = x * scale1 + shift1

        x = self.conv3(x)
        scale1 = self.cond3(scale1)
        shift1 = self.cond33(shift1)
        x = x * scale1 + shift1

        x = self.conv4(x)

        x = x.view(-1, 64 * 8 * 8)
        x = self.pre_fc(x)
        return x


class encoder(nn.Module):
    def __init__(self, inchannel, filter_nums):
        super(encoder, self).__init__()
        self.features = nn.Sequential(
            *conv_block(inchannel, 16, normalization=True),
            *conv_block(16, 32, normalization=True),
            *conv_block(32, 64, normalization=True),
            *conv_block(64, 64, normalization=True),
            nn.Dropout(p=0.5),
        )
        # /2,8*8
        self.fc = nn.Sequential(
            nn.Linear(64 * 8 * 8, filter_nums),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.features(x),
        x = x[0].view(-1, 64 * 8 * 8)
        sat = self.fc(x)
        return sat


def histcal(x, bins=256):
    N, C, H, W = x.shape
    x = x.view(N, -1)
    x_min, _ = x.min(-1)
    x_min = x_min.unsqueeze(-1)
    x_max, _ = x.max(-1)
    x_max = x_max.unsqueeze(-1)
    q_levels = torch.arange(bins).float().to(x.device)
    q_levels = q_levels.expand(N, bins)
    q_levels = (2 * q_levels + 1) / (2 * bins) * (x_max - x_min) + x_min
    q_levels = q_levels.unsqueeze(1)
    q_levels_inter = q_levels[:, :, 1] - q_levels[:, :, 0]
    q_levels_inter = q_levels_inter.unsqueeze(-1)
    x = x.unsqueeze(-1)
    quant = 1 - torch.abs(q_levels - x)
    quant = quant * (quant > (1 - q_levels_inter))
    sta = quant.sum(1)
    sta = sta / ((sta.sum(-1).unsqueeze(-1)) + 1e-6)

    return sta

class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()

        kernel_left = torch.FloatTensor([[0, 0, 0], [-1, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor([[0, 0, 0], [0, 1, -1], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor([[0, -1, 0], [0, 1, 0], [0, 0, 0]]).unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor([[0, 0, 0], [0, 1, 0], [0, -1, 0]]).unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)

    def forward(self, org, enhance):
        b, c, h, w = org.shape

        org_mean = torch.mean(org, 1, keepdim=True)
        enhance_mean = torch.mean(enhance, 1, keepdim=True)

        org_pool = self.pool(org_mean)
        enhance_pool = self.pool(enhance_mean)

        weight_diff = torch.max(
            torch.FloatTensor([1]).cuda() + 10000 * torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),
                                                              torch.FloatTensor([0]).cuda()),
            torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()), enhance_pool - org_pool)

        D_org_letf = F.conv2d(org_pool, self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool, self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool, self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool, self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool, self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool, self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool, self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool, self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf, 2)
        D_right = torch.pow(D_org_right - D_enhance_right, 2)
        D_up = torch.pow(D_org_up - D_enhance_up, 2)
        D_down = torch.pow(D_org_down - D_enhance_down, 2)
        E = (D_left + D_right + D_up + D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E

class Global_model(nn.Module):
    def __init__(self):
        super(Global_model, self).__init__()
        self.bri_encoder = Exp_pre(3,1)
        self.curve_encoder = Con_pre(3, 8)
        self.loss = nn.MSELoss()
        self.l1 = nn.L1Loss()
        self.l_spa = L_spa()
        # self.l_tv = L_TV()

    def forward(self, input, ref):
        dtype = next(self.parameters()).dtype
        gamma_factor = self.bri_encoder(input, ref)
        gamma = (gamma_factor[:, 0]).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(input.shape)
        input = input + 1e-6
        input = (input / (input ** gamma)).clamp(0, 1)
        thre = torch.mean(input, dim=[1, 2, 3]).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(input.shape)
        input = (input + gamma * (input - thre)).clamp(0, 1)

        ref_max = torch.max(ref, dim=1, keepdim=True).values
        ref_hist = histcal(ref_max).squeeze(1)
        ref_hist = ref_hist.to(dtype=dtype)
        curve_factor1 = self.curve_encoder(input, ref_hist)
        input = self.curve(8, input, curve_factor1)

        return input,torch.cat((gamma_factor, curve_factor1), dim=1)

    def curve(self, L, x, curve_param):
        fx = torch.zeros_like(x)
        for i in range(0, L):
            fx = torch.clamp(L * x - i, 0, 1) * curve_param[:, i].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(
                x.shape) + fx
            # fx = torch.clamp(L * x - i, 0, 1) * curve_param[i] + fx
        return fx

    def gradient(self, x):
        def sub_gradient(x):
            left_shift_x, right_shift_x, grad = torch.zeros_like(
                x), torch.zeros_like(x), torch.zeros_like(x)
            left_shift_x[:, :, 0:-1] = x[:, :, 1:]
            right_shift_x[:, :, 1:] = x[:, :, 0:-1]
            grad = 0.5 * (left_shift_x - right_shift_x)
            return grad

        return sub_gradient(x), sub_gradient(torch.transpose(x, 2, 3)).transpose(2, 3)

    def expLoss(self, inputs, refs):
        mean1 = torch.mean(inputs, dim=(1, 2, 3))
        mean2 = torch.mean(refs, dim=(1, 2, 3))
        return -torch.log(1 - self.loss(mean1, mean2))

    def conLoss(self, inputs, refs):
        std1 = torch.std(inputs, dim=(1, 2, 3))
        std2 = torch.std(refs, dim=(1, 2, 3))

        return -torch.log(1 - self.loss(std1, std2))

class Color_model(nn.Module):
    def __init__(self):
        super(Color_model, self).__init__()
        self.cd = 256
        self.cl = [86, 52, 52,
                   52, 86, 52,
                   52, 52, 86]
        # self.cl = [86,86,86]
        self.sat_encoder = Sat_pre(3, sum(self.cl))
        
    def interp(self, param, length):
        return F.interpolate(
            param.unsqueeze(1).unsqueeze(2), (1, length),
            mode='bicubic', align_corners=True
        ).squeeze(2).squeeze(1)

    def curve(self, x, func, depth):
        x_ind = (torch.clamp(x, 0, 1) * (depth - 1))
        x_ind = x_ind.round_().long().flatten(1).detach()
        out = torch.gather(func, 1, x_ind)
        return out.reshape(x.size())

    def forward(self, input, ref):
        _, _, H, W = input.size()

        sat = self.sat_encoder(input, ref) * 0.1
        fl = sat.split(self.cl, dim=1)

        residual = torch.cat([
            self.curve(input[:, [0], ...], self.interp(fl[i * 3 + 0], self.cd), self.cd) + \
            self.curve(input[:, [1], ...], self.interp(fl[i * 3 + 1], self.cd), self.cd) + \
            self.curve(input[:, [2], ...], self.interp(fl[i * 3 + 2], self.cd), self.cd)
            for i in range(3)], dim=1)
       
        return (input + residual).clamp(0,1), sat

class PE_Net(nn.Module):
    def __init__(self):
        super(PE_Net, self).__init__()
        self.global_model = Global_model()
        self.color_model = Color_model()

        self.cd = 256
        self.cl = [86, 52, 52,
                   52, 86, 52,
                   52, 52, 86]


    def forward(self, input, ref):
        dtype = next(self.parameters()).dtype
        input = input.to(dtype=dtype)
        ref = ref.to(dtype=dtype)
        input_down = transforms.Resize((128, 128))(input)
        ref_down = transforms.Resize((128, 128))(ref)
        
        img1, param1 = self.global_model(input_down, ref_down)

        img, param2 = self.color_model(img1, ref_down)
        param = torch.cat((param1, param2), dim=1)

        output = self.infer(input, param)

        return output


    def infer(self, input, param):
        dtype = next(self.parameters()).dtype
        input = input.to(dtype=dtype)
        param = param.to(dtype=dtype)
        _, _, H, W = input.size()
        gamma = (param[:, 0]).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(input.shape)

        input = input + 1e-6
        ori_input = input
        input = (input / (input ** gamma)).clamp(0, 1)
        thre = torch.mean(input, dim=[1, 2, 3]).unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(input.shape)

        input = (input + gamma * (input - thre)*0.5).clamp(0, 1)

        curve_factor1 = param[:, 1:9]
        input = self.curve_p(8, input, curve_factor1)
        input1 = input
        # print(input.shape, "!")
        curve_factor1 = param[:, 9:]
        fl = curve_factor1.split(self.cl, dim=1)

        # transform
        residual = torch.cat([
            self.curve(input[:, [0], ...], self.interp(fl[i * 3 + 0], self.cd), self.cd) + \
            self.curve(input[:, [1], ...], self.interp(fl[i * 3 + 1], self.cd), self.cd) + \
            self.curve(input[:, [2], ...], self.interp(fl[i * 3 + 2], self.cd), self.cd)
            for i in range(3)], dim=1)


        return (input+residual).clamp(0,1)


    def curve_p(self, L, x, curve_param):
        fx = torch.zeros_like(x)
        for i in range(0, L):
            fx = torch.clamp(L * x - i, 0, 1) * curve_param[:, i].unsqueeze(1).unsqueeze(2).unsqueeze(3).expand(
                x.shape) + fx
        return fx

    def interp(self, param, length):
        return F.interpolate(
            param.unsqueeze(1).unsqueeze(2), (1, length),
            mode='bicubic', align_corners=True
        ).squeeze(2).squeeze(1)

    def curve(self, x, func, depth):
        x_ind = (torch.clamp(x, 0, 1) * (depth - 1))
        x_ind = x_ind.round_().long().flatten(1).detach()
        out = torch.gather(func, 1, x_ind)
        return out.reshape(x.size())


