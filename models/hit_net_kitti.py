import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import math


def same_padding_conv(x, w, b, s):
    out_h = math.ceil(x.size(2) / s[0])
    out_w = math.ceil(x.size(3) / s[1])

    pad_h = max((out_h - 1) * s[0] + w.size(2) - x.size(2), 0)
    pad_w = max((out_w - 1) * s[1] + w.size(3) - x.size(3), 0)
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    x = F.conv2d(x, w, b, stride=s)
    return x


class SameConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return same_padding_conv(x, self.weight, self.bias, self.stride)


class UpsampleBlock(nn.Module):
    def __init__(self, c0, c1):
        super().__init__()
        self.up_conv = nn.Sequential(
            nn.ConvTranspose2d(c0, c1, 2, 2),
            nn.LeakyReLU(0.2),
        )
        self.merge_conv = nn.Sequential(
            nn.Conv2d(c1 * 2, c1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c1, c1, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, input, sc):
        x = self.up_conv(input)
        x = torch.cat((x, sc), dim=1)
        x = self.merge_conv(x)
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.down_0 = nn.Sequential(
            nn.Conv2d(3, C[0], 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.down_1 = nn.Sequential(
            SameConv2d(C[0], C[1], 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(C[1], C[1], 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.down_2 = nn.Sequential(
            SameConv2d(C[1], C[2], 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(C[2], C[2], 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.down_3 = nn.Sequential(
            SameConv2d(C[2], C[3], 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(C[3], C[3], 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.down_4 = nn.Sequential(
            SameConv2d(C[3], C[4], 4, 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(C[4], C[4], 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(C[4], C[4], 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(C[4], C[4], 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.up_3 = UpsampleBlock(C[4], C[3])
        self.up_2 = UpsampleBlock(C[3], C[2])
        self.up_1 = UpsampleBlock(C[2], C[1])
        self.up_0 = UpsampleBlock(C[1], C[0])

    def forward(self, input):
        x0 = self.down_0(input)
        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        o0 = self.down_4(x3)
        o1 = self.up_3(o0, x3)
        o2 = self.up_2(o1, x2)
        o3 = self.up_1(o2, x1)
        o4 = self.up_0(o3, x0)
        return o4, o3, o2, o1, o0


@functools.lru_cache()
@torch.no_grad()
def make_warp_coef(scale, device):
    center = (scale - 1) / 2
    index = torch.arange(scale, device=device) - center
    coef_y, coef_x = torch.meshgrid(index, index)
    coef_x = coef_x.reshape(1, -1, 1, 1)
    coef_y = coef_y.reshape(1, -1, 1, 1)
    return coef_x, coef_y


def disp_up(d, dx, dy, scale, tile_expand):
    n, _, h, w = d.size()
    coef_x, coef_y = make_warp_coef(scale, d.device)

    if tile_expand:
        d = d + coef_x * dx + coef_y * dy
    else:
        d = d * scale + coef_x * dx * 4 + coef_y * dy * 4

    d = d.reshape(n, 1, scale, scale, h, w)
    d = d.permute(0, 1, 4, 2, 5, 3)
    d = d.reshape(n, 1, h * scale, w * scale)
    return d


def hyp_up(hyp, scale=1, tile_scale=1):
    if scale != 1:
        d = disp_up(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], scale, tile_expand=False)
        p = F.interpolate(hyp[:, 1:], scale_factor=scale)
        hyp = torch.cat((d, p), dim=1)
    if tile_scale != 1:
        d = disp_up(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], tile_scale, tile_expand=True)
        p = F.interpolate(hyp[:, 1:], scale_factor=tile_scale)
        hyp = torch.cat((d, p), dim=1)
    return hyp


def warp_and_aggregate(hyp, left, right):
    scale = left.size(3) // hyp.size(3)
    assert scale == 4

    d_expand = disp_up(hyp[:, :1], hyp[:, 1:2], hyp[:, 2:3], scale, tile_expand=True)
    d_range = torch.arange(right.size(3), device=right.device)
    d_range = d_range.view(1, 1, 1, -1) - d_expand
    d_range = d_range.repeat(1, right.size(1), 1, 1)

    cost = [torch.sum(torch.abs(left), dim=1, keepdim=True)]
    for offset in [1, 0, -1]:
        index_float = d_range + offset
        index_long = torch.floor(index_float).long()
        index_left = torch.clip(index_long, min=0, max=right.size(3) - 1)
        index_right = torch.clip(index_long + 1, min=0, max=right.size(3) - 1)
        index_weight = index_float - index_left

        right_warp_left = torch.gather(right, dim=-1, index=index_left.long())
        right_warp_right = torch.gather(right, dim=-1, index=index_right.long())
        right_warp = right_warp_left + index_weight * (
            right_warp_right - right_warp_left
        )
        cost.append(torch.sum(torch.abs(left - right_warp), dim=1, keepdim=True))
    cost = torch.cat(cost, dim=1)

    n, c, h, w = cost.size()
    cost = cost.reshape(n, c, h // scale, scale, w // scale, scale)
    cost = cost.permute(0, 3, 5, 1, 2, 4)
    cost = cost.reshape(n, scale * scale * c, h // scale, w // scale)
    return cost


class ResBlock(nn.Module):
    def __init__(self, c0, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c0, c0, 3, 1, dilation, dilation),
            nn.LeakyReLU(0.2),
            nn.Conv2d(c0, c0, 3, 1, dilation, dilation),
        )
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, input):
        x = self.conv(input)
        x = x + input
        x = self.relu(x)
        return x


class LevalProp(nn.Module):
    def __init__(self, h_size=2):
        super().__init__()
        self.conv_neighbors = nn.Sequential(
            nn.Conv2d(64 * h_size, 16 * h_size, 1),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(32 * h_size, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.res_block = nn.Sequential(ResBlock(32), ResBlock(32))
        self.convn = nn.Conv2d(32, 17 * h_size, 3, 1, 1)
        self.h_size = h_size

    def forward(self, hyps, l, r):
        cost = [warp_and_aggregate(h, l, r) for h in hyps]
        cost = torch.cat(cost, dim=1)
        x = self.conv_neighbors(cost)
        hyps = torch.cat(hyps, dim=1)
        x = torch.cat((hyps, x), dim=1)
        x = self.conv1(x)
        x = self.res_block(x)
        x = self.convn(x)

        dh = x[:, : 16 * self.h_size]
        w = x[:, 16 * self.h_size :]
        return hyps + dh, w


def make_cost_volume_v2(left, right, max_disp):
    d_range = torch.arange(max_disp, device=left.device)
    d_range = d_range.view(1, 1, -1, 1, 1)

    x_index = torch.arange(left.size(3), device=left.device)
    x_index = x_index.view(1, 1, 1, 1, -1)

    x_index = torch.clip(4 * x_index + 1 - d_range, 0, right.size(3) - 1).repeat(
        right.size(0), right.size(1), 1, right.size(2), 1
    )
    right = torch.gather(
        right.unsqueeze(2).repeat(1, 1, max_disp, 1, 1), dim=-1, index=x_index
    )

    return left.unsqueeze(2) - right


class LevelInit(nn.Module):
    def __init__(self, cin, max_disp, cref=16):
        super().__init__()
        self.max_disp = max_disp
        self.conv_reduce = nn.Conv2d(cin, 16, 4)
        self.conv_em = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 16, 1),
            nn.LeakyReLU(0.2),
        )
        self.conv_hyp = nn.Sequential(
            nn.Conv2d(cref + 1, 13, 1),
            nn.LeakyReLU(0.2),
        )

    def forward(self, l, r, ref=None):
        lt = F.conv2d(
            l,
            self.conv_reduce.weight,
            self.conv_reduce.bias,
            stride=(4, 4),
        )
        lt = self.conv_em(lt)

        rt = same_padding_conv(
            r,
            self.conv_reduce.weight,
            self.conv_reduce.bias,
            s=(4, 1),
        )
        rt = self.conv_em(rt)

        cv = make_cost_volume_v2(lt, rt, self.max_disp)
        cv = torch.norm(cv, p=1, dim=1)
        cv_min, d = torch.min(cv, dim=1, keepdim=True)
        d = d.float()

        if ref is None:
            ref = lt
        p = torch.cat((cv_min, ref), dim=1)
        p = self.conv_hyp(p)
        p = torch.cat((d, torch.zeros_like(d), torch.zeros_like(d), p), dim=1)
        return p, cv


class Level(nn.Module):
    def __init__(self, cin, max_disp, h_size, cref=16):
        super().__init__()
        self.h_size = h_size
        self.init = LevelInit(cin, max_disp, cref)
        self.prop = LevalProp(self.h_size)

    def forward(self, l, r, h=None, ref=None):
        hi, cv = self.init(l, r, ref)
        if self.h_size == 1:
            h, w = self.prop([hi], l, r)
            return h, cv, hi[:, :1], []
        else:
            h, w = self.prop([hi, h], l, r)
            h0 = h[:, :16]
            h1 = h[:, 16:]
            w0 = w[:, :1]
            w1 = w[:, 1:]
            h = torch.where(w0 > w1, h0, h1)
            return h, cv, hi[:, :1], [[w0, h0[:, :1]], [w1, h1[:, :1]]]


class Refine(nn.Module):
    def __init__(self, cin, cres, dilations):
        super().__init__()
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(cin + 16, cres, 1),
            nn.LeakyReLU(0.2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(cres, cres, 3, 1, 1),
            nn.LeakyReLU(0.2),
        )
        self.res_block = []
        for d in dilations:
            self.res_block += [ResBlock(cres, d)]
        self.res_block = nn.Sequential(*self.res_block)
        self.convn = nn.Conv2d(cres, 16, 3, 1, 1)

    def forward(self, hpy, left):
        x = torch.cat((left, hpy), dim=1)
        x = self.conv1x1(x)
        x = self.conv1(x)
        x = self.res_block(x)
        x = self.convn(x)
        return hpy + x


class HITNet_KITTI(nn.Module):
    def __init__(self):
        super().__init__()
        self.align = 64
        self.max_disp = 256

        num_feature = [16, 16, 24, 24, 32]
        self.feature_extractor = FeatureExtractor(num_feature)

        self.level = nn.ModuleList(
            [
                Level(num_feature[-1], self.max_disp // 16, 1),
                Level(num_feature[-2], self.max_disp // 8, 2),
                Level(num_feature[-3], self.max_disp // 4, 2, num_feature[-1]),
                Level(num_feature[-4], self.max_disp // 2, 2, num_feature[-2]),
                Level(num_feature[-5], self.max_disp // 1, 2, num_feature[-3]),
            ]
        )

        self.refine = nn.ModuleList(
            [
                Refine(num_feature[-3], 32, [1, 3, 1, 1]),
                Refine(num_feature[-4], 32, [1, 3, 1, 1]),
                Refine(num_feature[-5], 16, [1, 1]),
            ]
        )


    def forward(self, left_img, right_img):
        n, c, h, w = left_img.size()
        w_pad = (self.align - (w % self.align)) % self.align
        h_pad = (self.align - (h % self.align)) % self.align

        left_img = F.pad(left_img, (0, w_pad, 0, h_pad))
        right_img = F.pad(right_img, (0, w_pad, 0, h_pad))

        lf = self.feature_extractor(left_img)
        rf = self.feature_extractor(right_img)

        h0, cv0, di0, wp0 = self.level[0](lf[-1], rf[-1])
        h1, cv1, di1, wp1 = self.level[1](lf[-2], rf[-2], hyp_up(h0, 2, 1))
        h2, cv2, di2, wp2 = self.level[2](lf[-3], rf[-3], hyp_up(h1, 2, 1), lf[-1])
        h3, cv3, di3, wp3 = self.level[3](lf[-4], rf[-4], hyp_up(h2, 2, 1), lf[-2])
        h4, cv4, di4, wp4 = self.level[4](lf[-5], rf[-5], hyp_up(h3, 2, 1), lf[-3])

        h5 = self.refine[0](h4, lf[-3])
        h6 = self.refine[1](hyp_up(h5, 1, 2), lf[-4])
        h7 = self.refine[2](hyp_up(h6, 1, 2), lf[-5])[:, :, :h, :w]

        return {
            "tile_size": 4,
            "disp": h7[:, 0:1],
            "multi_scale": [
                h0[:, 0:1],
                h1[:, 0:1],
                h2[:, 0:1],
                h3[:, 0:1],
                h4[:, 0:1],
                h5[:, 0:1],
                h6[:, 0:1],
                h7[:, 0:1],
            ],
            "cost_volume": [cv0, cv1, cv2, cv3, cv4],
            "slant": [
                [h0[:, 0:1], h0[:, 1:3]],
                [h1[:, 0:1], h1[:, 1:3]],
                [h2[:, 0:1], h2[:, 1:3]],
                [h3[:, 0:1], h3[:, 1:3]],
                [h4[:, 0:1], h4[:, 1:3]],
                [h5[:, 0:1], h5[:, 1:3]],
                [h6[:, 0:1], h6[:, 1:3]],
                [h7[:, 0:1], h7[:, 1:3]],
            ],
            "init_disp": [di0, di1, di2, di3, di4],
            "select": [wp1, wp2, wp3, wp4],
        }


if __name__ == "__main__":
    import cv2
    from thop import profile

    left = torch.rand(1, 3, 375, 1242)
    right = torch.rand(1, 3, 375, 1242)
    model = HITNet_KITTI()

    print(model(left, right)["disp"].size())

    total_ops, total_params = profile(
        model,
        (
            left,
            right,
        ),
    )
    print(
        "{:.4f} MACs(G)\t{:.4f} Params(M)".format(
            total_ops / (1000 ** 3), total_params / (1000 ** 2)
        )
    )
