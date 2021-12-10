import torch
import torch.nn as nn
import torch.nn.functional as F


def make_cost_volume(left, right, max_disp):
    cost_volume = torch.ones(
        (left.size(0), left.size(1), max_disp, left.size(2), left.size(3)),
        dtype=left.dtype,
        device=left.device,
    )

    cost_volume[:, :, 0, :, :] = left - right
    for d in range(1, max_disp):
        cost_volume[:, :, d, :, d:] = left[:, :, :, d:] - right[:, :, :, :-d]

    return cost_volume


def conv_3x3(in_c, out_c, s=1, d=1):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, s, d, dilation=d, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )


def conv_1x1(in_c, out_c):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 1, bias=False),
        nn.BatchNorm2d(out_c),
        nn.ReLU(),
    )


class ResBlock(nn.Module):
    def __init__(self, c0, dilation=1):
        super().__init__()
        self.conv = nn.Sequential(
            conv_3x3(c0, c0, d=dilation),
            conv_3x3(c0, c0, d=dilation),
        )

    def forward(self, input):
        x = self.conv(input)
        return x + input


class RefineNet(nn.Module):
    def __init__(self):
        super().__init__()
        d = [1, 2, 4, 8, 1, 1]
        self.conv0 = nn.Sequential(
            conv_3x3(4, 32),
            *[ResBlock(32, d[i]) for i in range(6)],
            nn.Conv2d(32, 1, 3, 1, 1),
        )

    def forward(self, disp, rgb):
        disp = (
            F.interpolate(disp, scale_factor=2, mode="bilinear", align_corners=False)
            * 2
        )
        rgb = F.interpolate(
            rgb, (disp.size(2), disp.size(3)), mode="bilinear", align_corners=False
        )
        x = torch.cat((disp, rgb), dim=1)
        x = self.conv0(x)
        return F.relu(disp + x)


class StereoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.k = 3
        self.align = 2 ** self.k
        self.max_disp = (192 + 1) // (2 ** self.k)

        self.feature_extractor = [conv_3x3(3, 32, 2), ResBlock(32)]
        for _ in range(self.k - 1):
            self.feature_extractor += [conv_3x3(32, 32, 2), ResBlock(32)]
        self.feature_extractor += [nn.Conv2d(32, 32, 3, 1, 1)]
        self.feature_extractor = nn.Sequential(*self.feature_extractor)

        self.cost_filter = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 32, 3, 1, 1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.Conv3d(32, 1, 3, 1, 1),
        )
        self.refine_layer = nn.ModuleList([RefineNet() for _ in range(self.k)])

    def forward(self, left_img, right_img):
        n, c, h, w = left_img.size()
        w_pad = (self.align - (w % self.align)) % self.align
        h_pad = (self.align - (h % self.align)) % self.align

        left_img = F.pad(left_img, (0, w_pad, 0, h_pad))
        right_img = F.pad(right_img, (0, w_pad, 0, h_pad))

        lf = self.feature_extractor(left_img)
        rf = self.feature_extractor(right_img)

        cost_volume = make_cost_volume(lf, rf, self.max_disp)
        cost_volume = self.cost_filter(cost_volume).squeeze(1)

        x = F.softmax(cost_volume, dim=1)
        d = torch.arange(0, self.max_disp, device=x.device, dtype=x.dtype)
        x = torch.sum(x * d.view(1, -1, 1, 1), dim=1, keepdim=True)

        multi_scale = []
        for refine in self.refine_layer:
            x = refine(x, left_img)
            scale = left_img.size(3) / x.size(3)
            full_res = F.interpolate(x * scale, left_img.shape[2:])[:, :, :h, :w]
            multi_scale.append(full_res)

        return {
            "disp": multi_scale[-1],
            "multi_scale": multi_scale,
        }


if __name__ == "__main__":
    from thop import profile

    left = torch.rand(1, 3, 540, 960)
    right = torch.rand(1, 3, 540, 960)
    model = StereoNet()

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
