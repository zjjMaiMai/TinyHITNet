import re
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


def np2torch(x, t=True, bgr=False):
    if len(x.shape) == 2:
        x = x[..., None]
    if bgr:
        x = x[..., [2, 1, 0]]
    if t:
        x = np.transpose(x, (2, 0, 1))
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255
    x = torch.from_numpy(x.copy())
    return x


def readPFM(file):
    file = open(file, "rb")

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode().rstrip()
    if header == "PF":
        color = True
    elif header == "Pf":
        color = False
    else:
        raise FileNotFoundError("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise FileNotFoundError("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data


def crop_and_pad(data: dict, crop_size, training):
    src_h, src_w = list(data.values())[0].shape[-2:]
    dst_w, dst_h = crop_size

    # top, bottom, left, right
    pad = [0, 0, 0, 0]
    if src_w < dst_w:
        p = dst_w - src_w
        pad[2] = p // 2
        pad[3] = p - p // 2
    if src_h < dst_h:
        p = dst_h - src_h
        pad[0] = p // 2
        pad[1] = p - p // 2

    if any(pad):
        for k in data.keys():
            value = 0
            if k == "disp":
                value = -1
            data[k] = F.pad(data[k], (pad[2], pad[3], pad[0], pad[1]), value=value)

    src_h, src_w = list(data.values())[0].shape[-2:]
    # top, bottom, left, right
    crop = [0, 0, 0, 0]
    if training:
        crop[2] = np.random.randint(0, src_w - dst_w + 1)
        crop[0] = np.random.randint(0, src_h - dst_h + 1)
    else:
        crop[2] = (src_w - dst_w) // 2
        crop[0] = (src_h - dst_h) // 2
    crop[3] = crop[2] + dst_w
    crop[1] = crop[0] + dst_h

    for k in data.keys():
        data[k] = data[k][:, crop[0] : crop[1], crop[2] : crop[3]]
    return data


def augmentation(data: dict, training):
    if not training:
        return data

    symmetric = np.random.uniform(0.8, 1.2)
    asymmetric = np.random.uniform(0.95, 1.05, size=2)
    data["left"] = TF.adjust_brightness(data["left"], symmetric * asymmetric[0])
    data["right"] = TF.adjust_brightness(data["right"], symmetric * asymmetric[1])

    symmetric = np.random.uniform(0.8, 1.2)
    asymmetric = np.random.uniform(0.95, 1.05, size=2)
    data["left"] = TF.adjust_contrast(data["left"], symmetric * asymmetric[0])
    data["right"] = TF.adjust_contrast(data["right"], symmetric * asymmetric[1])

    if np.random.rand() > 0.5:
        crop_w = np.random.randint(50, 250)
        crop_h = np.random.randint(50, 180)

        h, w = data["right"].size()[-2:]
        src_w = np.random.randint(0, w - crop_w + 1)
        src_h = np.random.randint(0, h - crop_h + 1)
        dst_w = np.random.randint(0, w - crop_w + 1)
        dst_h = np.random.randint(0, h - crop_h + 1)
        data["right"][:, dst_h : dst_h + crop_h, dst_w : dst_w + crop_w] = data[
            "right"
        ][:, src_h : src_h + crop_h, src_w : src_w + crop_w].clone()

    # if np.random.rand() > 0.5:
    #     std = np.random.uniform(0, 5, size=2) ** 0.5 / 255
    #     left_noise = torch.normal(mean=0, std=std[0], size=data["left"].size())
    #     right_noise = torch.normal(mean=0, std=std[1], size=data["right"].size())
    #     data["left"] = torch.clip(data["left"] + left_noise, 0, 1)
    #     data["right"] = torch.clip(data["right"] + right_noise, 0, 1)
    return data
