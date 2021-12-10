import torch
import torch.nn as nn
import torch.nn.functional as F


def robust_loss(x, a, c):
    abs_a_sub_2 = abs(a - 2)

    x = x / c
    x = x * x / abs_a_sub_2 + 1
    x = x ** (a / 2)
    x = x - 1
    x = x * abs_a_sub_2 / a
    return x


def calc_init_loss(cv, target, max_disp, k=1, tile_size=1):
    scale = target.size(3) // cv.size(3)
    scale_disp = max(1, scale // tile_size)

    target = target / scale_disp
    max_disp = max_disp / scale_disp

    target = F.max_pool2d(target, kernel_size=scale, stride=scale)
    mask = (target < max_disp) & (target > 1e-3)

    def rho(d):  # ρ(d)
        d = torch.clip(d, 0, cv.size(1) - 1)
        return torch.gather(cv, dim=1, index=d)

    def phi(d):  # φ(d)
        df = torch.floor(d).long()
        d_sub_df = d - df
        return d_sub_df * rho(df + 1) + (1 - d_sub_df) * rho(df)

    pixels = mask.sum() + 1e-6
    gt_loss = (phi(target) * mask).sum() / pixels

    d_range = torch.arange(0, max_disp, dtype=target.dtype, device=target.device)
    d_range = d_range.view(1, -1, 1, 1)
    d_range = d_range.repeat(target.size(0), 1, target.size(2), target.size(3))

    low = target - 1.5
    up = target + 1.5
    d_range_mask = (low <= d_range) & (d_range <= up) | (~mask)

    cv_nm = torch.masked_fill(cv, d_range_mask, float("inf"))
    cost_nm = torch.topk(cv_nm, k=k, dim=1, largest=False).values

    nm_loss = torch.clip(1 - cost_nm, min=0)
    nm_loss = (nm_loss * mask).sum() / pixels
    return gt_loss + nm_loss


def calc_multi_scale_loss(pred, target, max_disp, a=0.8, c=0.5, A=1, tile_size=1):
    scale = target.size(3) // pred.size(3)
    scale_disp = max(1, scale // tile_size)

    target = target / scale_disp
    max_disp = max_disp / scale_disp

    target = F.max_pool2d(target, kernel_size=scale, stride=scale)
    mask = (target < max_disp) & (target > 1e-3)
    diff = (pred - target).abs()

    if tile_size > 1 and scale_disp > 1:
        mask = (diff < A) & mask
    loss = robust_loss(diff, a=a, c=c)
    return (loss * mask).sum() / (mask.sum() + 1e-6)


def calc_slant_loss(dxy, dxy_gt, pred, target, max_disp, B=1, tile_size=1):
    scale = target.size(3) // pred.size(3)
    scale_disp = max(1, scale // tile_size)

    target = target / scale_disp
    max_disp = max_disp / scale_disp

    target, index = F.max_pool2d(
        target, kernel_size=scale, stride=scale, return_indices=True
    )
    mask = (target < max_disp) & (target > 1e-3)
    diff = (pred - target).abs()

    def retrieve_elements_from_indices(tensor, indices):
        flattened_tensor = tensor.flatten(start_dim=2)
        output = flattened_tensor.gather(
            dim=2, index=indices.flatten(start_dim=2)
        ).view_as(indices)
        return output

    dxy_gt = retrieve_elements_from_indices(dxy_gt, index.repeat(1, 2, 1, 1))

    mask = (diff < B) & mask
    loss = (dxy - dxy_gt).abs()
    return (loss * mask).sum() / (mask.sum() + 1e-6)


def calc_w_loss(w, pred, target, max_disp, C1=1, C2=1.5, tile_size=1):
    scale = target.size(3) // pred.size(3)
    scale_disp = max(1, scale // tile_size)

    target = target / scale_disp
    max_disp = max_disp / scale_disp

    target = F.max_pool2d(target, kernel_size=scale, stride=scale)
    mask = (target < max_disp) & (target > 1e-3)
    diff = (pred - target).abs()

    mask_c1 = (diff < C1) & mask
    loss_c1 = torch.clip(1 - w, min=0)
    loss_c1 = (loss_c1 * mask_c1).sum() / (mask_c1.sum() + 1e-6)

    mask_c2 = (diff > C2) & mask
    loss_c2 = torch.clip(w, min=0)
    loss_c2 = (loss_c2 * mask_c2).sum() / (mask_c2.sum() + 1e-6)
    return loss_c1 + loss_c2


def calc_loss(pred, batch, args):
    loss_dict = {}
    tile_size = pred.get("tile_size", 1)

    # multi scale loss
    for ids, d in enumerate(pred.get("multi_scale", [])):
        loss_dict[f"disp_loss_{ids}"] = calc_multi_scale_loss(
            d,
            batch["disp"],
            args.max_disp,
            a=args.robust_loss_a,
            c=args.robust_loss_c,
            A=args.HITTI_A,
            tile_size=tile_size,
        )

    # init loss
    for ids, cv in enumerate(pred.get("cost_volume", [])):
        loss_dict[f"init_loss_{ids}"] = calc_init_loss(
            cv,
            batch["disp"],
            args.max_disp,
            k=args.init_loss_k,
            tile_size=tile_size,
        )

    # slant loss
    for ids, (d, dxy) in enumerate(pred.get("slant", [])):
        loss_dict[f"slant_loss_{ids}"] = calc_slant_loss(
            dxy,
            batch["dxy"],
            d,
            batch["disp"],
            args.max_disp,
            B=args.HITTI_B,
            tile_size=tile_size,
        )

    # select loss
    for ids, sel in enumerate(pred.get("select", [])):
        w0, d0 = sel[0]
        w1, d1 = sel[1]
        loss_0 = calc_w_loss(
            w0,
            d0,
            batch["disp"],
            args.max_disp,
            C1=args.HITTI_C1,
            C2=args.HITTI_C2,
            tile_size=tile_size,
        )
        loss_1 = calc_w_loss(
            w1,
            d1,
            batch["disp"],
            args.max_disp,
            C1=args.HITTI_C1,
            C2=args.HITTI_C2,
            tile_size=tile_size,
        )
        loss_dict[f"select_loss_{ids}"] = loss_0 + loss_1

    return loss_dict


if __name__ == "__main__":
    cv = torch.rand(1, 320, 144, 240)
    target = torch.rand(1, 1, 576, 960) * 400
    calc_init_loss(cv, target, 320)
