import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl

from colormap import apply_colormap, dxy_colormap


class LogColorDepthMapCallback(pl.Callback):
    def __init__(self, step=100):
        super().__init__()
        self.step = step

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        if trainer.global_step % self.step != 0 or not trainer.is_global_zero:
            return

        pl_module.eval()
        for name, p in pl_module.named_parameters():
            pl_module.logger.experiment.add_histogram(
                name, p, global_step=trainer.global_step
            )

        with torch.no_grad():
            pred = pl_module(batch)

            # disp_gt/disp_pred/left/right/
            gt = (torch.clip(batch["disp"] / 192, 0, 1) * 255).long()
            gt = apply_colormap(gt)
            d = (torch.clip(pred["disp"] / 192, 0, 1) * 255).long()
            d = apply_colormap(d)
            gt = torch.stack((gt, d, batch["left"], batch["right"]), dim=0)
            gt = gt.flatten(end_dim=1)
            gt = torchvision.utils.make_grid(gt, nrow=d.size(0))
            pl_module.logger.experiment.add_image(
                f"all",
                gt,
                global_step=trainer.global_step,
            )

            # multi scale
            for ids, d in enumerate(pred.get("multi_scale", [])):
                tile_size = pred.get("tile_size", 1)
                scale = batch["disp"].size(3) // d.size(3)
                scale_disp = max(1, scale // tile_size)
                d = (torch.clip(d * scale_disp / 192, 0, 1) * 255).long()
                d = apply_colormap(d)
                d = torchvision.utils.make_grid(d, nrow=d.size(0))
                pl_module.logger.experiment.add_image(
                    f"disp_{ids}",
                    d,
                    global_step=trainer.global_step,
                )

            # dxy_pred
            for ids, (d, dxy) in enumerate(pred.get("slant", [])):
                dxy = dxy_colormap(dxy)
                dxy = torchvision.utils.make_grid(dxy, nrow=dxy.size(0))
                pl_module.logger.experiment.add_image(
                    f"dxy_{ids}",
                    dxy,
                    global_step=trainer.global_step,
                )

            # init_disp
            for ids, d in enumerate(pred.get("init_disp", [])):
                tile_size = pred.get("tile_size", 1)
                scale = batch["disp"].size(3) // d.size(3)
                scale_disp = max(1, scale // tile_size)
                d = (torch.clip(d * scale_disp / 192, 0, 1) * 255).long()
                d = apply_colormap(d)
                d = torchvision.utils.make_grid(d, nrow=d.size(0))
                pl_module.logger.experiment.add_image(
                    f"init_disp_{ids}",
                    d,
                    global_step=trainer.global_step,
                )

            # dxy_gt
            if "dxy" in batch:
                dxy = dxy_colormap(batch["dxy"])
                dxy = torchvision.utils.make_grid(dxy, nrow=dxy.size(0))
                pl_module.logger.experiment.add_image(
                    f"dxy_gt",
                    dxy,
                    global_step=trainer.global_step,
                )

            # select
            for ids, sel in enumerate(pred.get("select", [])):
                w0, d0 = sel[0]
                w1, d1 = sel[1]
                w = torchvision.utils.make_grid((w0 > w1).float(), nrow=w0.size(0))
                pl_module.logger.experiment.add_image(
                    f"select_{ids}",
                    w,
                    global_step=trainer.global_step,
                )

        pl_module.logger.experiment.flush()
        pl_module.train()
