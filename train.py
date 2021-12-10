import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from losses import calc_loss
from models import build_model
from dataset import build_dataset
from metrics.epe import EPEMetric
from metrics.rate import RateMetric
from torchmetrics import MetricCollection


class TrainModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.automatic_optimization = False
        self.model = build_model(self.hparams)
        self.max_disp = self.hparams.max_disp
        self.max_disp_val = self.hparams.max_disp_val
        if self.max_disp_val is None:
            self.max_disp_val = self.max_disp

        metric = MetricCollection(
            {
                "epe": EPEMetric(),
                "rate_1": RateMetric(1.0),
                "rate_3": RateMetric(3.0),
            }
        )
        self.train_metric = metric.clone(prefix="train_")
        self.val_metric = metric.clone(prefix="val_")

    def forward(self, batch):
        left = batch["left"] * 2 - 1
        right = batch["right"] * 2 - 1
        return self.model(left, right)

    def training_step(self, batch, batch_idx):
        scheduler = self.lr_schedulers()
        optimizer = self.optimizers()

        pred = self(batch)
        loss_dict = calc_loss(
            pred,
            batch,
            self.hparams,
        )
        loss = sum(loss_dict.values())

        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        scheduler.step()

        mask = (batch["disp"] < self.max_disp) & (batch["disp"] > 1e-3)
        self.train_metric(pred["disp"], batch["disp"], mask)
        self.log_dict(loss_dict, on_step=True)

    def training_epoch_end(self, outputs):
        self.log_dict(self.train_metric.compute(), prog_bar=False)
        self.train_metric.reset()

    def validation_step(self, batch, batch_idx):
        pred = self(batch)
        mask = (batch["disp"] < self.max_disp_val) & (batch["disp"] > 1e-3)
        self.val_metric(pred["disp"], batch["disp"], mask)

    def validation_epoch_end(self, outputs):
        self.log_dict(self.val_metric.compute(), prog_bar=True)
        self.val_metric.reset()

    def configure_optimizers(self):
        if self.hparams.optmizer == "Adam":
            opt = torch.optim.Adam(
                self.model.parameters(),
                lr=self.hparams.lr,
            )
        elif self.hparams.optmizer == "SGD":
            opt = torch.optim.SGD(
                self.model.parameters(),
                lr=self.hparams.lr,
                momentum=0.9,
            )
        elif self.hparams.optmizer == "RMS":
            opt = torch.optim.RMSprop(
                self.model.parameters(),
                lr=self.hparams.lr,
            )
        else:
            raise NotImplementedError

        if self.hparams.lr_decay_type == "Lambda":

            def lr_step(step):
                scale = 1.0
                for s, v in zip(
                    self.hparams.lr_decay[::2], self.hparams.lr_decay[1::2]
                ):
                    if step > s:
                        scale = v
                return scale

            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_step)
        elif self.hparams.lr_decay_type == "Step":
            if not self.hparams.lr_decay:
                self.hparams.lr_decay = [1.0, 1.0]

            scheduler = torch.optim.lr_scheduler.StepLR(
                opt,
                step_size=int(self.hparams.lr_decay[0]),
                gamma=self.hparams.lr_decay[1],
            )
        else:
            raise NotImplementedError

        return [opt], [scheduler]

    def train_dataloader(self):
        dataset = build_dataset(self.hparams, training=True)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size // self.trainer.num_gpus,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )

    def val_dataloader(self):
        dataset = build_dataset(self.hparams, training=False)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.hparams.batch_size_val // self.trainer.num_gpus,
            num_workers=self.hparams.num_workers_val,
            pin_memory=True,
        )


if __name__ == "__main__":
    from opt import build_parser
    from pytorch_lightning.plugins import DDPPlugin
    from pytorch_lightning.callbacks import LearningRateMonitor
    from pytorch_lightning import loggers as pl_loggers
    from callback import LogColorDepthMapCallback

    parser = build_parser()
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    pl.seed_everything(seed=args.seed)

    model = TrainModel(**vars(args))
    if args.pretrain is not None:
        ckpt = torch.load(args.pretrain)
        if "state_dict" in ckpt:
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.model.load_state_dict(ckpt)
            
    trainer = pl.Trainer.from_argparse_args(
        args,
        logger=pl_loggers.TensorBoardLogger(args.log_dir, args.exp_name),
        callbacks=[
            LearningRateMonitor(logging_interval="step"),
            LogColorDepthMapCallback(),
        ],
    )
    trainer.fit(model)
