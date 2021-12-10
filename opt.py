import argparse


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_disp", type=int, default=192)
    parser.add_argument("--max_disp_val", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2021)

    parser.add_argument("--log_dir", type=str, default="logs")
    parser.add_argument("--exp_name", type=str)
    parser.add_argument("--model", type=str, default="HITNet")
    parser.add_argument("--pretrain", type=str, default=None)

    parser.add_argument("--data_augmentation", type=int, required=True)

    parser.add_argument("--data_type_train", type=str, nargs="+")
    parser.add_argument("--data_root_train", type=str, nargs="+")
    parser.add_argument("--data_list_train", type=str, nargs="+")
    parser.add_argument("--data_size_train", type=int, nargs=2, required=True)

    parser.add_argument("--data_type_val", type=str, nargs="+")
    parser.add_argument("--data_root_val", type=str, nargs="+")
    parser.add_argument("--data_list_val", type=str, nargs="+")
    parser.add_argument("--data_size_val", type=int, nargs=2, required=True)

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--batch_size_val", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--num_workers_val", type=int, default=2)
    parser.add_argument(
        "--optmizer", type=str, default="Adam", choices=["SGD", "Adam", "RMS"]
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_decay", type=float, nargs="*", default=[])
    parser.add_argument(
        "--lr_decay_type", type=str, default="Lambda", choices=["Lambda", "Step"]
    )
    parser.add_argument("--weight_decay", type=float, default=0)

    parser.add_argument("--HITTI_A", type=float, default=1)
    parser.add_argument("--HITTI_B", type=float, default=1)
    parser.add_argument("--HITTI_C1", type=float, default=1)
    parser.add_argument("--HITTI_C2", type=float, default=1.5)
    parser.add_argument("--robust_loss_a", type=float, default=0.8)
    parser.add_argument("--robust_loss_c", type=float, default=0.5)
    parser.add_argument("--init_loss_k", type=int, default=1)
    return parser
