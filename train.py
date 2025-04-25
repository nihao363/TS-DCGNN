import argparse
import torch
import os
import utils

import Dataset
import functions
from Model import D2GNN
from Coach import Coach

log = utils.get_logger()

class EEG_Sample:
    def __init__(self, vid, speaker, label, eeg_data):
        self.vid = vid  # EEG 数据集的样本 ID
        self.speaker = speaker  # 说话人 ID
        self.label = label  # 情绪标签
        self.eeg_data = eeg_data  # EEG 数据（多通道时间序列数据）

def main(args):
    utils.set_seed(args.seed)

    args.data = os.path.join(
        args.data_dir_path, args.dataset, "data_" + args.dataset + ".pkl"
    )

    # 加载数据
    log.debug("Loading data from '%s'." % args.data)

    data = utils.load_pkl(args.data)
    log.info("Loaded data.")

    # 使用 EEG 数据集类加载数据
    trainset = EEGDataset.EEGDataset(data["train"], args)
    devset = EEGDataset.EEGDataset(data["dev"], args)
    testset = EEGDataset.EEGDataset(data["test"], args)

    log.debug("Building model...")

    model_file = "./model_checkpoints/model.pt"
    model = D2GNN(args).to(args.device)  # 使用 D2GNN 模型
    opt = functions.Optim(args.learning_rate, args.max_grad_value, args.weight_decay)
    opt.set_parameters(model.parameters(), args.optimizer)
    sched = opt.get_scheduler(args.scheduler)

    coach = Coach(trainset, devset, testset, model, opt, sched, args)
    if not args.from_begin:
        ckpt = torch.load(model_file)
        coach.load_ckpt(ckpt)
        print("Training from checkpoint...")

    # 开始训练
    log.info("Start training...")
    ret = coach.train()

    # 保存训练好的模型
    checkpoint = {
        "best_dev_f1": ret[0],
        "best_epoch": ret[1],
        "best_state": ret[2],
    }
    torch.save(checkpoint, model_file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="EEG Emotion Recognition Training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="eeg_emotion",  # 修改为 EEG 数据集名称
        help="Dataset name: eeg_emotion",
    )
    parser.add_argument(
        "--data_dir_path", type=str, help="Dataset directory path", default="./data"
    )
    parser.add_argument(
        "--from_begin", action="store_false", help="Training from begin.", default=True
    )
    parser.add_argument("--device", type=str, default="cuda", help="Computing device.")
    parser.add_argument(
        "--epochs", default=50, type=int, help="Number of training epochs."
    )
    parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["sgd", "rmsprop", "adam", "adamw"],
        help="Name of optimizer.",
    )
    parser.add_argument(
        "--scheduler", type=str, default="reduceLR", help="Name of scheduler."
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0002, help="Learning rate."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-8, help="Weight decay."
    )
    parser.add_argument(
        "--max_grad_value",
        default=-1,
        type=float,
        help="""If the norm of the gradient vector exceeds this,
                        normalize it to have the norm equal to max_grad_norm""",
    )
    parser.add_argument("--drop_rate", type=float, default=0.5, help="Dropout rate.")
    parser.add_argument(
        "--wp",
        type=int,
        default=5,
        help="Past context window size. Set wp to -1 to use all the past context.",
    )
    parser.add_argument(
        "--wf",
        type=int,
        default=5,
        help="Future context window size. Set wp to -1 to use all the future context.",
    )
    parser.add_argument(
        "--n_classes", type=int, default=6, help="Number of emotion classes."
    )
    parser.add_argument("--hidden_size", type=int, default=128, help="Hidden size of GCN.")
    parser.add_argument("--rnn", type=str, default="lstm", help="RNN type (lstm, gru).")
    parser.add_argument(
        "--class_weight", action="store_true", default=False, help="Use class weights in nll loss."
    )
    parser.add_argument("--seqcontext_nlayer", type=int, default=2)
    parser.add_argument("--gnn_nheads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=144, help="Random seed.")

    args = parser.parse_args()

    # 设置 EEG 数据集的嵌入维度（根据 EEG 数据的通道数和时间维度设置）
    args.dataset_embedding_dims = {
        "eeg_emotion": {
            "eeg": 256,  # 假设 EEG 数据的特征维度是 256
        },
    }

    log.debug(args)

    main(args)
