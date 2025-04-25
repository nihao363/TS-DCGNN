import copy
import time
import torch
from tqdm import tqdm
from sklearn import metrics
import utils
import numpy as np

log = utils.get_logger()

class Coach:
    def __init__(self, trainset, devset, testset, model, opt, sched, args):
        self.trainset = trainset
        self.devset = devset
        self.testset = testset
        self.model = model
        self.opt = opt
        self.scheduler = sched
        self.args = args

        # Dataset label dictionary
        self.dataset_label_dict = {
            "seed_eeg": {"ang": 0, "hap": 1, "sad": 2, "neu": 3},  # Assuming 4 emotion categories
        }

        self.label_to_idx = self.dataset_label_dict[args.dataset]

        self.best_dev_f1 = None
        self.best_epoch = None
        self.best_state = None

        # Set random seed for reproducibility
        self.set_seed(args.seed)

    def set_seed(self, seed):
        """ Set random seed for reproducibility """
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        np.random.seed(seed)
        log.info(f"Set random seed to {seed}")

    def load_ckpt(self, ckpt):
        """ Load checkpoint to restore the best model state """
        self.best_dev_f1 = ckpt["best_dev_f1"]
        self.best_epoch = ckpt["best_epoch"]
        self.best_state = ckpt["best_state"]
        self.model.load_state_dict(self.best_state)
        print("Loaded best model.....")

    def train(self):
        """ Train the model, calculate training loss, F1 score for dev and test sets """
        log.debug(self.model)
        best_dev_f1, best_epoch, best_state = (
            self.best_dev_f1,
            self.best_epoch,
            self.best_state,
        )

        dev_f1s = []
        test_f1s = []
        train_losses = []
        best_test_f1 = None

        # Training loop
        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch(epoch)
            dev_f1, dev_loss = self.evaluate()
            self.scheduler.step(dev_loss)
            test_f1, _ = self.evaluate(test=True)

            if best_dev_f1 is None or dev_f1 > best_dev_f1:
                best_dev_f1 = dev_f1
                best_test_f1 = test_f1
                best_epoch = epoch
                best_state = copy.deepcopy(self.model.state_dict())

                torch.save(
                    {"args": self.args, "state_dict": self.model},
                    "model_checkpoints/"
                    + self.args.dataset
                    + "_best_dev_f1_model_seed{}_".format(self.args.seed)
                    + self.args.modalities
                    + ".pt",
                )

                log.info("Save the best model.")
            log.info("[Test set] [f1 {:.4f}]".format(test_f1))

            dev_f1s.append(dev_f1)
            test_f1s.append(test_f1)
            train_losses.append(train_loss)

        # Load the best model and output the best results
        self.model.load_state_dict(best_state)
        log.info("")
        log.info("Best in epoch {}:".format(best_epoch))
        dev_f1, _ = self.evaluate()
        log.info("[Dev set] [f1 {:.4f}]".format(dev_f1))
        test_f1, _ = self.evaluate(test=True)
        log.info("[Test set] f1 {}".format(test_f1))

        return best_dev_f1, best_epoch, best_state, train_losses, dev_f1s, test_f1s

    def train_epoch(self, epoch):
        """ Train one epoch, calculate loss and update the model """
        start_time = time.time()
        epoch_loss = 0
        self.model.train()

        self.trainset.shuffle()
        for idx in tqdm(range(len(self.trainset)), desc="train epoch {}".format(epoch)):
            self.model.zero_grad()
            data = self.trainset[idx]
            for k, v in data.items():
                if not k == "utterance_texts":
                    data[k] = v.to(self.args.device)

            nll = self.model.get_loss(data)
            epoch_loss += nll.item()
            nll.backward()
            self.opt.step()

        end_time = time.time()
        log.info("")
        log.info(
            "[Epoch %d] [Loss: %f] [Time: %f]"
            % (epoch, epoch_loss, end_time - start_time)
        )
        return epoch_loss

    def evaluate(self, test=False):
        """ Evaluate the model performance on dev or test set """
        dev_loss = 0
        dataset = self.testset if test else self.devset
        self.model.eval()
        with torch.no_grad():
            golds = []
            preds = []
            for idx in tqdm(range(len(dataset)), desc="test" if test else "dev"):
                data = dataset[idx]
                golds.append(data["label_tensor"])
                for k, v in data.items():
                    if not k == "utterance_texts":
                        data[k] = v.to(self.args.device)
                y_hat = self.model(data)
                preds.append(y_hat.detach().to("cpu"))

                nll = self.model.get_loss(data)
                dev_loss += nll.item()

            golds = torch.cat(golds, dim=-1).numpy()
            preds = torch.cat(preds, dim=-1).numpy()
            f1 = metrics.f1_score(golds, preds, average="weighted")

            if test:
                print(
                    metrics.classification_report(
                        golds, preds, target_names=self.label_to_idx.keys(), digits=4
                    )
                )

        return f1, dev_loss
