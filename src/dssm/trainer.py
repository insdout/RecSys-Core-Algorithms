import os
import torch
import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from .callback import EarlyStopper
from .loss_functions import BPRLoss


class MatchTrainer(object):
    """
    Args:
        model (nn.Module): any matching model.
        mode (int, optional): the training mode, `{0:point-wise, 1:pair-wise, 2:list-wise}`. Defaults to 0.
        optimizer_fn (torch.optim): optimizer function of pytorch (default = `torch.optim.Adam`).
        optimizer_params (dict): parameters of optimizer_fn.
        scheduler_fn (torch.optim.lr_scheduler) : torch scheduling class, eg. `torch.optim.lr_scheduler.StepLR`.
        scheduler_params (dict): parameters of optimizer scheduler_fn.
        n_epoch (int): epoch number of training.
        earlystop_patience (int): how long to wait after last time validation auc improved (default=10).
        device (str): `"cpu"` or `"cuda:0"`
        gpus (list): id of multi gpu (default=[]). If the length >=1, then the model will wrapped by nn.DataParallel.
        model_path (str): the path you want to save the model (default="./"). 
    """

    def __init__(
        self,
        model,
        mode=0,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=None,
        scheduler_fn=None,
        scheduler_params=None,
        n_epoch=10,
        earlystop_patience=10,
        device="cpu",
        gpus=None,
        model_path="./",
    ):
        self.model = model
        if gpus is None:
            gpus = []
        self.gpus = gpus
        if len(gpus) > 1:
            print('parallel running on these gpus:', gpus)
            self.model = torch.nn.DataParallel(self.model, device_ids=gpus)
        self.device = torch.device(device)
        self.model.to(self.device)
        if optimizer_params is None:
            optimizer_params = {
                "lr": 1e-3,
                "weight_decay": 1e-5
            }
        self.mode = mode
        if mode == 0:  #point-wise loss, binary cross_entropy
            self.criterion = torch.nn.BCELoss()
        elif mode == 1:  #pair-wise loss
            self.criterion = BPRLoss()
        elif mode == 2:  #list-wise loss, softmax
            self.criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError("mode only contain value in %s, but got %s" % ([0, 1, 2], mode))
        self.optimizer = optimizer_fn(self.model.parameters(), **optimizer_params)
        self.scheduler = None
        if scheduler_fn is not None:
            self.scheduler = scheduler_fn(self.optimizer, **scheduler_params)
        self.evaluate_fn = roc_auc_score
        self.n_epoch = n_epoch
        self.early_stopper = EarlyStopper(patience=earlystop_patience)
        self.model_path = model_path

    def train_one_epoch(self, data_loader, log_interval=10):
        self.model.train()
        total_loss = 0
        tk0 = tqdm.tqdm(data_loader, desc="train", smoothing=0, mininterval=1.0)
        for i, (x_dict, y) in enumerate(tk0):
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
            y = y.to(self.device)
            if self.mode == 0:
                y = y.float()
            else:
                y = y.long()
            if self.mode == 1:
                pos_score, neg_score = self.model(x_dict)
                loss = self.criterion(pos_score, neg_score)
            else:
                y_pred = self.model(x_dict)
                loss = self.criterion(y_pred, y)
            self.model.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            if (i + 1) % log_interval == 0:
                tk0.set_postfix(loss=total_loss / log_interval)
                total_loss = 0

    def fit(self, train_dataloader, val_dataloader=None):
        for epoch_i in range(self.n_epoch):
            print('epoch:', epoch_i)
            self.train_one_epoch(train_dataloader)
            if self.scheduler is not None:
                if epoch_i % self.scheduler.step_size == 0:
                    print("Current lr : {}".format(self.optimizer.state_dict()['param_groups'][0]['lr']))
                self.scheduler.step()

            if val_dataloader:
                auc = self.evaluate(self.model, val_dataloader)
                print('epoch:', epoch_i, 'validation: auc:', auc)
                if self.early_stopper.stop_training(auc, self.model.state_dict()):
                    print(f'validation: best auc: {self.early_stopper.best_auc}')
                    self.model.load_state_dict(self.early_stopper.best_weights)
                    break
        torch.save(self.model.state_dict(), os.path.join(self.model_path,
                                                            "model.pth"))

    def predict(self, model, data_loader):
        model.eval()
        predicts = list()
        y_true = list()
        with torch.no_grad():
            tk0 = tqdm.tqdm(data_loader, desc="predict", smoothing=0, mininterval=1.0)
            for i, x in enumerate(tk0):
                if isinstance(x, list):
                    x_dict = x[0]
                    y = x[1]
                else:
                    x_dict = x
                    y = x['label'].numpy()
                x_dict = {k: v.to(self.devic) for k, v in x_dict.items()}
                y_true.extend(y)
                y_pred = model(x_dict)
                predicts.extend(y_pred.tolist())
        return np.array(predicts), np.array(y_true)
