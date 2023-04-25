import time
import os
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from model import device
from masking import create_mask
from model import MusicTransformer,hparams

datapath = "C:/Users/z7913/Documents/project_real/preprocessed_data.pt"
ckpt_path = "C:/Users/z7913/Documents/project_real/checkpoint.pt"
save_path = "C:/Users/z7913/Documents/project_real/final_model.pt"

def transformer_lr_schedule(d_model, step_num, warmup_steps=4000):
    if warmup_steps <= 0:
        step_num += 4000
        warmup_steps = 4000
    step_num = step_num + 1e-6  # avoid division by 0
    if type(step_num) == torch.Tensor:
        arg = torch.min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))
    else:
        arg = min(step_num ** -0.5, step_num * (warmup_steps ** -1.5))
    return (d_model ** -0.5) * arg
def loss_fn(prediction, target, criterion=F.cross_entropy):
    mask = torch.ne(target, torch.zeros_like(target))           # ones where target is 0
    _loss = criterion(prediction, target, reduction='none')     # loss before masking
    # multiply mask to loss elementwise to zero out pad positions
    mask = mask.to(_loss.dtype)
    _loss *= mask
    # output is average over the number of values that were not masked
    return torch.sum(_loss) / torch.sum(mask)
def train_step(model: MusicTransformer, opt, sched, inp, tar):
    # forward pass
    predictions = model(inp, mask=create_mask(inp, n=inp.dim() + 2))
    # backward pass
    opt.zero_grad()
    loss = loss_fn(predictions.transpose(-1, -2), tar)
    loss.backward()
    opt.step()
    sched.step()
    return float(loss)
def val_step(model: MusicTransformer, inp, tar):
    predictions = model(inp, mask=create_mask(inp, n=max(inp.dim() + 2, 2)))
    loss = loss_fn(predictions.transpose(-1, -2), tar)
    return float(loss)
class MusicTransformerTrainer:

    def __init__(self, hparams_, datapath, batch_size, warmup_steps=4000,
                 ckpt_path="music_transformer_ckpt.pt", load_from_checkpoint=False):
        # get the data
        self.datapath = datapath
        self.batch_size = batch_size
        input_data, target_data = torch.load(datapath)
        data = input_data.long().to(device)  # 或 target_data，具體取決於您想要使用的張量


        # max absolute position must be able to acount for the largest sequence in the data
        if hparams_["max_abs_position"] > 0:
            hparams_["max_abs_position"] = max(hparams_["max_abs_position"], data.shape[-1])

        # train / validation split: 80 / 20
        train_len = round(data.shape[0] * 0.8)
        train_data = data[:train_len]
        val_data = data[train_len:]
        print(f"There are {data.shape[0]} samples in the data, {len(train_data)} training samples and {len(val_data)} "
              "validation samples")
        # datasets and dataloaders: split data into first (n-1) and last (n-1) tokens
        self.train_ds = TensorDataset(train_data[:, :-1], train_data[:, 1:])
        self.train_dl = DataLoader(dataset=self.train_ds, batch_size=batch_size, shuffle=True)
        self.val_ds = TensorDataset(val_data[:, :-1], val_data[:, 1:])
        self.val_dl = DataLoader(dataset=self.val_ds, batch_size=batch_size, shuffle=True)
        # create model
        self.model = MusicTransformer(**hparams_).to(device)
        self.hparams = hparams_
        # setup training
        self.warmup_steps = warmup_steps
        self.optimizer = optim.Adam(self.model.parameters(), lr=1.0, betas=(0.9, 0.98))
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda x: transformer_lr_schedule(self.hparams['d_model'], x, self.warmup_steps)
        )
        # setup checkpointing / saving
        self.ckpt_path = ckpt_path
        self.train_losses = []
        self.val_losses = []

        # load checkpoint if necessesary
        if load_from_checkpoint and os.path.isfile(self.ckpt_path):
            self.load()
    def save(self, ckpt_path=None):
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "train_losses": self.train_losses,
            "validation_losses": self.val_losses,
            "warmup_steps": self.warmup_steps,
            "hparams": self.hparams
        }
        torch.save(ckpt, self.ckpt_path)
        return
    def load(self, ckpt_path=None):
        if ckpt_path is not None:
            self.ckpt_path = ckpt_path
        ckpt = torch.load(self.ckpt_path)
        del self.model, self.optimizer, self.scheduler
        # create and load model
        self.model = MusicTransformer(**ckpt["hparams"]).to(device)
        self.hparams = ckpt["hparams"]
        print("Loading the model...", end="")
        print(self.model.load_state_dict(ckpt["model_state_dict"]))
        # create and load load optimizer and scheduler
        self.warmup_steps = ckpt["warmup_steps"]
        self.optimizer = optim.Adam(self.model.parameters(), lr=1.0, betas=(0.9, 0.98))
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler = optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lambda x: transformer_lr_schedule(self.hparams['d_model'], x, self.warmup_steps)
        )
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        # load loss histories
        self.train_losses = ckpt["train_losses"]
        self.val_losses = ckpt["validation_losses"]
        return
    def fit(self, epochs):
        print_interval = epochs // 10 + int(epochs < 10)
        train_losses = []
        val_losses = []
        start = time.time()
        print("Beginning training...")
        try:
            for epoch in range(epochs):
                train_epoch_losses = []
                val_epoch_losses = []
                self.model.train()
                for train_inp, train_tar in self.train_dl:
                    loss = train_step(self.model, self.optimizer, self.scheduler, train_inp, train_tar)
                    train_epoch_losses.append(loss)
                self.model.eval()
                for val_inp, val_tar in self.val_dl:
                    loss = val_step(self.model, val_inp, val_tar)
                    val_epoch_losses.append(loss)
                # mean losses for the epoch
                train_mean = sum(train_epoch_losses) / len(train_epoch_losses)
                val_mean = sum(val_epoch_losses) / len(val_epoch_losses)
                # store complete history of losses in member lists and relative history for this session in output lists
                self.train_losses.append(train_mean)
                train_losses.append(train_mean)
                self.val_losses.append(val_mean)
                val_losses.append(val_mean)
                if ((epoch + 1) % print_interval) == 0:
                    print(f"Epoch {epoch + 1} Time taken {round(time.time() - start, 2)} seconds "
                          f"Train Loss {train_losses[-1]} Val Loss {val_losses[-1]}")
                    # print("Checkpointing...")
                    # self.save()
                    # print("Done")
                    start = time.time()
        except KeyboardInterrupt:
            pass
        print("Checkpointing...")
        self.save()
        print("Done")
        return train_losses, val_losses
if __name__ == "__main__":
   
   # 设置训练参数
    batch_size_ = 8
    warmup_steps_ = 3000
    epochs = 5000  # 设置训练轮次

    # 设置训练器
    print("Setting up the trainer...")
    trainer = MusicTransformerTrainer(hparams, datapath, batch_size_, warmup_steps_,
                                  ckpt_path, load_from_checkpoint=False)
    print()

       # 训练模型
    trainer.fit(epochs)
    # done training, save the model
    print("Saving...")
    save_file = {
        "state_dict": trainer.model.state_dict(),
        "hparams": trainer.hparams
    }
    torch.save(save_file, save_path)
    print("Done!")