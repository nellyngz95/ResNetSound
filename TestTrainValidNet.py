import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from ProcessedData import Net
from torch.utils.data import DataLoader
import torch.functional as F


class MyLearner(pl.LightningModule):

    def __init__(self, model, learning_rate=3e-4):

        super().__init__()
        self.learning_rate = learning_rate
        self.model = model

    def forward(self, x):
        x = self.model(x)
        x = F.log_softmax(x, dim=1)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        return loss

    def validation_step(self, batch, batch_idx, split='val'):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log(f'{split}_loss', loss, prog_bar=True)
        self.log(f'{split}_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, split='test')

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def train_dataloader(self):
        return train_loader

    def val_dataloader(self):
        return valid_loader

    def test_dataloader(self):
        return test_loader
    
#change to cuda and 1 gpu in the next function
    device = torch.device('cpu')

def eval_acc(model, device, dataloader, debug_name=None):
    model = model.to(device).eval()
    count = correct = 0
    for X, gt in dataloader:
        logits = model(X.to(device))
        preds = torch.argmax(logits, dim=1)
        correct += sum(preds.cpu() == gt)
        count += len(gt)
    acc = correct/count
    if debug_name:
        print(f'{debug_name} acc = {acc:.4f}')
    return acc

learner = MyLearner(Net(len(train_dataset)))
checkpoint = pl.callbacks.ModelCheckpoint(monitor='val_acc')
trainer = pl.Trainer('cpu', max_epochs=100, callbacks=[checkpoint])
trainer.fit(learner);

learner.load_state_dict(torch.load(checkpoint.best_model_path)['state_dict'])

eval_acc(learner.model, device, learner.val_dataloader(), 'val')
eval_acc(learner.model, device, learner.test_dataloader(), 'test');