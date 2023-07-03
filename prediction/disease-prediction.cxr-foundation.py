import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
from argparse import ArgumentParser

num_classes = 14
batch_size = 150
epochs = 20
num_workers = 4
data_dir = '<PATH_TO_CXR_NUMPY_EMBEDDINGS>'


class CheXpertDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        self.labels = [
            'No Finding',
            'Enlarged Cardiomediastinum',
            'Cardiomegaly',
            'Lung Opacity',
            'Lung Lesion',
            'Edema',
            'Consolidation',
            'Pneumonia',
            'Atelectasis',
            'Pneumothorax',
            'Pleural Effusion',
            'Pleural Other',
            'Fracture',
            'Support Devices']

        self.samples = []
        for idx, _ in enumerate(tqdm(range(len(self.data)), desc='Loading Data')):
            img_path = self.data.loc[idx, 'path_preproc']
            img_label = np.zeros(len(self.labels), dtype='float32')
            for i in range(0, len(self.labels)):
                img_label[i] = np.array(self.data.loc[idx, self.labels[i].strip()] == 1, dtype='float32')

            sample = {'image_path': img_path, 'label': img_label}
            self.samples.append(sample)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        sample = self.get_sample(item)

        features = torch.from_numpy(sample['features'])
        label = torch.from_numpy(sample['label'])

        return {'features': features, 'label': label}

    def get_sample(self, item):
        sample = self.samples[item]
        img_fn = os.path.basename(sample['image_path'])
        feat_fn = os.path.join(data_dir, img_fn.replace('.jpg', '.dat'))

        features = np.fromfile(feat_fn).astype(np.float32)

        return {'features': features, 'label': sample['label']}


class CheXpertDataModule(pl.LightningDataModule):
    def __init__(self, csv_train, csv_val, csv_test, batch_size, num_workers):
        super().__init__()
        self.csv_train = csv_train
        self.csv_val = csv_val
        self.csv_test = csv_test
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_set = CheXpertDataset(self.csv_train)
        self.val_set = CheXpertDataset(self.csv_val)
        self.test_set = CheXpertDataset(self.csv_test)

        print('#train: ', len(self.train_set))
        print('#val:   ', len(self.val_set))
        print('#test:  ', len(self.test_set))

    def train_dataloader(self):
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_set, self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)


class MLP(pl.LightningModule):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        
        # CXR-MLP-5
        # self.model = nn.Sequential(
        #     nn.Linear(1376, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, 128),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(128, num_classes)
        # )

        # CXR-MLP-3
        # self.model = nn.Sequential(
        #     nn.Linear(1376, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, num_classes)
        # )

        # CXR-Linear
        self.model = nn.Sequential(
            nn.Linear(1376, num_classes)
        )

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        params_to_update = []
        for param in self.parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
        optimizer = torch.optim.Adam(params_to_update, lr=0.001)
        return optimizer

    def unpack_batch(self, batch):
        return batch['features'], batch['label']

    def process_batch(self, batch):
        feat, lab = self.unpack_batch(batch)
        out = self.forward(feat)
        prob = torch.sigmoid(out)
        loss = F.binary_cross_entropy(prob, lab)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('val_loss', loss)

    def test_step(self, batch, batch_idx):
        loss = self.process_batch(batch)
        self.log('test_loss', loss)


def test(model, data_loader, device):
    model.eval()
    logits = []
    preds = []
    targets = []

    with torch.no_grad():
        for index, batch in enumerate(tqdm(data_loader, desc='Test-loop')):
            feat, lab = batch['features'].to(device), batch['label'].to(device)
            out = model(feat)
            pred = torch.sigmoid(out)
            logits.append(out)
            preds.append(pred)
            targets.append(lab)

        logits = torch.cat(logits, dim=0)
        preds = torch.cat(preds, dim=0)
        targets = torch.cat(targets, dim=0)

        counts = []
        for i in range(0,num_classes):
            t = targets[:, i] == 1
            c = torch.sum(t)
            counts.append(c)
        print(counts)

    return preds.cpu().numpy(), targets.cpu().numpy(), logits.cpu().numpy()


def main(hparams):

    # sets seeds for numpy, torch, python.random and PYTHONHASHSEED.
    pl.seed_everything(42, workers=True)

    # data
    data = CheXpertDataModule(csv_train='../datafiles/chexpert/chexpert.sample.train.csv',
                              csv_val='../datafiles/chexpert/chexpert.sample.val.csv',
                              csv_test='../datafiles/chexpert/chexpert.resample.test.csv',
                              batch_size=batch_size,
                              num_workers=num_workers)

    # model
    model_type = MLP
    model = model_type(num_classes=num_classes)

    # Create output directory
    out_name = 'cxr-foundation-linear'
    out_dir = './' + out_name
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode='min')

    # train
    trainer = pl.Trainer(
        callbacks=[checkpoint_callback],
        log_every_n_steps = 5,
        max_epochs=epochs,
        gpus=hparams.gpus,
        logger=TensorBoardLogger('.', name=out_name),
    )
    trainer.logger._default_hp_metric = False
    trainer.fit(model, data)

    model = model_type.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_classes=num_classes)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:" + str(hparams.dev) if use_cuda else "cpu")

    model.to(device)

    cols_names_classes = ['class_' + str(i) for i in range(0,num_classes)]
    cols_names_logits = ['logit_' + str(i) for i in range(0, num_classes)]
    cols_names_targets = ['target_' + str(i) for i in range(0, num_classes)]

    print('VALIDATION')
    preds_val, targets_val, logits_val = test(model, data.val_dataloader(), device)
    df = pd.DataFrame(data=preds_val, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_val, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_val, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.val.csv'), index=False)

    print('TESTING')
    preds_test, targets_test, logits_test = test(model, data.test_dataloader(), device)
    df = pd.DataFrame(data=preds_test, columns=cols_names_classes)
    df_logits = pd.DataFrame(data=logits_test, columns=cols_names_logits)
    df_targets = pd.DataFrame(data=targets_test, columns=cols_names_targets)
    df = pd.concat([df, df_logits, df_targets], axis=1)
    df.to_csv(os.path.join(out_dir, 'predictions.test.csv'), index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--gpus', default=1)
    parser.add_argument('--dev', default=0)
    args = parser.parse_args()

    main(args)
