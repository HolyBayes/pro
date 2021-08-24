import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import itertools
import torch.nn.functional as F
from tqdm import trange

class PairDataset(Dataset):
    def __init__(self, X1, X2, c, max_size=None):
        self.X1 = X1
        self.X2 = X2
        self.c = c
        self.max_size = max_size
    
    def __len__(self):
        if self.max_size is None:
            return self.X1.shape[0] * self.X2.shape[0]
        else:
            return self.max_size
    
    def __getitem__(self, idx):
        if self.max_size is None:
            idx1, idx2 = idx % self.X1.shape[0], idx // self.X1.shape[0]
            return self.X1[idx1], self.X2[idx2], self.c
        else:
            # pick random pair
            x1 = self.X1[np.random.choice(np.arange(self.X1.shape[0]))]
            x2 = self.X2[np.random.choice(np.arange(self.X2.shape[0]))]
            return x1, x2, self.c
        

class PROTrainDataset(Dataset):
    def __init__(self, X_pos, X_neg, max_size=None, c1=0, c2=4, c3=8):
        self.max_size = max_size
        self.datasets = [PairDataset(X_pos, X_pos, c1, max_size), 
                         PairDataset(X_pos, X_neg, c2, max_size), PairDataset(X_neg, X_pos, c2, max_size), 
                         PairDataset(X_neg, X_neg, c3, max_size)]
        
    def __len__(self):
        return sum([len(x) for x in self.datasets])
        
    def __getitem__(self, idx):
        if self.max_size is None:
            # pick random pair
            dataset_idx = np.random.choice([0,1,2,3])
            return self.datasets[dataset_idx][idx]
        else:
            cumsizes = np.cumsum([len(x) for x in self.datasets])
            for i, size in enumerate(cumsizes):
                if idx < size:
                    if i == 0:
                        return self.datasets[i][idx]
                    return self.datasets[i][idx-cumsizes[i-1]]
        
class PROTestDataset(Dataset):
    def __init__(self, X_test, X_pos, X_neg):
        self.X_pos = X_pos
        self.X_neg = X_neg
        self.X_test = X_test
    
    def __len__(self):
        return self.X_test.shape[0]
    
    def __getitem__(self, idx):
        return self.X_test[idx], self.X_pos[np.random.choice(np.arange(self.X_pos.shape[0]))], self.X_neg[np.random.choice(np.arange(self.X_neg.shape[0]))]
        



class PRO(nn.Module):
    def __init__(self, X_pos, X_neg, feature_encoder, anomaly_detector, c1=0, c2=4, c3=8, l2=1e-2, batch_size=512, batches_per_epoch=20):
        super().__init__()
        self.feature_encoder = feature_encoder
        self.anomaly_detector = anomaly_detector
        
        self.X_pos = X_pos
        self.X_neg = X_neg
        max_size = batch_size * batches_per_epoch // 4
        self.batch_size, self.max_size, self.c1, self.c2, self.c3 = batch_size, max_size, c1, c2, c3
        self.opt = optim.RMSprop(itertools.chain(*[self.feature_encoder.parameters(), 
                                                  self.anomaly_detector.parameters()
                                                 ]), weight_decay=l2)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    def forward(self, X1, X2):
        Z1, Z2 = self.feature_encoder(X1), self.feature_encoder(X2)
        Z = torch.cat([Z1, Z2], -1)
        return self.anomaly_detector(Z)
        
    def fit(self, epoches=50):
        train_dataset = PROTrainDataset(self.X_pos, self.X_neg, self.max_size, self.c1, self.c2, self.c3)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        for epoch in trange(epoches):
            for (X1, X2, c) in train_loader:
                self.opt.zero_grad()
                X1, X2, c = [x.to(self.device).float() for x in [X1, X2, c]]
                pred = self.forward(X1, X2)
                loss = F.mse_loss(pred.squeeze(), c.squeeze())
                loss.backward()
                self.opt.step()
                
        
    def predict(self, X_test, E=30, batch_size=512):
        test_dataset = PROTestDataset(X_test, self.X_pos, self.X_neg)
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        score = np.zeros(X_test.shape[0])
        for _ in trange(E):
            preds = []
            for (batch_X_test, batch_X_pos, batch_X_neg) in test_loader:
                batch_X_test, batch_X_pos, batch_X_neg = [x.to(self.device).float() for x in [batch_X_test, batch_X_pos, batch_X_neg]]
                with torch.no_grad():
                    preds_ = 0.5*(self.forward(batch_X_neg, batch_X_test) + self.forward(batch_X_test, batch_X_pos)).detach().cpu().numpy().squeeze(1)
                preds.append(preds_)
            preds = np.concatenate(preds)
            score += preds
        return score/E
