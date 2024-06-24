import torch
import model.HashNet as HashNet
from train.MSRVTT import MSRVTT_train_dataset, MSRVTT_val_dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm


class CLIP4H:
    def __init__(self):
        self.epoch = 200
        self.batch_size = 3
        self.lr = 0.01
        self.decrease_epoch = 150
        self.lr_decrease_factor = 0.1
        self.bit_size = 256

        self.model = HashNet.get_model(self.bit_size)

        self.weight_decay = 0.0005
        self.momentum = 0.9
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum,
                                         weight_decay=self.weight_decay)

    def train(self):
        train_set = MSRVTT_train_dataset()
        train_loader = DataLoader(train_set, batch_size=self.batch_size)

        for epoch in range(self.epoch):
            for image_feature, text_feature in train_loader:  # Fv and Ft

                F_I = image_feature.to('cuda')
                F_T = text_feature.to('cuda')

                # Construct cross-modal affinity matrix
                F_I = F.normalize(F_I)
                F_T = F.normalize(F_T)

                # tensor.mm(tensor): matrix multiple
                # tensor.t(): matrix transposition
                S_IT = F_I.mm(F_T.t())
                S_TI = F_T.mm(F_I.t())

                # Set diagonal elements to 1
                # torch.diag_embed: get diagonal elements from tensor
                complete_S_IT_diagonal = torch.diag_embed(1 - S_IT.diagonal())
                complete_S_TI_diagonal = torch.diag_embed(1 - S_TI.diagonal())
                S_IT = S_IT + complete_S_IT_diagonal
                S_TI = S_TI + complete_S_TI_diagonal

                # CLIP_base
                S_C = 0.5 * S_TI + 0.5 * S_IT

                # dynamic weighting
                S = dynamic_weighting(S_C)

                # HashNet
                # H m*z m = batch_size z = bit_size
                hid_I, code_I = self.model(F_I)
                hid_T, code_T = self.model(F_T)

                H_I = F.normalize(hid_I)
                H_T = F.normalize(hid_T)

                # m*m
                HI_HI = H_I.mm(H_I.t())
                HT_HT = H_T.mm(H_T.t())
                HI_HT = H_I.mm(H_T.t())
                HT_HI = H_T.mm(H_I.t())

                # ||A||²F = ∑i∑j aij²
                # mse_loss = ∑i∑j aij² / batch_size (m)
                # each loss / m, so no difference
                intra_loss = F.mse_loss(HI_HI, S) + F.mse_loss(HT_HT, S)
                inter_loss = F.mse_loss(HI_HT, S) + F.mse_loss(HT_HI, S)
                consistency_loss = F.mse_loss(H_I, H_T)

                loss = 0.1 * intra_loss + 1 * inter_loss + 2 * consistency_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


def dynamic_weighting(S):
    # mean min max values of Sc in each batch
    S_mean = torch.mean(S)
    S_min = torch.min(S)
    S_max = torch.max(S)

    # torch.exp(x) = e^x
    # S[S < S_mean] includes all elements that < S_mean
    # = Si,j
    W_low = torch.exp(-0.5 * (S_mean - S[S <= S_mean]) / (S_mean - S_min) - 0.5)
    S[S <= S_mean] = W_low * S[S <= S_mean]

    W_high = torch.exp(0.5 * (S[S > S_mean] - S_mean) / (S_max - S_mean) - 0.5)
    S[S > S_mean] = W_high * S[S > S_mean]

    S[S > 1.0] = 1.0
    S[S < -1.0] = -1.0
    return S


def train():
    trainer = CLIP4H()
    trainer.train()