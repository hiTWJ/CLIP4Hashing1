from torch import nn
import torch


class HashNet(nn.Module):
    def __init__(self, bit_size):
        super(HashNet, self).__init__()
        self.hash = nn.Sequential(nn.Linear(512, 4096),
                                  nn.ReLU(),
                                  nn.Linear(4096, 4096),
                                  nn.ReLU(),
                                  )
        # use ReLU in MLP, then use min-max hashing layer to get hashcode
        self.fc_encode = nn.Linear(4096, bit_size)

    class Hash(torch.autograd.Function):
        @staticmethod
        def forward(ctx, U):
            temp_U = U.t()
            max_value, max_index = torch.max(temp_U, dim=1)
            min_value, min_index = torch.min(temp_U, dim=1)
            maxmin_values = torch.stack([max_value, min_value], dim=1).unsqueeze(2)
            temp_U = temp_U.unsqueeze(2)
            dist = torch.cdist(temp_U, maxmin_values, p=1).to('cuda')
            differences = dist[:, :, 0] - dist[:, :, 1]
            B = torch.zeros(differences.shape).to('cuda')
            B[differences < 0] = 1
            B[differences >= 0] = -1
            B = B.t().to('cuda')
            ctx.save_for_backward(U, B)
            return B

        @staticmethod
        def backward(ctx, g):
            U, B = ctx.saved_tensors
            add_g = (U - B) / (B.numel())
            grad = g + args.gamma * add_g
            return grad

    def forward(self, x):
        feature = self.hash(x)
        hid = self.fc_encode(feature)  # Hz z = bit_size
        code = HashNet.Hash.apply(hid)
        return hid, code


def get_model(bit_size):
    return HashNet(bit_size)
