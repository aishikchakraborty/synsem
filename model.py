import torch
import torch.nn as nn
import torch.nn.functional as F

class Synsem(nn.Module):
    def __init__(self, ntoken, ninp, pos_classes_x, pos_classes_y, pretrained_emb):
        super(Synsem, self).__init__()
        self.org_emb = nn.Embedding(ntoken, ninp)
        self.syn_emb = nn.Embedding(ntoken, ninp//2)
        self.sem_emb = nn.Embedding(ntoken, ninp//2)
        self.W = nn.Parameter(torch.rand((ninp, ninp)))

        # POS Losses
        self.linear11 = nn.Linear(ninp//2, 100)
        self.linear12 = nn.Linear(100, pos_classes_x)
        self.relu = nn.ReLU()

        self.linear21 = nn.Linear(ninp//2, 100)
        self.linear22 = nn.Linear(100, pos_classes_y)

        self.org_emb.weight.data.copy_(pretrained_emb)
        self.org_emb.weight.requires_grad = False

    def forward(self, x, y):
        syn_emb_x = self.syn_emb(x)
        sem_emb_x = self.sem_emb(x)
        org_emb_x = self.org_emb(x)
        org_emb_reconstructed_x = torch.cat((syn_emb_x, sem_emb_x), dim=1)

        syn_emb_y = self.syn_emb(y)
        sem_emb_y = self.sem_emb(y)
        org_emb_y = self.org_emb(y)
        org_emb_reconstructed_y = torch.cat((syn_emb_y, sem_emb_y), dim=1)
        translated_y = torch.mm(org_emb_reconstructed_x, self.W)

        pos_x = self.linear12(self.relu(self.linear11(syn_emb_x)))
        pos_y = self.linear22(self.relu(self.linear21(syn_emb_y)))

        translation_loss  = F.mse_loss(translated_y, org_emb_reconstructed_y)
        reconstruction_loss = F.mse_loss(org_emb_x, org_emb_reconstructed_x) + F.mse_loss(org_emb_y, org_emb_reconstructed_y)
        semantic_loss = F.mse_loss(sem_emb_x, sem_emb_y)

        return translation_loss, reconstruction_loss, semantic_loss, pos_x, pos_y
