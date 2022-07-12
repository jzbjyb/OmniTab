import torch
from torch import nn
import numpy as np


class CLIPLoss(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, is_paired: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_paired = is_paired
        self.source_projection = nn.Parameter(torch.empty(input_dim, output_dim), requires_grad=True)
        self.target_projection = nn.Parameter(torch.empty(input_dim, output_dim), requires_grad=True)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07), requires_grad=True)
        self.initialize_parameters()

    @staticmethod
    def get_diag_label(repr, binary: bool = False):  # (B, emb_size)
        l = np.sqrt(repr.size(0))
        assert l == int(l), 'batch size is not valid'
        l = int(l)
        if binary:
            label = torch.diag(torch.ones(l)).long().to(repr.device).view(-1)
        else:
            label = torch.arange(l).to(repr.device)
        return l, label

    def initialize_parameters(self):
        nn.init.normal_(self.source_projection, std=self.input_dim ** -0.5)
        nn.init.normal_(self.target_projection, std=self.input_dim ** -0.5)

    def forward(self, *args, **kwargs):
        if self.is_paired:
            return self.forward_paired(*args, **kwargs)
        else:
            return self.forward_normal(*args, **kwargs)

    def preprocess(self, tensor, projection):
        tensor = tensor @ projection
        tensor = tensor / tensor.norm(dim=-1, keepdim=True)
        return tensor

    def forward_normal(self,
                       source,  # (B, emb_size)
                       target,  # (B, emb_size)
                       return_repr: bool = False):
        # projection and normalize
        source = self.preprocess(source, self.source_projection)
        target = self.preprocess(target, self.target_projection)
        if return_repr:
            return source, target

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_source = logit_scale * source @ target.t()  # (B, B)
        logits_per_target = logit_scale * target @ source.t()  # (B, B)

        # cross entropy loss
        bs = source.size(0)
        label = torch.arange(bs).to(source.device)
        loss_fct = nn.CrossEntropyLoss(reduction='mean')
        pre_source_loss = loss_fct(logits_per_source, label)
        pre_target_loss = loss_fct(logits_per_target, label)
        avg_loss = (pre_source_loss + pre_target_loss) / 2
        return avg_loss

    def forward_paired(self,
                       source,  # (B, emb_size)
                       target,  # (B, emb_size)
                       labels=None,  # (B, )
                       return_repr: bool = False):
        # projection and normalize
        source = self.preprocess(source, self.source_projection)
        target = self.preprocess(target, self.target_projection)
        if return_repr:
            return source, target

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * (source * target).sum(-1)  # (B, )

        if labels is None:  # cross entropy loss
            l, label = CLIPLoss.get_diag_label(source)
            logits = logits.view((l, l))
            loss_fct = nn.CrossEntropyLoss(reduction='mean')
            pre_source_loss = loss_fct(logits, label)
            pre_target_loss = loss_fct(logits.t(), label)
            avg_loss = (pre_source_loss + pre_target_loss) / 2
        else:  # binary loss
            loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
            avg_loss = loss_fct(logits.view(-1), labels.view(-1).float())
        return avg_loss
