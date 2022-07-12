import torch
from torch import nn
import torch.nn.functional as F
from transformers import ElectraForMaskedLM, ElectraForPreTraining


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -torch.log(-torch.log(noise))


def gumbel_sample(logits, temperature = 1.):
    return ((logits / temperature) + gumbel_noise(logits)).argmax(dim=-1)


class ELECTRAModel(nn.Module):
    def __init__(self,
                 generator: ElectraForMaskedLM,
                 discriminator: ElectraForPreTraining):
        super().__init__()
        self.generator, self.discriminator = generator, discriminator
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(0., 1.)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        a_tensor = next(self.parameters())
        device, dtype = a_tensor.device, a_tensor.dtype
        dtype = torch.float32
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(
            torch.tensor(0., device=device, dtype=dtype), torch.tensor(1., device=device, dtype=dtype))

    def forward(self, input_ids, token_type_ids, attention_mask, masked_lm_labels):
        is_mlm_applied = masked_lm_labels.ne(-1)
        gen_logits = self.generator(input_ids, attention_mask, token_type_ids)[0]  # (B, L, vocab_size)
        mlm_gen_logits = gen_logits[is_mlm_applied, :]  # (#mlm_positions, vocab_size)

        with torch.no_grad():
            # sampling
            pred_toks = self.sample(mlm_gen_logits)  # (#mlm_positions, )
            # produce inputs for discriminator
            generated = input_ids.clone()  # (B, L)
            generated[is_mlm_applied] = pred_toks  # (B, L)
            # produce labels for discriminator
            is_replaced = is_mlm_applied.clone()  # (B, L)
            is_replaced[is_mlm_applied] = (pred_toks != masked_lm_labels[is_mlm_applied])  # (B, L)

        disc_logits = self.discriminator(generated, attention_mask, token_type_ids)[0]  # (B, L)

        return gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied

    def sample(self, logits, method: str='fp32_gumbel'):
        if method == 'fp32_gumbel':
            return gumbel_sample(logits)  # faster than sampling from the distribution
            #gumbel = self.gumbel_dist.sample(logits.shape).to(logits.device)
            #return (logits.float() + gumbel).argmax(dim=-1)
        elif method == 'fp16_gumbel':  # 5.06 ms
            gumbel = self.gumbel_dist.sample(logits.shape).to(logits.device)
            return (logits + gumbel).argmax(dim=-1)
        elif method == 'multinomial':  # 2.X ms
            return torch.multinomial(F.softmax(logits, dim=-1), 1).squeeze()


class ELECTRALoss(object):
    def __init__(self, loss_weights=(1.0, 50.0)):
        self.gen_weight, self.dis_weight = loss_weights
        self.gen_loss_fc = nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        self.disc_loss_fc = nn.BCEWithLogitsLoss(reduction='mean')

    def __call__(self, prediction, target):
        gen_logits, generated, disc_logits, is_replaced, attention_mask, is_mlm_applied = prediction
        gen_loss = self.gen_loss_fc(gen_logits[is_mlm_applied, :], target[is_mlm_applied])
        disc_logits = disc_logits.masked_select(attention_mask.bool())
        is_replaced = is_replaced.masked_select(attention_mask.bool())
        disc_loss = self.disc_loss_fc(disc_logits.float(), is_replaced.float())
        return gen_loss * self.gen_weight + disc_loss * self.dis_weight
