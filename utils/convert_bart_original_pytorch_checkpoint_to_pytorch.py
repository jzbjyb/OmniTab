# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Convert BART checkpoint."""


import argparse
import os
from pathlib import Path

import fairseq
import torch
from packaging import version
from torch import nn

from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartForSequenceClassification,
    BartModel,
    BartTokenizer,
)
from transformers.utils import logging


FAIRSEQ_MODELS = ["bart.large", "bart.large.mnli", "bart.large.cnn", "bart_xsum/model.pt"]
extra_arch = {"bart.large": BartModel, "bart.large.mnli": BartForSequenceClassification}
if version.parse(fairseq.__version__) < version.parse("0.9.0"):
    raise Exception("requires fairseq >= 0.9.0")


logging.set_verbosity_info()
logger = logging.get_logger(__name__)

SAMPLE_TEXT = " Hello world! cécé herlolip"

mnli_rename_keys = [
    ("model.classification_heads.mnli.dense.weight", "classification_head.dense.weight"),
    ("model.classification_heads.mnli.dense.bias", "classification_head.dense.bias"),
    ("model.classification_heads.mnli.out_proj.weight", "classification_head.out_proj.weight"),
    ("model.classification_heads.mnli.out_proj.bias", "classification_head.out_proj.bias"),
]


def remove_ignore_keys_(state_dict):
    ignore_keys = [
        "encoder.version",
        "decoder.version",
        "model.encoder.version",
        "model.decoder.version",
        "_float_tensor",
    ]
    for k in ignore_keys:
        state_dict.pop(k, None)


def rename_key(dct, old, new):
    val = dct.pop(old)
    dct[new] = val


def load_xsum_checkpoint_remote(checkpoint_path):
    """Checkpoint path should end in model.pt"""
    sd = torch.load(checkpoint_path, map_location="cpu")
    hub_interface = torch.hub.load("pytorch/fairseq", "bart.large.cnn").eval()
    hub_interface.model.load_state_dict(sd["model"])
    return hub_interface


def load_xsum_checkpoint(tapex_large_ckpt, bart_large_dir: str = 'data/runs/bart.large.fairseq'):
    from fairseq.models.bart import BARTModel
    tapex_sd = torch.load(tapex_large_ckpt, map_location='cpu')
    bart_sd = torch.load(os.path.join(bart_large_dir, 'model.pt'), map_location='cpu')
    mask_token_emb = bart_sd['model']['encoder.embed_tokens.weight'][-1]
    print('mask token vector', mask_token_emb)
    # tapex didn't use mask token
    for involved_key in ['encoder.embed_tokens.weight',
                         'decoder.embed_tokens.weight',
                         'decoder.output_projection.weight']:
        tapex_sd['model'][involved_key][-1, :] = mask_token_emb
    bart = BARTModel.from_pretrained(model_name_or_path=bart_large_dir, checkpoint_file='model.pt').eval()
    bart.model.load_state_dict(tapex_sd['model'])
    return bart


def make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


@torch.no_grad()
def convert_bart_checkpoint(checkpoint_path, pytorch_dump_folder_path, hf_checkpoint_name=None, bart_dir=None):
    """
    Copy/paste/tweak model's weights to our BERT structure.
    """
    if not os.path.exists(checkpoint_path):
        bart = torch.hub.load("pytorch/fairseq", checkpoint_path).eval()
    else:
        bart = load_xsum_checkpoint(checkpoint_path, bart_dir)

    bart.model.upgrade_state_dict(bart.model.state_dict())
    if hf_checkpoint_name is None:
        hf_checkpoint_name = checkpoint_path.replace(".", "-")
    config = BartConfig.from_pretrained(hf_checkpoint_name)
    tokens = bart.encode(SAMPLE_TEXT).unsqueeze(0)
    tokens2 = BartTokenizer.from_pretrained(hf_checkpoint_name).encode(SAMPLE_TEXT, return_tensors="pt").unsqueeze(0)
    assert torch.eq(tokens, tokens2).all()

    if checkpoint_path == "bart.large.mnli":
        state_dict = bart.state_dict()
        remove_ignore_keys_(state_dict)
        state_dict["model.shared.weight"] = state_dict["model.decoder.embed_tokens.weight"]
        for src, dest in mnli_rename_keys:
            rename_key(state_dict, src, dest)
        model = BartForSequenceClassification(config).eval()
        model.load_state_dict(state_dict)
        fairseq_output = bart.predict("mnli", tokens, return_logits=True)
        new_model_outputs = model(tokens)[0]  # logits
    else:  # no classification heads to worry about
        state_dict = bart.model.state_dict()
        remove_ignore_keys_(state_dict)
        state_dict["shared.weight"] = state_dict["decoder.embed_tokens.weight"]
        fairseq_output = bart.extract_features(tokens)
        '''
        if hf_checkpoint_name == "facebook/bart-large":
            model = BartModel(config).eval()
            model.load_state_dict(state_dict)
            new_model_outputs = model(tokens).model[0]
        '''
        if True:  # always use word embeding
            model = BartForConditionalGeneration(config).eval()  # an existing summarization ckpt
            if 'decoder.output_projection.weight' in state_dict:  # remove output layer
                state_dict.pop('decoder.output_projection.weight', None)

            # bart-base vocab from fairseq is larger because of dummy tokens (https://github.com/pytorch/fairseq/issues/2242) remove them
            # the old versions of transformers are also affected by this: https://github.com/huggingface/transformers/issues/9731
            for key in ['encoder.embed_tokens.weight', 'decoder.embed_tokens.weight', 'shared.weight']:
                state_dict[key] = torch.cat([state_dict[key][:50264], state_dict[key][-1:]], 0)

            model.model.load_state_dict(state_dict)
            if hasattr(model, "lm_head"):
                model.lm_head = make_linear_from_emb(model.model.shared)
            new_model_outputs = model.model(tokens)[0]

    # Check results
    assert fairseq_output.shape == new_model_outputs.shape
    assert (fairseq_output == new_model_outputs).all().item()
    Path(pytorch_dump_folder_path).mkdir(exist_ok=True)
    save_function = lambda obj, f: torch.save(obj, f, _use_new_zipfile_serialization=False)
    model.save_pretrained(pytorch_dump_folder_path, save_function=save_function)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "fairseq_path", type=str, help="bart.large, bart.large.cnn or a path to a model.pt on local filesystem."
    )
    parser.add_argument("pytorch_dump_folder_path", default=None, type=str, help="Path to the output PyTorch model.")
    parser.add_argument(
        "--hf_config", default=None, type=str, help="Which huggingface architecture to use: bart-large-xsum"
    )
    parser.add_argument('--bart_dir', default=None, type=str, help='directory to the fairseq bart model')
    args = parser.parse_args()
    convert_bart_checkpoint(args.fairseq_path, args.pytorch_dump_folder_path,
                            hf_checkpoint_name=args.hf_config, bart_dir=args.bart_dir)
