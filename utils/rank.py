from argparse import ArgumentParser
from pathlib import Path
import contextlib
import os
import sys
import random
from tqdm import tqdm
from functools import partial
import torch
import torch.nn as nn
import torch.distributed
from torch.utils.data import DataLoader, DistributedSampler
import numpy as np
from fairseq import optim
from fairseq.optim import lr_scheduler
from fairseq.optim.adam import FairseqAdam
from fairseq.optim.lr_scheduler.polynomial_decay_schedule import PolynomialDecaySchedule
from fairseq.options import eval_str_list
from omnitab import TableBertModel
from omnitab.utils import compute_mrr
from omnitab.ranker import RankDataset, Ranker
from utils.comm import init_distributed_mode, init_signal_handler


def build_optimizer(model, args):
    params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = optim.build_optimizer(args, params)
    scheduler = lr_scheduler.build_lr_scheduler(args, optimizer)
    scheduler.step_update(0)
    return optimizer, scheduler


def main():
    parser = ArgumentParser()
    parser.add_argument('--sample_file', type=Path, required=True)
    parser.add_argument('--db_file', type=Path, required=True)
    parser.add_argument('--output_file', type=Path, required=True)
    parser.add_argument('--model_path', type=Path, default=True)
    parser.add_argument('--load_from', type=Path, default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--group_by', type=int, default=10)
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--cpu', action='store_true', help="Whether not to use CUDA when available")

    # distributed training
    parser.add_argument('--ddp-backend', type=str, default='pytorch', choices=['pytorch', 'apex'])
    parser.add_argument('--local_rank', '--local-rank', type=int, default=-1,
                        help='local_rank for distributed training on gpus')
    parser.add_argument('--master-port', type=int, default=-1,
                        help='Master port (for multi-node SLURM jobs)')
    parser.add_argument('--debug-slurm', action='store_true', help='Debug multi-GPU / multi-node within a SLURM job')

    # optimizer
    parser.add_argument('--max-epoch', type=int, default=10)
    parser.add_argument("--lr-scheduler", type=str, default='polynomial_decay', help='Learning rate scheduler')
    parser.add_argument("--optimizer", type=str, default='adam', help='Optimizer to use')
    parser.add_argument('--clip-norm', default=0., type=float, help='clip gradient')
    parser.add_argument('--lr', '--learning-rate', default='0.00005', type=eval_str_list,
                        metavar='LR_1,LR_2,...,LR_N',
                        help='learning rate for the first N epochs; all epochs >N using LR_N'
                             ' (note: this may be interpreted differently depending on --lr-scheduler)')
    FairseqAdam.add_args(parser)
    PolynomialDecaySchedule.add_args(parser)

    args = parser.parse_args()

    # init distributed
    init_distributed_mode(args)
    init_signal_handler()
    device = torch.device(f'cuda:{torch.cuda.current_device()}')
    assert not args.multi_gpu or args.finetune, 'multi gpu only supported with --finetune'
    assert torch.cuda.is_available(), 'no GPU detected'

    print('device: {} gpu_id: {}, distributed training: {}'.format(device, args.local_rank, bool(args.multi_gpu)))

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # build model
    if args.multi_gpu and args.local_rank != 0:
        torch.distributed.barrier()
    model = Ranker(TableBertModel.from_pretrained(str(args.model_path)))
    if args.multi_gpu and args.local_rank == 0:
        torch.distributed.barrier()
    if args.load_from:
        print('load from {}'.format(args.load_from))
        state_dict = torch.load(args.load_from, map_location='cpu')
        model.load_state_dict(state_dict, strict=True)

    model = model.to(device)
    if args.multi_gpu:
        if args.ddp_backend == 'pytorch':
            model = nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.local_rank], output_device=args.local_rank,
                broadcast_buffers=False, find_unused_parameters=True)
        else:
            import apex
            model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)
        model_ptr = model.module
    else:
        model_ptr = model
    model.train() if args.finetune else model.eval()

    # build dataset
    rankdataset = RankDataset(args.sample_file, args.db_file, group_by=args.group_by, tokenizer=model_ptr.tokenizer)
    sampler = DistributedSampler(rankdataset, num_replicas=args.world_size, rank=args.global_rank) if args.multi_gpu else None
    dataloader = DataLoader(rankdataset, sampler=sampler, batch_size=args.batch_size, num_workers=0,
                            collate_fn=partial(RankDataset.collate, model=model_ptr.model))

    # build optimizer
    args.total_num_updates = len(rankdataset) * args.max_epoch // args.batch_size // args.world_size
    args.warmup_updates = int(args.total_num_updates * 0.1)
    print('#samples in the data {}, total_num_updates {}, warmup_updates {}'.format(
        len(rankdataset), args.total_num_updates, args.warmup_updates))
    optimizer, scheduler = build_optimizer(model, args)

    num_updates = 0
    with contextlib.ExitStack() if args.finetune else torch.no_grad():
        for e in range(args.max_epoch if args.finetune else 1):
            all_scores = []
            all_labels = []
            with tqdm(total=len(dataloader), desc='Epoch {}'.format(e),
                      file=sys.stdout, miniters=10, disable=not args.is_master) as pbar:
                for tensor_dict, labels in dataloader:
                    loss, scores = model(tensor_dict, labels=labels)
                    all_scores.extend(scores.detach().cpu().numpy().tolist())
                    all_labels.extend(labels.cpu().numpy().tolist())
                    if args.finetune:
                        optimizer.zero_grad()
                        optimizer.backward(loss)
                        optimizer.clip_grad_norm(args.clip_norm)
                        optimizer.step()
                        num_updates += 1
                        scheduler.step_update(num_updates)
                    pbar.update(1)
                    pbar.set_postfix_str('loss: {}'.format(loss.item()))
            if args.finetune and args.output_file and args.is_master:  # save model (only by master)
                print('save model for epoch {}'.format(e + 1))
                os.makedirs(args.output_file, exist_ok=True)
                torch.save(model_ptr.state_dict(), str(args.output_file / 'model_epoch{}.bin'.format(e + 1)))

    if not args.finetune:
        mrrs = []
        with open(args.output_file, 'w') as fout:
            for i in range(0, len(all_labels), args.group_by):
                ls = all_labels[i:i + args.group_by]
                ss = all_scores[i:i + args.group_by]
                mrrs.append(compute_mrr(ss, ls))
                for s in ss:
                    fout.write('{}\n'.format(s))
        print('MRR {}'.format(np.mean(mrrs)))


if __name__ == '__main__':
    main()
