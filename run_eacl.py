import logging
import os
import numpy as np
import pickle as pk
import datetime
import torch.nn as nn
import torch.optim as optim
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import time
from model import GraphSmile
from eacl import EmotionAnchoredContrastiveLoss, AnchorAngleLoss, AnchorAdaptationLoss
from sklearn.metrics import confusion_matrix, classification_report
from trainer import (
    train_or_eval_eacl_stage1,
    train_or_eval_eacl_stage2,
    seed_everything,
)
from dataloader import (
    IEMOCAPDataset_BERT,
    IEMOCAPDataset_BERT4,
    MELDDataset_BERT,
    CMUMOSEIDataset7,
)
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--no_cuda', action='store_true', default=False)
parser.add_argument('--gpu', default='2', type=str, help='GPU ids')
parser.add_argument('--port', default='15302', help='MASTER_PORT')
parser.add_argument('--classify', default='emotion', help='sentiment, emotion')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR')
parser.add_argument('--l2', type=float, default=0.0001, metavar='L2')
parser.add_argument('--batch_size', type=int, default=16, metavar='BS')
parser.add_argument('--tensorboard', action='store_true', default=False)
parser.add_argument('--modals', default='avl')
parser.add_argument('--dataset', default='IEMOCAP',
                    help='MELD/IEMOCAP/IEMOCAP4/CMUMOSEI7')
parser.add_argument('--textf_mode', default='textf0',
                    help='concat4/concat2/textf0/textf1/textf2/textf3/sum2/sum4')
parser.add_argument('--conv_fpo', nargs='+', type=int, default=[3, 1, 1])
parser.add_argument('--hidden_dim', type=int, default=512)
parser.add_argument('--win', nargs='+', type=int, default=[17, 17])
parser.add_argument('--heter_n_layers', nargs='+', type=int, default=[7, 7, 7])
parser.add_argument('--drop', type=float, default=0.2, metavar='dropout')
parser.add_argument('--shift_win', type=int, default=19)
parser.add_argument('--loss_type', default='emo_sen_sft',
                    help='emo_sen_sft/emo_sen/emo_sft/emo')
parser.add_argument('--lambd', nargs='+', type=float, default=[1.0, 1.0, 0.7],
                    help='[lambda_emo, lambda_sen, lambda_sft] for multi-task in stage 1')

# EACL-specific arguments
parser.add_argument('--lambda1', type=float, default=0.9,
                    help='weight of (L_sup + lambda2*L_Ag) vs L_CE (0.9 IEMOCAP, 0.1 MELD)')
parser.add_argument('--lambda2', type=float, default=0.01,
                    help='weight of anchor angle loss within contrastive term')
parser.add_argument('--temperature', type=float, default=0.1,
                    help='temperature tau for contrastive losses')
parser.add_argument('--stage1_epochs', type=int, default=114,
                    help='epochs for representation learning stage')
parser.add_argument('--stage2_epochs', type=int, default=6,
                    help='epochs for anchor adaptation stage')
parser.add_argument('--stage2_lr', type=float, default=1e-3,
                    help='learning rate for anchor adaptation (stage 2)')

args = parser.parse_args()

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = args.port
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
world_size = torch.cuda.device_count()
os.environ['WORLD_SIZE'] = str(world_size)

MELD_path = '/mnt/Academia/Teasis/Code/Test_GraphSmile/dataset/meld_multi_features.pkl'
IEMOCAP_path = '/mnt/Academia/Teasis/Code/Test_GraphSmile/dataset/iemocap_multi_features.pkl'
IEMOCAP4_path = '/mnt/Academia/Teasis/Code/Test_GraphSmile/dataset/iemocap_multi_features_4.pkl'
CMUMOSEI7_path = ''

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def init_ddp(local_rank):
    try:
        if not dist.is_initialized():
            torch.cuda.set_device(local_rank)
            os.environ['RANK'] = str(local_rank)
            dist.init_process_group(backend='nccl', init_method='env://')
        else:
            logger.info('Distributed process group already initialized.')
    except Exception as e:
        logger.error(f'Failed to initialize distributed process group: {e}')
        raise


def get_train_valid_sampler(trainset, valid_ratio):
    size = len(trainset)
    idx = list(range(size))
    split = int(valid_ratio * size)
    return DistributedSampler(idx[split:]), DistributedSampler(idx[:split])


def get_data_loaders(path, dataset_class, batch_size, valid_ratio, num_workers, pin_memory):
    trainset = dataset_class(path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_ratio)
    train_loader = DataLoader(
        trainset, batch_size=batch_size, sampler=train_sampler,
        collate_fn=trainset.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    valid_loader = DataLoader(
        trainset, batch_size=batch_size, sampler=valid_sampler,
        collate_fn=trainset.collate_fn, num_workers=num_workers, pin_memory=pin_memory)
    testset = dataset_class(path, train=False)
    test_loader = DataLoader(
        testset, batch_size=batch_size, collate_fn=testset.collate_fn,
        num_workers=num_workers, pin_memory=pin_memory)
    return train_loader, valid_loader, test_loader


def setup_samplers(trainset, valid_ratio, epoch):
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_ratio=valid_ratio)
    train_sampler.set_epoch(epoch)
    valid_sampler.set_epoch(epoch)


def sync_anchor_grad(anchors):
    """Average anchor gradients across all DDP processes."""
    if dist.is_initialized() and anchors.grad is not None:
        dist.all_reduce(anchors.grad, op=dist.ReduceOp.SUM)
        anchors.grad.div_(dist.get_world_size())


def init_anchors_from_data(model, anchors, train_loader, cuda, n_classes):
    """Initialize anchors from per-class mean of feat_fusion (Option A from EACL paper)."""
    model.eval()
    hidden_dim = anchors.size(1)
    class_sums = torch.zeros(n_classes, hidden_dim, device=anchors.device)
    class_counts = torch.zeros(n_classes, device=anchors.device)

    with torch.no_grad():
        for data in train_loader:
            textf0, textf1, textf2, textf3, visuf, acouf, qmask, umask, label_emotion, label_sentiment = (
                [d.cuda() for d in data[:-1]] if cuda else data[:-1])

            dia_lengths = []
            label_emotions = []
            for j in range(umask.size(1)):
                dia_lengths.append((umask[:, j] == 1).nonzero().tolist()[-1][0] + 1)
                label_emotions.append(label_emotion[:dia_lengths[j], j])
            label_emo = torch.cat(label_emotions)

            _, _, _, feat_fusion = model(
                textf0, textf1, textf2, textf3, visuf, acouf, umask, qmask, dia_lengths)

            for c in range(n_classes):
                mask = (label_emo == c)
                if mask.sum() > 0:
                    class_sums[c] += feat_fusion[mask].sum(0)
                    class_counts[c] += mask.sum().float()

    # Reduce across DDP processes so all ranks get the same init
    if dist.is_initialized():
        dist.all_reduce(class_sums, op=dist.ReduceOp.SUM)
        dist.all_reduce(class_counts, op=dist.ReduceOp.SUM)

    with torch.no_grad():
        for c in range(n_classes):
            if class_counts[c] > 0:
                anchors.data[c] = class_sums[c] / class_counts[c]


dataset_cls_map = {
    'IEMOCAP': IEMOCAPDataset_BERT,
    'IEMOCAP4': IEMOCAPDataset_BERT4,
    'MELD': MELDDataset_BERT,
    'CMUMOSEI7': CMUMOSEIDataset7,
}
path_map = {
    'IEMOCAP': IEMOCAP_path,
    'IEMOCAP4': IEMOCAP4_path,
    'MELD': MELD_path,
    'CMUMOSEI7': CMUMOSEI7_path,
}


def main(local_rank):
    print(f'Running run_eacl main on rank {local_rank}.')
    init_ddp(local_rank)

    today = datetime.datetime.now()
    name_ = args.modals + '_' + args.dataset + '_eacl'

    cuda = torch.cuda.is_available() and not args.no_cuda

    if args.dataset == 'IEMOCAP':
        embedding_dims = [1024, 342, 1582]
        n_classes_emo = 6
    elif args.dataset == 'IEMOCAP4':
        embedding_dims = [1024, 512, 100]
        n_classes_emo = 4
    elif args.dataset == 'MELD':
        embedding_dims = [1024, 342, 300]
        n_classes_emo = 7
    elif args.dataset == 'CMUMOSEI7':
        embedding_dims = [1024, 35, 384]
        n_classes_emo = 7

    seed_everything()

    # ── Model (DDP-wrapped backbone) ──────────────────────────────────────────
    model = GraphSmile(args, embedding_dims, n_classes_emo)
    model = model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                find_unused_parameters=True)

    # ── Emotion anchors: standalone nn.Parameter, NOT inside DDP ─────────────
    # Keeping anchors outside DDP prevents the "parameter marked ready twice" error
    # that occurs when a parameter is unused in forward() but used in the loss backward.
    # Manual dist.all_reduce in sync_anchor_grad() handles multi-GPU gradient averaging.
    emotion_anchors = nn.Parameter(
        torch.randn(n_classes_emo, args.hidden_dim, device=local_rank))
    nn.init.xavier_uniform_(emotion_anchors.data.unsqueeze(0))

    # Loss functions
    loss_fn_sen = nn.NLLLoss()
    loss_fn_shift = nn.NLLLoss()
    loss_fn_eacl = EmotionAnchoredContrastiveLoss(temperature=args.temperature)
    loss_fn_anchor = AnchorAngleLoss()
    loss_fn_ada = AnchorAdaptationLoss(temperature=args.temperature)

    # Data loaders
    train_loader, valid_loader, test_loader = get_data_loaders(
        path_map[args.dataset], dataset_cls_map[args.dataset],
        args.batch_size, 0.1, 0, False)

    # Anchor warm-start: per-class mean of feat_fusion on one pass of training data
    init_anchors_from_data(model, emotion_anchors, train_loader, cuda, n_classes_emo)
    if local_rank == 0:
        print("Emotion anchors initialized from per-class mean of feat_fusion.")

    lambd_sen = args.lambd[1] if len(args.lambd) > 1 else 1.0
    lambd_sft = args.lambd[2] if len(args.lambd) > 2 else 1.0

    best_f1_emo, best_f1_sen = None, None
    best_label_emo, best_pred_emo = None, None
    best_label_sen, best_pred_sen = None, None
    all_f1_emo, all_acc_emo = [], []

    # ── Stage 1: train backbone + anchors together ────────────────────────────
    print(f"\n=== EACL Stage 1: {args.stage1_epochs} epochs (backbone + anchors) ===")

    optimizer = optim.AdamW(
        list(model.parameters()) + [emotion_anchors],
        lr=args.lr, weight_decay=args.l2, amsgrad=True)

    for epoch in range(args.stage1_epochs):
        trainset = dataset_cls_map[args.dataset](path_map[args.dataset])
        setup_samplers(trainset, valid_ratio=0.1, epoch=epoch)
        start_time = time.time()

        train_loss, _, _, train_acc_emo, train_f1_emo, _, _, train_acc_sen, train_f1_sen, _ = \
            train_or_eval_eacl_stage1(
                model, emotion_anchors,
                loss_fn_eacl, loss_fn_anchor, loss_fn_sen, loss_fn_shift,
                train_loader, epoch, cuda, args.modals, optimizer, True,
                args.dataset, args.lambda1, args.lambda2, lambd_sen, lambd_sft,
                args.loss_type, args.shift_win)

        # Sync anchor gradients across GPUs (backbone grads are synced by DDP)
        sync_anchor_grad(emotion_anchors)

        valid_loss, _, _, valid_acc_emo, valid_f1_emo, _, _, valid_acc_sen, valid_f1_sen, _ = \
            train_or_eval_eacl_stage1(
                model, emotion_anchors,
                loss_fn_eacl, loss_fn_anchor, loss_fn_sen, loss_fn_shift,
                valid_loader, epoch, cuda, args.modals, None, False,
                args.dataset, args.lambda1, args.lambda2, lambd_sen, lambd_sft,
                args.loss_type, args.shift_win)

        print(f'[S1] epoch: {epoch+1}/{args.stage1_epochs}, '
              f'train_loss: {train_loss}, train_f1_emo: {train_f1_emo}, '
              f'valid_loss: {valid_loss}, valid_f1_emo: {valid_f1_emo}')

        if local_rank == 0:
            test_loss, test_label_emo, test_pred_emo, test_acc_emo, test_f1_emo, \
            test_label_sen, test_pred_sen, test_acc_sen, test_f1_sen, _ = \
                train_or_eval_eacl_stage1(
                    model, emotion_anchors,
                    loss_fn_eacl, loss_fn_anchor, loss_fn_sen, loss_fn_shift,
                    test_loader, epoch, cuda, args.modals, None, False,
                    args.dataset, args.lambda1, args.lambda2, lambd_sen, lambd_sft,
                    args.loss_type, args.shift_win)

            all_f1_emo.append(test_f1_emo)
            all_acc_emo.append(test_acc_emo)
            print(f'[S1] test_loss: {test_loss}, test_acc_emo: {test_acc_emo}, '
                  f'test_f1_emo: {test_f1_emo}, test_acc_sen: {test_acc_sen}, '
                  f'test_f1_sen: {test_f1_sen}, '
                  f'time: {round(time.time()-start_time,2)}s')
            print('-' * 100)

            if args.classify == 'emotion':
                if best_f1_emo is None or best_f1_emo < test_f1_emo:
                    best_f1_emo = test_f1_emo
                    best_f1_sen = test_f1_sen
                    best_label_emo, best_pred_emo = test_label_emo, test_pred_emo
                    best_label_sen, best_pred_sen = test_label_sen, test_pred_sen
            elif args.classify == 'sentiment':
                if best_f1_sen is None or best_f1_sen < test_f1_sen:
                    best_f1_emo = test_f1_emo
                    best_f1_sen = test_f1_sen
                    best_label_emo, best_pred_emo = test_label_emo, test_pred_emo
                    best_label_sen, best_pred_sen = test_label_sen, test_pred_sen

            if (epoch + 1) % 10 == 0:
                np.set_printoptions(suppress=True)
                print(classification_report(best_label_emo, best_pred_emo,
                                            digits=4, zero_division=0))
                print(confusion_matrix(best_label_emo, best_pred_emo))
                print('-' * 100)

        dist.barrier()

    # ── Stage 2: freeze backbone, adapt anchors only ──────────────────────────
    print(f"\n=== EACL Stage 2: {args.stage2_epochs} epochs (anchors only) ===")

    raw_model = model.module if hasattr(model, 'module') else model
    raw_model.freeze_backbone()

    optimizer2 = optim.AdamW([emotion_anchors], lr=args.stage2_lr,
                              weight_decay=0, amsgrad=True)

    best_f1_emo_s2, best_label_emo_s2, best_pred_emo_s2 = None, None, None

    for epoch in range(args.stage2_epochs):
        trainset = dataset_cls_map[args.dataset](path_map[args.dataset])
        setup_samplers(trainset, valid_ratio=0.1, epoch=args.stage1_epochs + epoch)
        start_time = time.time()

        train_loss, _, _, train_acc_emo, train_f1_emo = train_or_eval_eacl_stage2(
            model, emotion_anchors, loss_fn_ada, train_loader, cuda, optimizer2, True)

        sync_anchor_grad(emotion_anchors)

        valid_loss, _, _, valid_acc_emo, valid_f1_emo = train_or_eval_eacl_stage2(
            model, emotion_anchors, loss_fn_ada, valid_loader, cuda, None, False)

        print(f'[S2] epoch: {epoch+1}/{args.stage2_epochs}, '
              f'train_loss: {train_loss}, train_f1_emo: {train_f1_emo}, '
              f'valid_f1_emo: {valid_f1_emo}')

        if local_rank == 0:
            test_loss, test_label_s2, test_pred_s2, test_acc_s2, test_f1_s2 = \
                train_or_eval_eacl_stage2(
                    model, emotion_anchors, loss_fn_ada, test_loader, cuda, None, False)

            all_f1_emo.append(test_f1_s2)
            all_acc_emo.append(test_acc_s2)
            print(f'[S2] test_loss: {test_loss}, test_acc_emo: {test_acc_s2}, '
                  f'test_f1_emo: {test_f1_s2}, '
                  f'time: {round(time.time()-start_time,2)}s')
            print('-' * 100)

            if best_f1_emo_s2 is None or best_f1_emo_s2 < test_f1_s2:
                best_f1_emo_s2 = test_f1_s2
                best_label_emo_s2 = test_label_s2
                best_pred_emo_s2 = test_pred_s2

        dist.barrier()

    # Use stage 2 result if it beats stage 1
    if local_rank == 0 and best_f1_emo_s2 is not None:
        if best_f1_emo is None or best_f1_emo_s2 > best_f1_emo:
            best_f1_emo = best_f1_emo_s2
            best_label_emo = best_label_emo_s2
            best_pred_emo = best_pred_emo_s2
            print(f"Stage 2 anchor classifier improved best F1 to {best_f1_emo}")

    raw_model.unfreeze_backbone()

    # ── Final reporting ────────────────────────────────────────────────────────
    if local_rank == 0:
        print("\nTest performance (GraphSmile + EACL)..")
        print(f"Best Acc: {max(all_acc_emo)}, Best F-Score: {max(all_f1_emo)}")

        result_path = f"results/record_{today.year}_{today.month}_{today.day}.pk"
        if not os.path.exists(result_path):
            with open(result_path, "wb") as f:
                pk.dump({}, f)
        with open(result_path, "rb") as f:
            record = pk.load(f)
        record.setdefault(name_, []).append(max(all_f1_emo))
        record.setdefault(name_ + "record", []).append(
            classification_report(best_label_emo, best_pred_emo, digits=4, zero_division=0))
        with open(result_path, "wb") as f:
            pk.dump(record, f)

        print(classification_report(best_label_emo, best_pred_emo, digits=4, zero_division=0))
        print(confusion_matrix(best_label_emo, best_pred_emo))

    dist.destroy_process_group()


if __name__ == "__main__":
    print(args)
    print("torch.cuda.is_available():", torch.cuda.is_available())
    n_gpus = torch.cuda.device_count()
    print(f"Use {n_gpus} GPUs")
    mp.spawn(fn=main, args=(), nprocs=n_gpus)
