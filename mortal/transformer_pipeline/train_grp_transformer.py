import _bootstrap
import prelude

import random
import torch
import logging
from os import path
from glob import glob
from datetime import datetime
from torch import optim
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.tensorboard import SummaryWriter
from model import GRP_Transformer
from libriichi.dataset import Grp
from common import tqdm
from config import config


class GrpFileDatasetsIter(IterableDataset):
    def __init__(
        self,
        file_list,
        file_batch_size=50,
        cycle=False
    ):
        # 这个可迭代数据集按“文件批次”读取日志，再把每个对局拆成多个前缀样本。
        super().__init__()
        self.file_list = file_list
        self.file_batch_size = file_batch_size
        self.cycle = cycle
        self.buffer = []
        self.iterator = None

    def build_iter(self):
        # 按文件批次加载、在内存中打乱样本，然后逐个产出。
        # `cycle=True` 时会反复遍历文件列表，用于无限训练流。
        while True:
            random.shuffle(self.file_list)
            for start_idx in range(0, len(self.file_list), self.file_batch_size):
                self.populate_buffer(start_idx)
                buffer_size = len(self.buffer)
                for i in random.sample(range(buffer_size), buffer_size):
                    yield self.buffer[i]
                self.buffer.clear()
            if not self.cycle:
                break

    def populate_buffer(self, start_idx):
        # 把一个文件批次里的 mjai 日志读出来，并把每局游戏展开成
        # “前 1 局 / 前 2 局 / ... / 前 n 局”的所有前缀序列样本。
        file_list = self.file_list[start_idx:start_idx + self.file_batch_size]
        data = Grp.load_gz_log_files(file_list)

        for game in data:
            feature = game.take_feature()
            rank_by_player = game.take_rank_by_player()

            for i in range(feature.shape[0]):
                inputs_seq = torch.as_tensor(feature[:i + 1], dtype=torch.float64)
                self.buffer.append((
                    inputs_seq,
                    rank_by_player,
                ))

    def __iter__(self):
        # DataLoader 会反复调用 __iter__；这里缓存生成器，避免重复初始化。
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator


def collate(batch):
    # Transformer 不能像 GRU 那样吃 PackedSequence。
    # 这里把变长前缀序列补齐成一个 dense batch，并额外构造 bool mask，
    # 其中 True 表示该位置是 padding，需要在注意力里被忽略。
    pin_memory = torch.cuda.is_available()
    inputs = []
    lengths = []
    rank_by_players = []

    for inputs_seq, rank_by_player in batch:
        inputs.append(inputs_seq)
        lengths.append(len(inputs_seq))
        rank_by_players.append(rank_by_player)

    lengths = torch.tensor(lengths, dtype=torch.int64)
    rank_by_players = torch.tensor(rank_by_players, dtype=torch.int64, pin_memory=pin_memory)

    padded_inputs = pad_sequence(inputs, batch_first=True)
    batch_size, max_len, _ = padded_inputs.shape
    padding_mask = torch.arange(max_len).expand(batch_size, max_len) >= lengths.unsqueeze(1)

    if pin_memory:
        padded_inputs = padded_inputs.pin_memory()
        padding_mask = padding_mask.pin_memory()

    return padded_inputs, padding_mask, rank_by_players


def build_file_index(train_globs, val_globs, file_index):
    # 根据配置里的 glob 规则扫描训练/验证文件，并把结果缓存到索引文件。
    # 下次启动时如果索引仍然有效，就不必再次遍历整个目录树。
    train_file_list = []
    val_file_list = []
    for pat in train_globs:
        train_file_list.extend(glob(pat, recursive=True))
    for pat in val_globs:
        val_file_list.extend(glob(pat, recursive=True))
    train_file_list.sort(reverse=True)
    val_file_list.sort(reverse=True)
    torch.save({'train_file_list': train_file_list, 'val_file_list': val_file_list}, file_index)
    return train_file_list, val_file_list


def train():
    # 训练主流程基本复用 train_grp.py：
    # 读取配置 -> 建模/恢复 -> 构造 DataLoader -> 训练 -> 验证 -> 存盘。
    # 唯一的关键差异是 batch 输入从 PackedSequence 换成了 padding mask 流程。
    cfg = config['grp_transformer']
    batch_size = cfg['control']['batch_size']
    save_every = cfg['control']['save_every']
    val_steps = cfg['control']['val_steps']

    requested_device = torch.device(cfg['control']['device'])
    if requested_device.type == 'cuda' and not torch.cuda.is_available():
        logging.warning(
            'configured device %s is unavailable; falling back to cpu',
            requested_device,
        )
        device = torch.device('cpu')
    else:
        device = requested_device
    torch.backends.cudnn.benchmark = cfg['control']['enable_cudnn_benchmark']
    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    grp = GRP_Transformer(**cfg['network']).to(device)
    optimizer = optim.AdamW(grp.parameters())

    state_file = cfg['state_file']
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        grp.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        steps = state['steps']
    else:
        steps = 0

    lr = cfg['optim']['lr']
    optimizer.param_groups[0]['lr'] = lr

    file_index = cfg['dataset']['file_index']
    train_globs = cfg['dataset']['train_globs']
    val_globs = cfg['dataset']['val_globs']
    if path.exists(file_index):
        index = torch.load(file_index, weights_only=True)
        train_file_list = index['train_file_list']
        val_file_list = index['val_file_list']
        missing_train = sum(not path.exists(p) for p in train_file_list)
        missing_val = sum(not path.exists(p) for p in val_file_list)
        if missing_train or missing_val:
            logging.warning(
                'file index is stale; rebuilding (missing train=%s, missing val=%s)',
                missing_train,
                missing_val,
            )
            train_file_list, val_file_list = build_file_index(train_globs, val_globs, file_index)
    else:
        logging.info('building file index...')
        train_file_list, val_file_list = build_file_index(train_globs, val_globs, file_index)
    writer = SummaryWriter(cfg['control']['tensorboard_dir'])

    train_file_data = GrpFileDatasetsIter(
        file_list=train_file_list,
        file_batch_size=cfg['dataset']['file_batch_size'],
        cycle=True,
    )
    train_data_loader = iter(DataLoader(
        dataset=train_file_data,
        batch_size=batch_size,
        drop_last=True,
        num_workers=0,
        collate_fn=collate,
    ))

    val_file_data = GrpFileDatasetsIter(
        file_list=val_file_list,
        file_batch_size=cfg['dataset']['file_batch_size'],
        cycle=True,
    )
    val_data_loader = iter(DataLoader(
        dataset=val_file_data,
        batch_size=batch_size,
        drop_last=True,
        num_workers=0,
        collate_fn=collate,
    ))

    stats = {
        'train_loss': 0,
        'train_acc': 0,
        'val_loss': 0,
        'val_acc': 0,
    }
    logging.info(f'train file list size: {len(train_file_list):,}')
    logging.info(f'val file list size: {len(val_file_list):,}')

    approx_percent = steps * batch_size / (len(train_file_list) * 10) * 100
    logging.info(f'total steps: {steps:,} est. {approx_percent:6.3f}%')

    pb = tqdm(total=save_every, desc='TRAIN')
    for inputs, padding_mask, rank_by_players in train_data_loader:
        inputs = inputs.to(dtype=torch.float64, device=device)
        padding_mask = padding_mask.to(dtype=torch.bool, device=device)
        rank_by_players = rank_by_players.to(dtype=torch.int64, device=device)

        logits = grp.forward_padded(inputs, padding_mask)
        labels = grp.get_label(rank_by_players)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.inference_mode():
            stats['train_loss'] += loss
            stats['train_acc'] += (logits.argmax(-1) == labels).to(torch.float64).mean()

        steps += 1
        pb.update(1)

        if steps % save_every == 0:
            pb.close()

            with torch.inference_mode():
                grp.eval()
                pb = tqdm(total=val_steps, desc='VAL')
                for idx, (inputs, padding_mask, rank_by_players) in enumerate(val_data_loader):
                    if idx == val_steps:
                        break
                    inputs = inputs.to(dtype=torch.float64, device=device)
                    padding_mask = padding_mask.to(dtype=torch.bool, device=device)
                    rank_by_players = rank_by_players.to(dtype=torch.int64, device=device)

                    logits = grp.forward_padded(inputs, padding_mask)
                    labels = grp.get_label(rank_by_players)
                    loss = F.cross_entropy(logits, labels)

                    stats['val_loss'] += loss
                    stats['val_acc'] += (logits.argmax(-1) == labels).to(torch.float64).mean()
                    pb.update(1)
                pb.close()
                grp.train()

            writer.add_scalars('loss', {
                'train': stats['train_loss'] / save_every,
                'val': stats['val_loss'] / val_steps,
            }, steps)
            writer.add_scalars('acc', {
                'train': stats['train_acc'] / save_every,
                'val': stats['val_acc'] / val_steps,
            }, steps)
            writer.add_scalar('lr', lr, steps)
            writer.flush()

            for k in stats:
                stats[k] = 0
            approx_percent = steps * batch_size / (len(train_file_list) * 10) * 100
            logging.info(f'total steps: {steps:,} est. {approx_percent:6.3f}%')

            state = {
                'model': grp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'steps': steps,
                'timestamp': datetime.now().timestamp(),
            }
            torch.save(state, state_file)
            pb = tqdm(total=save_every, desc='TRAIN')
    pb.close()


if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        pass
