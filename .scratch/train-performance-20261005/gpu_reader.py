"""临时 GPU 紧凑块缓存，保留原 reader 的样本顺序和跨块拼 batch。"""
import torch
from precompute_pipeline.common import codec


class PackedDeviceDecoder:
    def __init__(self):
        self.chunk = None
        self.arrays = None
        self.shifts = torch.arange(8, dtype=torch.uint8, device='cuda')
        self.planes = [torch.tensor(x, device='cuda') for x in (
            codec.SCALAR_BINARY_PLANES, codec.TILE_BINARY_PLANES,
            codec.SCALAR_FLOAT_PLANES, codec.TILE_FLOAT_PLANES)]
        self.equality_checked = False
        self.chunks_uploaded = 0

    def __call__(self, chunk, device, indices=None):
        """每个当前块上传一次；选择和逐位恢复均在 GPU 完成。"""
        indices = codec._normalize_indices(indices, chunk.sample_count)
        if chunk is not self.chunk:
            self.chunk = chunk
            self.arrays = {name: torch.tensor(getattr(chunk, name).copy(), device=device)
                for name in ('scalar_binary_bits', 'tile_binary_bits', 'scalar_float_values',
                             'tile_float_values', 'actions', 'mask_bits', 'steps_to_done',
                             'rewards', 'ranks')}
            for name in ('actions', 'steps_to_done', 'ranks'):
                self.arrays[name] = self.arrays[name].long()
            self.chunks_uploaded += 1
        selected = torch.tensor(indices, dtype=torch.long, device=device)
        count = len(indices)
        def unpack(value, width):
            return ((value.unsqueeze(-1) >> self.shifts) & 1).reshape(count, -1)[:, :width]
        scalar = unpack(self.arrays['scalar_binary_bits'][selected], len(codec.SCALAR_BINARY_PLANES))
        tile = unpack(self.arrays['tile_binary_bits'][selected], codec.TILE_BINARY_BITS).reshape(
            count, len(codec.TILE_BINARY_PLANES), codec.OBS_TILES)
        obs = torch.zeros((count, codec.OBS_PLANES, codec.OBS_TILES), device=device)
        obs[:, self.planes[0], :] = scalar.float().unsqueeze(-1)
        obs[:, self.planes[1], :] = tile.float()
        obs[:, self.planes[2], :] = self.arrays['scalar_float_values'][selected].unsqueeze(-1)
        obs[:, self.planes[3], :] = self.arrays['tile_float_values'][selected]
        result = (obs, self.arrays['actions'][selected],
                  unpack(self.arrays['mask_bits'][selected], codec.ACTION_SPACE).bool(),
                  self.arrays['steps_to_done'][selected], self.arrays['rewards'][selected],
                  self.arrays['ranks'][selected])
        if not self.equality_checked:
            reference = codec.decode_chunk_torch(chunk, device, indices[:32])
            if not all(torch.equal(left[:32], right) for left, right in zip(result, reference)):
                raise RuntimeError('GPU compact cache differs from original decoder')
            self.equality_checked = True
        return result
