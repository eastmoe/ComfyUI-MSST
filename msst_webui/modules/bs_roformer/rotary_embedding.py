import torch
from torch import nn


def _rotate_half(x):
	x = x.reshape(*x.shape[:-1], -1, 2)
	x1, x2 = x.unbind(dim=-1)
	return torch.stack((-x2, x1), dim=-1).flatten(-2)


class RotaryEmbedding(nn.Module):
	"""Minimal rotary embedding used by BS-RoFormer attention blocks."""

	def __init__(self, dim, theta=10000, interpolate_factor=1.0):
		super().__init__()
		if dim % 2 != 0:
			raise ValueError("RotaryEmbedding dim must be even")
		if interpolate_factor < 1.0:
			raise ValueError("interpolate_factor must be >= 1.0")

		freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
		self.register_buffer("freqs", freqs, persistent=False)
		self.interpolate_factor = interpolate_factor
		self._cache = {}

	def _get_freqs(self, seq_len, *, device, dtype, offset=0):
		cache_key = (seq_len, offset, device.type, device.index, dtype)
		if cache_key in self._cache:
			return self._cache[cache_key]

		positions = (torch.arange(seq_len, device=device, dtype=self.freqs.dtype) + offset) / self.interpolate_factor
		freqs = torch.einsum("n,f->nf", positions, self.freqs.to(device=device))
		freqs = torch.repeat_interleave(freqs, repeats=2, dim=-1).to(dtype=dtype)
		self._cache[cache_key] = freqs
		return freqs

	def rotate_queries_or_keys(self, t, seq_dim=None, offset=0, freq_seq_len=None):
		seq_dim = -2 if seq_dim is None else seq_dim
		seq_len = t.shape[seq_dim]
		freq_len = seq_len if freq_seq_len is None else freq_seq_len
		if freq_len < seq_len:
			raise ValueError("freq_seq_len must be >= sequence length")

		freqs = self._get_freqs(freq_len, device=t.device, dtype=t.dtype, offset=offset)[-seq_len:]
		if seq_dim == -3:
			freqs = freqs.unsqueeze(1)

		rot_dim = freqs.shape[-1]
		if rot_dim > t.shape[-1]:
			raise ValueError(f"feature dimension {t.shape[-1]} is too small for rotary dimension {rot_dim}")

		t_rot = t[..., :rot_dim]
		t_pass = t[..., rot_dim:]
		t_rot = (t_rot * freqs.cos()) + (_rotate_half(t_rot) * freqs.sin())
		return torch.cat((t_rot, t_pass), dim=-1)
