from collections.abc import Mapping


class AttrDict(dict):
	"""Small dict subclass with recursive attribute access for YAML configs."""

	def __init__(self, *args, **kwargs):
		super().__init__()
		if args == (None,):
			args = ()
		self.update(*args, **kwargs)

	def __getattribute__(self, key):
		if not key.startswith("_"):
			try:
				return dict.__getitem__(self, key)
			except KeyError:
				pass
		return super().__getattribute__(key)

	def __setattr__(self, key, value):
		self[key] = self._wrap(value)

	def __setitem__(self, key, value):
		super().__setitem__(key, self._wrap(value))

	def update(self, *args, **kwargs):
		for key, value in dict(*args, **kwargs).items():
			self[key] = value

	def copy(self):
		return AttrDict(self)

	def to_dict(self):
		return {key: self._unwrap(value) for key, value in dict.items(self)}

	@classmethod
	def _wrap(cls, value):
		if isinstance(value, cls):
			return value
		if isinstance(value, Mapping):
			return cls(value)
		if isinstance(value, list):
			return [cls._wrap(item) for item in value]
		if isinstance(value, tuple):
			return tuple(cls._wrap(item) for item in value)
		return value

	@classmethod
	def _unwrap(cls, value):
		if isinstance(value, cls):
			return value.to_dict()
		if isinstance(value, list):
			return [cls._unwrap(item) for item in value]
		if isinstance(value, tuple):
			return tuple(cls._unwrap(item) for item in value)
		return value
