from torch.utils.data import DataLoader


class SimpleDataModule:
	def __init__(self, train_dataset, val_dataset, test_dataset, batch_size: int, num_workers: int, **dataloader_kwargs):
		self.train_dataset = train_dataset
		self.val_dataset = val_dataset
		self.test_dataset = test_dataset
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.dataloader_kwargs = dataloader_kwargs

	def train_dataloader(self):
		return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, **self.dataloader_kwargs)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, **self.dataloader_kwargs)

	def test_dataloader(self):
		return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, **self.dataloader_kwargs)

	def predict_dataloader(self):
		return self.test_dataloader()
