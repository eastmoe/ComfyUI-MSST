from .datamodule import SimpleDataModule


def __getattr__(name):
	if name == "DivideAndRemasterDataModule":
		from .dnr.datamodule import DivideAndRemasterDataModule

		return DivideAndRemasterDataModule
	if name == "MUSDB18DataModule":
		from .musdb.datamodule import MUSDB18DataModule

		return MUSDB18DataModule
	raise AttributeError(name)


__all__ = ["SimpleDataModule", "DivideAndRemasterDataModule", "MUSDB18DataModule"]
