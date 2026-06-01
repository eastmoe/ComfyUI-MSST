import torch


ACCELERATOR_DEVICES = ("cpu", "cuda", "xpu", "mps")


def is_xpu_available() -> bool:
	try:
		return hasattr(torch, "xpu") and torch.xpu.is_available()
	except (AttributeError, RuntimeError):
		return False


def is_mps_available() -> bool:
	try:
		return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
	except (AttributeError, RuntimeError):
		return False


def device_type(device) -> str:
	try:
		return torch.device(device).type
	except (RuntimeError, TypeError):
		return str(device).split(":", 1)[0]


def select_device(device="auto", device_ids=None, logger=None) -> str:
	device_ids = device_ids or [0]
	device_id = int(device_ids[0]) if device_ids else 0

	if device == "cpu":
		return "cpu"
	if device == "cuda":
		return "cuda"
	if device == "xpu":
		return f"xpu:{device_id}"
	if device == "mps":
		return "mps"

	if torch.cuda.is_available():
		if logger:
			logger.debug("CUDA is available in Torch, setting Torch device to CUDA")
		return f"cuda:{device_id}"
	if is_xpu_available():
		if logger:
			logger.debug("Intel XPU is available in Torch, setting Torch device to XPU")
		return f"xpu:{device_id}"
	if is_mps_available():
		if logger:
			logger.debug("Apple Silicon MPS/CoreML is available in Torch, setting Torch device to MPS")
		return "mps"

	return "cpu"


def amp_device_type(device) -> str:
	dtype = device_type(device)
	return dtype if dtype in {"cuda", "xpu"} else "cpu"


def amp_enabled(device, requested=True) -> bool:
	return bool(requested) and device_type(device) in {"cuda", "xpu"}


def clear_device_cache(device=None) -> None:
	if device is None or device_type(device) == "cuda":
		if torch.cuda.is_available():
			try:
				torch.cuda.synchronize()
			except RuntimeError:
				pass
			torch.cuda.empty_cache()
			try:
				torch.cuda.ipc_collect()
			except RuntimeError:
				pass

	if device is None or device_type(device) == "xpu":
		if is_xpu_available():
			try:
				torch.xpu.synchronize()
			except (AttributeError, RuntimeError):
				pass
			try:
				torch.xpu.empty_cache()
			except (AttributeError, RuntimeError):
				pass

	if device is None or device_type(device) == "mps":
		if is_mps_available():
			torch.mps.empty_cache()
