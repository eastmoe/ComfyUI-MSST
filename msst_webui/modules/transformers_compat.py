from importlib import metadata


MIN_TRANSFORMERS_VERSION = (4, 35, 0)
MAX_TRANSFORMERS_MAJOR = 6


def _parse_version(version):
	parts = []
	for part in version.split("."):
		number = ""
		for char in part:
			if not char.isdigit():
				break
			number += char
		if number == "":
			break
		parts.append(int(number))

	while len(parts) < 3:
		parts.append(0)

	return tuple(parts[:3])


def _validate_transformers_version():
	try:
		version = metadata.version("transformers")
	except metadata.PackageNotFoundError as exc:
		raise ImportError(
			"The swin_upernet model requires Hugging Face transformers. "
			"Install transformers>=4.35,<6 in the ComfyUI Python environment."
		) from exc

	parsed = _parse_version(version)
	if parsed < MIN_TRANSFORMERS_VERSION or parsed[0] >= MAX_TRANSFORMERS_MAJOR:
		raise ImportError(
			f"Unsupported transformers version {version}. "
			"ComfyUI-MSST supports transformers>=4.35,<6."
		)

	return version


def get_upernet_for_semantic_segmentation():
	_validate_transformers_version()

	try:
		from transformers import UperNetForSemanticSegmentation
	except ImportError:
		try:
			from transformers.models.upernet import UperNetForSemanticSegmentation
		except ImportError as exc:
			try:
				from transformers.models.upernet.modeling_upernet import UperNetForSemanticSegmentation
			except ImportError:
				raise ImportError(
					"Could not import UperNetForSemanticSegmentation from transformers. "
					"Use a transformers 4.x or 5.x release that includes the UPerNet model."
				) from exc

	return UperNetForSemanticSegmentation
