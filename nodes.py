import gc
import json
import os
import shutil
import sys
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


PLUGIN_DIR = Path(__file__).resolve().parent
DEFAULT_MSST_ROOT = PLUGIN_DIR / "msst_webui"
MSST_ROOT = Path(os.environ.get("COMFY_MSST_WEBUI_PATH", DEFAULT_MSST_ROOT)).resolve()
try:
    import folder_paths

    DEFAULT_MSST_MODEL_BASE = Path(folder_paths.models_dir) / "MSST"
    DEFAULT_OUTPUT_ROOT = Path(folder_paths.get_output_directory()) / "MSST"
except Exception:
    DEFAULT_MSST_MODEL_BASE = PLUGIN_DIR / "models" / "MSST"
    DEFAULT_OUTPUT_ROOT = PLUGIN_DIR / "output" / "MSST"
DEFAULT_MODEL_ROOT = DEFAULT_MSST_MODEL_BASE / "pretrain"
MODEL_ROOT = Path(os.environ.get("COMFY_MSST_MODEL_ROOT", DEFAULT_MODEL_ROOT)).resolve()
SOME_WEIGHT_ROOT = Path(os.environ.get("COMFY_MSST_SOME_WEIGHT_ROOT", DEFAULT_MSST_MODEL_BASE / "SOME_weights")).resolve()
OUTPUT_ROOT = Path(os.environ.get("COMFY_MSST_OUTPUT_ROOT", DEFAULT_OUTPUT_ROOT)).resolve()
CATEGORY = "音频/Comfy-MSST"

MODEL_TYPES = [
    "bs_roformer",
    "mel_band_roformer",
    "segm_models",
    "htdemucs",
    "mdx23c",
    "swin_upernet",
    "bandit",
    "bandit_v2",
    "scnet",
    "scnet_unofficial",
    "torchseg",
    "apollo",
    "bs_mamba2",
]
MODEL_CLASSES = ["vocal_models", "multi_stem_models", "single_stem_models"]
MODEL_EXTENSIONS = {".ckpt", ".pth", ".th", ".chpt", ".pt", ".bin"}
ENSEMBLE_MODES = ["avg_wave", "median_wave", "min_wave", "max_wave", "avg_fft", "median_fft", "min_fft", "max_fft"]
STEM_PRESETS = [
    "vocals",
    "instrumental",
    "drums",
    "bass",
    "other",
    "guitar",
    "piano",
    "dry",
    "wet",
    "no noise",
    "noise",
    "no reverb",
    "reverb",
    "no echo",
    "echo",
    "restored",
]
STEM_CHOICE_FALLBACK = "custom"

_RUNTIME_LOCK = threading.RLock()
_MODEL_CACHE: Dict[Tuple[Any, ...], Any] = {}


def _ui(display_name: str, tooltip: str, **extra: Any) -> Dict[str, Any]:
    extra["display_name"] = display_name
    extra["tooltip"] = tooltip
    return extra


@dataclass(frozen=True)
class MSSTModelSpec:
    model_type: str
    model_path: str
    config_path: str
    model_name: str = ""


@dataclass(frozen=True)
class VRModelSpec:
    model_path: str
    model_name: str = ""


def _catalog_path() -> Path:
    for rel in ("data/models_info.json", "data_backup/models_info.json"):
        path = MSST_ROOT / rel
        if path.is_file():
            return path
    return MSST_ROOT / "data_backup/models_info.json"


def _load_catalog() -> Dict[str, Dict[str, Any]]:
    path = _catalog_path()
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _catalog_choices(classes: Iterable[str]) -> List[str]:
    catalog = _load_catalog()
    choices = [name for name, data in catalog.items() if data.get("model_class") in classes]
    return sorted(choices, key=str.lower) or ["manual"]


def _normalise_rel(path: str) -> str:
    text = path.strip().strip('"').replace("\\", "/").lstrip("./")
    if text.startswith("pretrain/"):
        text = text[len("pretrain/") :]
    return text


def _catalog_entry_for_model(model_rel: str) -> Optional[Tuple[str, Dict[str, Any]]]:
    wanted = _normalise_rel(model_rel)
    by_name: List[Tuple[str, Dict[str, Any]]] = []
    for name, data in _load_catalog().items():
        target = _normalise_rel(str(data.get("target_position", "")))
        if target == wanted:
            return name, data
        if Path(target).name == Path(wanted).name:
            by_name.append((name, data))
    return by_name[0] if len(by_name) == 1 else None


def _scan_model_choices(classes: Iterable[str]) -> List[str]:
    choices: List[str] = []
    for model_class in classes:
        root = MODEL_ROOT / model_class
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in MODEL_EXTENSIONS:
                choices.append(path.relative_to(MODEL_ROOT).as_posix())
    return sorted(choices, key=str.lower) or ["No local models found"]


def _local_msst_choices() -> List[str]:
    return _scan_model_choices(MODEL_CLASSES)


def _local_vr_choices() -> List[str]:
    return _scan_model_choices(["VR_Models"])


def _config_for_model_rel(model_rel: str) -> Path:
    rel = Path(_normalise_rel(model_rel) + ".yaml")
    candidates = [
        _resolve_runtime_path(str(Path("configs") / rel)),
        _resolve_runtime_path(str(Path("configs_backup") / rel)),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[-1]


def _infer_model_type_from_name(model_name: str) -> Optional[str]:
    name = model_name.lower()
    if "htdemucs" in name or name.endswith(".th"):
        return "htdemucs"
    if "mdx23c" in name or "drumsep" in name:
        return "mdx23c"
    if "bs_mamba2" in name:
        return "bs_mamba2"
    if "bs_roformer" in name or "bs-roformer" in name:
        return "bs_roformer"
    if "mel_band_roformer" in name or "melband_roformer" in name or "_mbr" in name or "mbr_" in name:
        return "mel_band_roformer"
    if "bandit_v2" in name:
        return "bandit_v2"
    if "bandit" in name:
        return "bandit"
    if "scnet_unofficial" in name:
        return "scnet_unofficial"
    if "scnet" in name:
        return "scnet"
    if "torchseg" in name:
        return "torchseg"
    if "apollo" in name:
        return "apollo"
    if "swin_upernet" in name:
        return "swin_upernet"
    if "segm" in name:
        return "segm_models"
    return None


def _add_stem_choice(choices: Dict[str, str], name: Any) -> None:
    if name is None:
        return
    text = str(name).strip()
    if not text or text.lower() in {"none", "null", "~"}:
        return
    choices.setdefault(_normalize_key(text), text)


def _read_config_stems(path: Path) -> List[str]:
    try:
        import yaml

        with path.open("r", encoding="utf-8") as file:
            config = yaml.load(file, Loader=yaml.BaseLoader) or {}
    except Exception:
        return []

    training = config.get("training") or {}
    stems: List[str] = []
    instruments = training.get("instruments") or []
    if isinstance(instruments, list):
        stems.extend(str(item) for item in instruments)
    target = training.get("target_instrument")
    if target:
        stems.append(str(target))
    return stems


def _stem_choices() -> List[str]:
    choices: Dict[str, str] = {}
    for stem in STEM_PRESETS:
        _add_stem_choice(choices, stem)

    for data in _load_catalog().values():
        _add_stem_choice(choices, data.get("primary_stem"))
        _add_stem_choice(choices, data.get("secondary_stem"))

    for root_name in ("configs", "configs_backup"):
        root = MSST_ROOT / root_name
        if not root.is_dir():
            continue
        for path in root.rglob("*.yaml"):
            for stem in _read_config_stems(path):
                _add_stem_choice(choices, stem)

    return list(choices.values()) + [STEM_CHOICE_FALLBACK]


def _local_msst_spec(model_rel: str) -> MSSTModelSpec:
    if model_rel == "No local models found":
        raise FileNotFoundError(f"No MSST models were found under: {MODEL_ROOT}")
    model_path = _resolve_model_path(model_rel)
    config_path = _config_for_model_rel(model_rel)
    _check_file(str(config_path), "MSST config")

    entry = _catalog_entry_for_model(model_rel)
    model_name = Path(model_rel).name
    if entry is not None:
        catalog_name, data = entry
        model_type = data.get("model_type")
        model_name = data.get("model_name") or catalog_name or model_name
    else:
        model_type = _infer_model_type_from_name(model_name)

    if not model_type:
        raise ValueError(f"Could not infer model type for {model_rel}. Add it to models_info.json or use MSST Model From Paths.")

    return MSSTModelSpec(
        model_type=str(model_type),
        model_path=str(model_path),
        config_path=str(config_path),
        model_name=model_name,
    )


def _resolve_runtime_path(path: str) -> Path:
    raw = Path(path.strip().strip('"'))
    if raw.is_absolute():
        return raw
    return (MSST_ROOT / raw).resolve()


def _resolve_some_weight_path(path: str) -> Path:
    raw_text = path.strip().strip('"').replace("\\", "/")
    raw = Path(raw_text)
    if raw.is_absolute():
        return raw
    parts = list(raw.parts)
    if parts and parts[0] == ".":
        parts = parts[1:]
    if len(parts) >= 2 and parts[0] == "tools" and parts[1] == "SOME_weights":
        return (SOME_WEIGHT_ROOT / Path(*parts[2:])).resolve()
    return (SOME_WEIGHT_ROOT / raw).resolve()


def _resolve_model_path(path: str) -> Path:
    raw_text = path.strip().strip('"').replace("\\", "/")
    raw = Path(raw_text)
    if raw.is_absolute():
        return raw
    parts = list(raw.parts)
    if parts and parts[0] == ".":
        parts = parts[1:]
    if parts and parts[0] == "pretrain":
        return (MODEL_ROOT / Path(*parts[1:])).resolve()
    return (MODEL_ROOT / raw).resolve()


def _config_from_catalog_target(target_position: str) -> Path:
    target = target_position.strip().strip('"').replace("\\", "/").lstrip("./")
    if target.startswith("pretrain/"):
        config_rel = target.replace("pretrain/", "configs/", 1) + ".yaml"
        backup_rel = target.replace("pretrain/", "configs_backup/", 1) + ".yaml"
        for rel in (config_rel, backup_rel):
            candidate = _resolve_runtime_path(rel)
            if candidate.is_file():
                return candidate
        return _resolve_runtime_path(backup_rel)
    return _resolve_runtime_path(target + ".yaml")


def _ensure_runtime_layout() -> None:
    if not MSST_ROOT.is_dir():
        raise FileNotFoundError(
            f"MSST-WEBUI source directory was not found: {MSST_ROOT}. "
            "Set COMFY_MSST_WEBUI_PATH to the original MSST-WEBUI directory."
        )

    for src_name, dst_name in (("data_backup", "data"), ("configs_backup", "configs")):
        src = MSST_ROOT / src_name
        dst = MSST_ROOT / dst_name
        if src.is_dir() and not dst.exists():
            shutil.copytree(src, dst)

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    SOME_WEIGHT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


@contextmanager
def _msst_runtime():
    with _RUNTIME_LOCK:
        _ensure_runtime_layout()
        old_cwd = Path.cwd()
        msst_root_str = str(MSST_ROOT)
        swapped_modules = _swap_conflicting_modules(MSST_ROOT)
        inserted = False
        if sys.path[0] != msst_root_str:
            sys.path.insert(0, msst_root_str)
            inserted = True
        os.chdir(MSST_ROOT)
        try:
            yield
        finally:
            os.chdir(old_cwd)
            if inserted:
                try:
                    sys.path.remove(msst_root_str)
                except ValueError:
                    pass
            _restore_conflicting_modules(swapped_modules, MSST_ROOT)


def _check_file(path: str, label: str) -> None:
    if not Path(path).is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def _module_belongs_to_root(module: Any, root: Path) -> bool:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        return False
    try:
        Path(module_file).resolve().relative_to(root)
        return True
    except ValueError:
        return False


def _swap_conflicting_modules(root: Path) -> Dict[str, Any]:
    swapped: Dict[str, Any] = {}
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            module = sys.modules.get(name)
            if module is not None and not _module_belongs_to_root(module, root):
                swapped[name] = module
                del sys.modules[name]
    return swapped


def _restore_conflicting_modules(swapped: Dict[str, Any], root: Path) -> None:
    for name in list(sys.modules):
        if name == "utils" or name.startswith("utils."):
            module = sys.modules.get(name)
            if module is not None and _module_belongs_to_root(module, root):
                del sys.modules[name]
    sys.modules.update(swapped)


def _audio_to_numpy(audio: Dict[str, Any]) -> Tuple[np.ndarray, int]:
    waveform = audio["waveform"]
    sample_rate = int(audio["sample_rate"])
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.as_tensor(waveform)
    waveform = waveform.detach().cpu().float()
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 3:
        raise ValueError(f"Expected AUDIO waveform with shape [batch, channels, samples], got {tuple(waveform.shape)}")
    return waveform.numpy(), sample_rate


def _numpy_to_audio(batch: np.ndarray, sample_rate: int) -> Dict[str, Any]:
    if batch.ndim == 2:
        batch = batch[None, :, :]
    tensor = torch.from_numpy(np.ascontiguousarray(batch)).float()
    return {"waveform": tensor, "sample_rate": int(sample_rate)}


def _stem_to_channels_first(stem: np.ndarray) -> np.ndarray:
    stem = np.asarray(stem, dtype=np.float32)
    if stem.ndim == 1:
        return stem[None, :]
    if stem.ndim != 2:
        raise ValueError(f"Expected separated stem to be 1D or 2D, got {stem.shape}")
    if stem.shape[0] <= 8 and stem.shape[0] < stem.shape[1]:
        return stem
    return stem.T


def _resample_channels(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    import librosa

    channels = [librosa.resample(y=audio[ch], orig_sr=orig_sr, target_sr=target_sr) for ch in range(audio.shape[0])]
    min_len = min(ch.shape[-1] for ch in channels)
    return np.stack([ch[:min_len] for ch in channels], axis=0).astype(np.float32, copy=False)


def _pad_stack(items: List[np.ndarray]) -> np.ndarray:
    max_channels = max(item.shape[0] for item in items)
    max_samples = max(item.shape[1] for item in items)
    padded = []
    for item in items:
        out = np.zeros((max_channels, max_samples), dtype=np.float32)
        out[: item.shape[0], : item.shape[1]] = item
        padded.append(out)
    return np.stack(padded, axis=0)


def _normalize_key(name: str) -> str:
    return name.strip().lower().replace("_", " ").replace("-", " ")


def _safe_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in name.strip())
    return safe.strip("_") or "audio"


def _make_stems(stems_by_name: Dict[str, List[np.ndarray]], sample_rate: int, meta: Dict[str, Any]) -> Dict[str, Any]:
    stems = {name: _numpy_to_audio(_pad_stack(items), sample_rate) for name, items in stems_by_name.items()}
    return {"stems": stems, "sample_rate": int(sample_rate), "stem_names": list(stems.keys()), "meta": meta}


def _get_separator(cache_key: Tuple[Any, ...], factory, cache_model: bool):
    if cache_model and cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]
    separator = factory()
    if cache_model:
        _MODEL_CACHE[cache_key] = separator
    return separator


def _cleanup_separator(separator: Any, cache_model: bool, cache_key: Tuple[Any, ...]) -> None:
    if not cache_model:
        try:
            separator.del_cache()
        finally:
            _MODEL_CACHE.pop(cache_key, None)
            del separator
            gc.collect()


def _parse_device_ids(device_ids: str) -> List[int]:
    ids = []
    for part in str(device_ids).replace(",", " ").split():
        try:
            ids.append(int(part))
        except ValueError:
            pass
    return ids or [0]


def _parse_optional_int(value: int) -> Optional[int]:
    value = int(value)
    return None if value < 1 else value


def _infer_model_sample_rate(config_path: str, fallback: int = 44100) -> int:
    try:
        import yaml

        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}
        return int((config.get("audio") or {}).get("sample_rate") or fallback)
    except Exception:
        return fallback


def _separate_msst(
    spec: MSSTModelSpec,
    audio: Dict[str, Any],
    device: str,
    device_ids: str,
    use_tta: bool,
    batch_size: int,
    num_overlap: int,
    chunk_size: int,
    normalize: str,
    cache_model: bool,
    debug: bool,
) -> Dict[str, Any]:
    _check_file(spec.model_path, "MSST model")
    _check_file(spec.config_path, "MSST config")

    waveform, source_sr = _audio_to_numpy(audio)
    target_sr = _infer_model_sample_rate(spec.config_path, source_sr)
    inference_params = {
        "batch_size": _parse_optional_int(batch_size),
        "num_overlap": _parse_optional_int(num_overlap),
        "chunk_size": _parse_optional_int(chunk_size),
        "normalize": None if normalize == "config" else normalize == "true",
    }
    cache_key = (
        "msst",
        spec.model_type,
        spec.model_path,
        spec.config_path,
        device,
        tuple(_parse_device_ids(device_ids)),
        use_tta,
        tuple(sorted(inference_params.items())),
        debug,
    )

    with _msst_runtime():
        from inference.msst_infer import MSSeparator
        from utils.logger import get_logger

        def factory():
            return MSSeparator(
                model_type=spec.model_type,
                config_path=spec.config_path,
                model_path=spec.model_path,
                device=device,
                device_ids=_parse_device_ids(device_ids),
                output_format="wav",
                use_tta=use_tta,
                store_dirs="",
                logger=get_logger(),
                debug=debug,
                inference_params=inference_params,
            )

        separator = _get_separator(cache_key, factory, cache_model)
        stems_by_name: Dict[str, List[np.ndarray]] = {}
        try:
            for batch in waveform:
                mix = _resample_channels(batch, source_sr, target_sr)
                result = separator.separate(mix)
                for name, stem in result.items():
                    stems_by_name.setdefault(str(name), []).append(_stem_to_channels_first(stem))
        finally:
            _cleanup_separator(separator, cache_model, cache_key)

    return _make_stems(
        stems_by_name,
        target_sr,
        {"backend": "MSST", "model_type": spec.model_type, "model_name": spec.model_name or Path(spec.model_path).name},
    )


def _separate_vr(
    spec: VRModelSpec,
    audio: Dict[str, Any],
    use_cpu: bool,
    batch_size: int,
    window_size: int,
    aggression: int,
    enable_tta: bool,
    enable_post_process: bool,
    post_process_threshold: float,
    high_end_process: bool,
    cache_model: bool,
    debug: bool,
) -> Dict[str, Any]:
    _check_file(spec.model_path, "VR model")
    waveform, source_sr = _audio_to_numpy(audio)
    target_sr = 44100
    vr_params = {
        "batch_size": int(batch_size),
        "window_size": int(window_size),
        "aggression": int(aggression),
        "enable_tta": bool(enable_tta),
        "enable_post_process": bool(enable_post_process),
        "post_process_threshold": float(post_process_threshold),
        "high_end_process": bool(high_end_process),
    }
    cache_key = ("vr", spec.model_path, use_cpu, tuple(sorted(vr_params.items())), debug)

    with _msst_runtime():
        from inference.vr_infer import VRSeparator
        from utils.logger import get_logger

        def factory():
            return VRSeparator(
                logger=get_logger(),
                debug=debug,
                model_file=spec.model_path,
                output_dir="",
                output_format="wav",
                use_cpu=use_cpu,
                vr_params=vr_params,
            )

        separator = _get_separator(cache_key, factory, cache_model)
        stems_by_name: Dict[str, List[np.ndarray]] = {}
        try:
            for batch in waveform:
                mix = _resample_channels(batch, source_sr, target_sr)
                result = separator.separate(mix)
                for name, stem in result.items():
                    stems_by_name.setdefault(str(name), []).append(_stem_to_channels_first(stem))
        finally:
            _cleanup_separator(separator, cache_model, cache_key)

    return _make_stems(stems_by_name, target_sr, {"backend": "VR", "model_name": spec.model_name or Path(spec.model_path).name})


def _catalog_msst_spec(model_name: str, model_class: str) -> MSSTModelSpec:
    catalog = _load_catalog()
    if model_name not in catalog:
        raise ValueError(f"Model is not present in models_info.json: {model_name}")
    data = catalog[model_name]
    actual_class = data.get("model_class")
    if model_class != "auto" and actual_class != model_class:
        raise ValueError(f"Model {model_name} belongs to {actual_class}, not {model_class}")
    model_path = _resolve_model_path(data["target_position"])
    config_path = _config_from_catalog_target(data["target_position"])
    return MSSTModelSpec(
        model_type=data["model_type"],
        model_path=str(model_path),
        config_path=str(config_path),
        model_name=model_name,
    )


def _catalog_vr_spec(model_name: str) -> VRModelSpec:
    catalog = _load_catalog()
    if model_name not in catalog:
        raise ValueError(f"Model is not present in models_info.json: {model_name}")
    data = catalog[model_name]
    if data.get("model_class") not in ("VR_Models", "UVR_VR_Models"):
        raise ValueError(f"Model {model_name} is not a VR model")
    return VRModelSpec(model_path=str(_resolve_model_path(data["target_position"])), model_name=model_name)


class ComfyMSSTModelFromCatalog:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (_local_msst_choices(), _ui("模型名称", "扫描 models/MSST/pretrain 下已存在的 MSST 模型。选择后会自动匹配同名 YAML 配置。")),
                "model_class": (["auto"] + MODEL_CLASSES, _ui("模型类别", "auto 会自动使用模型所在目录；也可手动限制为人声、单音轨或多音轨模型。")),
            }
        }

    RETURN_TYPES = ("MSST_MODEL", "STRING")
    RETURN_NAMES = ("MSST模型", "模型信息")
    OUTPUT_TOOLTIPS = ("已解析好的 MSST 模型对象，连接到 MSST 分离节点。", "模型名称、架构类型和配置文件路径。")
    DESCRIPTION = "从本地 models/MSST/pretrain 自动列出 MSST 模型，并按模型名自动匹配 configs/configs_backup 中的配置文件。"
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, model_name: str, model_class: str):
        if model_name == "No local models found":
            raise FileNotFoundError(f"No MSST models were found under: {MODEL_ROOT}")
        if "/" in model_name or "\\" in model_name:
            if model_class != "auto":
                actual_class = _normalise_rel(model_name).split("/", 1)[0]
                if actual_class != model_class:
                    raise ValueError(f"Model {model_name} belongs to {actual_class}, not {model_class}")
            spec = _local_msst_spec(model_name)
        else:
            spec = _catalog_msst_spec(model_name, model_class)
        info = f"{spec.model_name} | type={spec.model_type} | config={spec.config_path}"
        return (spec, info)


class ComfyMSSTModelFromPaths:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (MODEL_TYPES, _ui("模型架构", "用于推理的 MSST 模型架构，例如 bs_roformer、mel_band_roformer、htdemucs。")),
                "model_path": ("STRING", _ui("模型路径", "模型文件路径。相对路径会从 models/MSST/pretrain 解析。", default="pretrain/vocal_models/model.ckpt")),
                "config_path": ("STRING", _ui("配置路径", "模型 YAML 配置文件路径。相对路径会从 MSST WebUI 目录解析。", default="configs/vocal_models/model.ckpt.yaml")),
            }
        }

    RETURN_TYPES = ("MSST_MODEL", "STRING")
    RETURN_NAMES = ("MSST模型", "模型信息")
    OUTPUT_TOOLTIPS = ("手动路径构造的 MSST 模型对象。", "模型名称、架构类型和配置文件路径。")
    DESCRIPTION = "手动指定 MSST 模型文件、模型架构和 YAML 配置文件；适合第三方或自训练模型。"
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, model_type: str, model_path: str, config_path: str):
        model = _resolve_model_path(model_path)
        config = _resolve_runtime_path(config_path)
        spec = MSSTModelSpec(model_type=model_type, model_path=str(model), config_path=str(config), model_name=model.name)
        return (spec, f"{spec.model_name} | type={spec.model_type} | config={spec.config_path}")


class ComfyMSSTVRModelFromCatalog:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_name": (_local_vr_choices(), _ui("VR模型名称", "扫描 models/MSST/pretrain/VR_Models 下已存在的 UVR/VR 模型。"))}}

    RETURN_TYPES = ("MSST_VR_MODEL", "STRING")
    RETURN_NAMES = ("VR模型", "模型信息")
    OUTPUT_TOOLTIPS = ("已解析好的 VR 模型对象，连接到 MSST VR 分离节点。", "VR 模型名称和路径。")
    DESCRIPTION = "从本地 models/MSST/pretrain/VR_Models 自动列出 UVR/VR 模型。"
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, model_name: str):
        if "/" in model_name or "\\" in model_name:
            if model_name == "No local models found":
                raise FileNotFoundError(f"No MSST VR models were found under: {MODEL_ROOT / 'VR_Models'}")
            model = _resolve_model_path(model_name)
            entry = _catalog_entry_for_model(model_name)
            resolved_name = entry[0] if entry is not None else model.name
            spec = VRModelSpec(model_path=str(model), model_name=resolved_name)
        else:
            spec = _catalog_vr_spec(model_name)
        return (spec, f"{spec.model_name} | path={spec.model_path}")


class ComfyMSSTVRModelFromPath:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_path": ("STRING", _ui("VR模型路径", "VR/UVR 模型文件路径。相对路径会从 models/MSST/pretrain 解析。", default="pretrain/VR_Models/1_HP-UVR.pth"))}}

    RETURN_TYPES = ("MSST_VR_MODEL", "STRING")
    RETURN_NAMES = ("VR模型", "模型信息")
    OUTPUT_TOOLTIPS = ("手动路径构造的 VR 模型对象。", "VR 模型名称和路径。")
    DESCRIPTION = "手动指定 VR/UVR 模型路径，适合第三方模型。"
    FUNCTION = "load"
    CATEGORY = CATEGORY

    def load(self, model_path: str):
        model = _resolve_model_path(model_path)
        spec = VRModelSpec(model_path=str(model), model_name=model.name)
        return (spec, f"{spec.model_name} | path={spec.model_path}")


class ComfyMSSTSeparate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", _ui("输入音频", "需要分离的混合音频。")),
                "msst_model": ("MSST_MODEL", _ui("MSST模型", "由“加载 MSST 模型”节点输出的模型对象。")),
                "device": (["auto", "cuda", "cpu", "mps"], _ui("推理设备", "auto 自动选择可用硬件；CPU 很慢，通常建议使用 CUDA。", default="auto")),
                "device_ids": ("STRING", _ui("显卡编号", "CUDA 设备编号，多个编号用逗号分隔，例如 0 或 0,1。", default="0")),
                "use_tta": ("BOOLEAN", _ui("启用TTA", "测试时增强。可能略微改善效果，但会显著增加推理时间。", default=False)),
                "cache_model": ("BOOLEAN", _ui("缓存模型", "保持模型在内存/显存中，重复运行更快；显存紧张时可关闭。", default=True)),
            },
            "optional": {
                "batch_size": ("INT", _ui("批大小", "0 表示使用配置文件默认值；增大可能提速但占用更多显存。", default=0, min=0, max=1024, step=1)),
                "num_overlap": ("INT", _ui("重叠次数", "0 表示使用配置默认值；较大值可能减少切片痕迹但会变慢。", default=0, min=0, max=64, step=1)),
                "chunk_size": ("INT", _ui("切片长度", "0 表示使用配置默认值；Apollo 修复类模型通常建议较大的 chunk_size。", default=0, min=0, max=10000000, step=1)),
                "normalize": (["config", "true", "false"], _ui("响度归一化", "config 使用配置文件设置；true/false 可强制开关输入归一化。", default="config")),
                "debug": ("BOOLEAN", _ui("调试日志", "输出更详细的 MSST 推理日志，排查问题时开启。", default=False)),
            },
        }

    RETURN_TYPES = ("MSST_STEMS", "STRING")
    RETURN_NAMES = ("分离音轨组", "音轨名称")
    OUTPUT_TOOLTIPS = ("包含所有分离结果的音轨字典，连接到 Get Stem 获取指定音轨。", "本次分离实际输出的音轨名称列表。")
    DESCRIPTION = "使用 MSST 模型分离输入音频，输出一个音轨组。MSST 模型的可用音轨通常来自 YAML 配置中的 training.instruments/target_instrument。"
    FUNCTION = "separate"
    CATEGORY = CATEGORY

    def separate(
        self,
        audio,
        msst_model,
        device,
        device_ids,
        use_tta,
        cache_model,
        batch_size=0,
        num_overlap=0,
        chunk_size=0,
        normalize="config",
        debug=False,
    ):
        stems = _separate_msst(msst_model, audio, device, device_ids, use_tta, batch_size, num_overlap, chunk_size, normalize, cache_model, debug)
        return (stems, ", ".join(stems["stem_names"]))


class ComfyMSSTVRSeparate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", _ui("输入音频", "需要分离的混合音频。")),
                "vr_model": ("MSST_VR_MODEL", _ui("VR模型", "由“加载 VR 模型”节点输出的模型对象。")),
                "use_cpu": ("BOOLEAN", _ui("强制CPU", "强制使用 CPU 推理。通常很慢，仅在 GPU 不可用或兼容性问题时使用。", default=False)),
                "batch_size": ("INT", _ui("批大小", "VR 推理批大小，越大越占显存。", default=2, min=1, max=64, step=1)),
                "window_size": ("INT", _ui("窗口大小", "VR 模型的窗口大小，常见值 512；不确定时保持默认。", default=512, min=320, max=1024, step=1)),
                "aggression": ("INT", _ui("提取强度", "主音轨提取强度，范围 -100 到 100；人声/伴奏模型通常使用 5。", default=5, min=-100, max=100, step=1)),
                "enable_tta": ("BOOLEAN", _ui("启用TTA", "测试时增强。可能提升效果，但会增加推理时间。", default=False)),
                "cache_model": ("BOOLEAN", _ui("缓存模型", "保持模型在内存/显存中，重复运行更快；显存紧张时可关闭。", default=True)),
            },
            "optional": {
                "enable_post_process": ("BOOLEAN", _ui("启用后处理", "识别副音轨残留伪影，某些歌曲可能改善分离效果。", default=False)),
                "post_process_threshold": ("FLOAT", _ui("后处理阈值", "后处理强度阈值，常用 0.2。", default=0.2, min=0.1, max=0.3, step=0.01)),
                "high_end_process": ("BOOLEAN", _ui("高频处理", "启用 VR 的 high_end_process 高频处理选项。", default=False)),
                "debug": ("BOOLEAN", _ui("调试日志", "输出更详细的 VR 推理日志，排查问题时开启。", default=False)),
            },
        }

    RETURN_TYPES = ("MSST_STEMS", "STRING")
    RETURN_NAMES = ("分离音轨组", "音轨名称")
    OUTPUT_TOOLTIPS = ("包含所有 VR 分离结果的音轨字典，连接到 Get Stem 获取指定音轨。", "本次分离实际输出的音轨名称列表。")
    DESCRIPTION = "使用 UVR/VR 模型分离输入音频。VR 模型的音轨名来自模型索引中的 primary_stem 与 secondary_stem。"
    FUNCTION = "separate"
    CATEGORY = CATEGORY

    def separate(
        self,
        audio,
        vr_model,
        use_cpu,
        batch_size,
        window_size,
        aggression,
        enable_tta,
        cache_model,
        enable_post_process=False,
        post_process_threshold=0.2,
        high_end_process=False,
        debug=False,
    ):
        stems = _separate_vr(
            vr_model,
            audio,
            use_cpu,
            batch_size,
            window_size,
            aggression,
            enable_tta,
            enable_post_process,
            post_process_threshold,
            high_end_process,
            cache_model,
            debug,
        )
        return (stems, ", ".join(stems["stem_names"]))


class ComfyMSSTGetStem:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stems": ("MSST_STEMS", _ui("音轨组", "MSST/VR 分离节点输出的所有音轨。")),
                "stem_name": (_stem_choices(), _ui("音轨名称", "从下拉列表选择需要取出的音轨；候选来自原 MSST WebUI 配置和 VR 模型索引。", default="vocals")),
            },
            "optional": {
                "custom_stem_name": ("STRING", _ui("自定义音轨名", "填写后优先使用这里的名称，适合第三方模型输出的特殊音轨。", default="")),
                "fallback": (["error", "first"], _ui("找不到时", "error 报错；first 返回第一个可用音轨。", default="error")),
            },
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("音频", "实际音轨名")
    OUTPUT_TOOLTIPS = ("取出的指定音轨音频。", "实际匹配到的音轨名称。")
    DESCRIPTION = "从 MSST/VR 分离结果中取出指定音轨。支持大小写不敏感和部分名称匹配。"
    FUNCTION = "get"
    CATEGORY = CATEGORY

    @classmethod
    def VALIDATE_INPUTS(cls, stem_name, custom_stem_name=""):
        return True

    def get(self, stems, stem_name, custom_stem_name="", fallback="error"):
        if custom_stem_name:
            stem_name = custom_stem_name
        available = stems.get("stems", {})
        normalized = {_normalize_key(name): name for name in available.keys()}
        key = normalized.get(_normalize_key(stem_name))
        if key is None:
            contains = [name for norm, name in normalized.items() if _normalize_key(stem_name) in norm]
            key = contains[0] if contains else None
        if key is None and fallback == "first" and available:
            key = next(iter(available.keys()))
        if key is None:
            raise ValueError(f"Stem '{stem_name}' not found. Available stems: {', '.join(available.keys())}")
        return (available[key], key)


class ComfyMSSTStemByPreset:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "stems": ("MSST_STEMS", _ui("音轨组", "MSST/VR 分离节点输出的所有音轨。")),
                "stem": (STEM_PRESETS, _ui("常用音轨", "常见音轨快捷选择，如 vocals、instrumental、drums、bass、reverb 等。")),
            },
            "optional": {"fallback": (["error", "first"], _ui("找不到时", "error 报错；first 返回第一个可用音轨。", default="error"))},
        }

    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("音频", "实际音轨名")
    OUTPUT_TOOLTIPS = ("取出的指定音轨音频。", "实际匹配到的音轨名称。")
    DESCRIPTION = "从分离结果中按常用预设快速取出音轨。"
    FUNCTION = "get"
    CATEGORY = CATEGORY

    def get(self, stems, stem, fallback="error"):
        return ComfyMSSTGetStem().get(stems, stem, fallback=fallback)


class ComfyMSSTListStems:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"stems": ("MSST_STEMS", _ui("音轨组", "MSST/VR 分离节点输出的所有音轨。"))}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("音轨名称",)
    OUTPUT_TOOLTIPS = ("逗号分隔的音轨名称列表。",)
    DESCRIPTION = "列出分离结果中实际包含的所有音轨名称，方便确认后再用 Get Stem 取出。"
    FUNCTION = "list"
    CATEGORY = CATEGORY

    def list(self, stems):
        return (", ".join(stems.get("stem_names", stems.get("stems", {}).keys())),)


class ComfyMSSTEnsembleAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_1": ("AUDIO", _ui("音频1", "参与合奏/融合的第一段音频。")),
                "audio_2": ("AUDIO", _ui("音频2", "参与合奏/融合的第二段音频。")),
                "mode": (ENSEMBLE_MODES, _ui("合奏模式", "wave 为波形域合成，fft 为频域合成；min_fft 可用于较保守地减少激进模型影响。", default="avg_wave")),
                "weights": ("STRING", _ui("权重", "每路音频的合成权重，用逗号或空格分隔，例如 1,1,0.8。", default="1,1")),
            },
            "optional": {
                "audio_3": ("AUDIO", _ui("音频3", "可选的第三段合奏音频。")),
                "audio_4": ("AUDIO", _ui("音频4", "可选的第四段合奏音频。")),
                "audio_5": ("AUDIO", _ui("音频5", "可选的第五段合奏音频。")),
                "audio_6": ("AUDIO", _ui("音频6", "可选的第六段合奏音频。")),
                "audio_7": ("AUDIO", _ui("音频7", "可选的第七段合奏音频。")),
                "audio_8": ("AUDIO", _ui("音频8", "可选的第八段合奏音频。")),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("合奏音频",)
    OUTPUT_TOOLTIPS = ("融合后的音频。",)
    DESCRIPTION = "将多个模型分离出的同一音轨进行合奏/融合。通常用于把多个模型的优点混合起来。"
    FUNCTION = "ensemble"
    CATEGORY = CATEGORY

    def ensemble(self, audio_1, audio_2, mode, weights, **optional):
        audios = [audio_1, audio_2] + [optional[key] for key in sorted(optional.keys()) if optional.get(key) is not None]
        parsed_weights = []
        for part in str(weights).replace(",", " ").split():
            try:
                parsed_weights.append(float(part))
            except ValueError:
                pass
        if len(parsed_weights) < len(audios):
            parsed_weights.extend([1.0] * (len(audios) - len(parsed_weights)))
        parsed_weights = parsed_weights[: len(audios)]

        arrays = []
        sample_rates = []
        for audio in audios:
            batch, sr = _audio_to_numpy(audio)
            arrays.append(batch[0])
            sample_rates.append(sr)
        target_sr = sample_rates[0]
        arrays = [_resample_channels(array, sr, target_sr) for array, sr in zip(arrays, sample_rates)]
        min_len = min(array.shape[-1] for array in arrays)
        min_channels = min(array.shape[0] for array in arrays)
        arrays = [array[:min_channels, :min_len] for array in arrays]

        with _msst_runtime():
            from utils.ensemble import average_waveforms

            result = average_waveforms(arrays, parsed_weights, mode)
        return (_numpy_to_audio(result, target_sr),)


class ComfyMSSTSubtractAudio:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", _ui("原始音频", "被减去内容的基准音频。")),
                "subtract": ("AUDIO", _ui("要减去的音频", "从原始音频中扣除的音频，例如已分离的人声或伴奏。")),
                "gain": ("FLOAT", _ui("减法增益", "扣除音频的强度。1 为正常相减，负数会变成相加。", default=1.0, min=-4.0, max=4.0, step=0.01)),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("音频",)
    OUTPUT_TOOLTIPS = ("相减后的音频。",)
    DESCRIPTION = "对两段音频做波形相减，可用于从原曲中扣除已提取音轨来得到残差。"
    FUNCTION = "subtract_audio"
    CATEGORY = CATEGORY

    def subtract_audio(self, audio, subtract, gain):
        a, sr_a = _audio_to_numpy(audio)
        b, sr_b = _audio_to_numpy(subtract)
        b0 = _resample_channels(b[0], sr_b, sr_a)
        a0 = a[0]
        channels = min(a0.shape[0], b0.shape[0])
        length = min(a0.shape[-1], b0.shape[-1])
        out = a0[:, :length].copy()
        out[:channels, :length] -= float(gain) * b0[:channels, :length]
        return (_numpy_to_audio(out, sr_a),)


class ComfyMSSTPresetChain:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", _ui("输入音频", "需要按预设链处理的音频。")),
                "preset_json_path": ("STRING", _ui("预设JSON路径", "MSST WebUI 预设文件路径。相对路径会从 MSST WebUI 目录解析。", default="presets/example.json")),
                "device": (["auto", "cuda", "cpu", "mps"], _ui("MSST推理设备", "预设链中 MSST 模型使用的设备。", default="auto")),
                "device_ids": ("STRING", _ui("显卡编号", "CUDA 设备编号，多个编号用逗号分隔，例如 0 或 0,1。", default="0")),
                "force_vr_cpu": ("BOOLEAN", _ui("VR强制CPU", "预设链中的 VR 模型是否强制使用 CPU。", default=False)),
                "use_tta": ("BOOLEAN", _ui("启用TTA", "预设链中的模型是否启用测试时增强。", default=False)),
                "cache_model": ("BOOLEAN", _ui("缓存模型", "缓存预设链中加载的模型，重复运行更快；显存紧张时可关闭。", default=True)),
            },
            "optional": {"debug": ("BOOLEAN", _ui("调试日志", "输出更详细的预设链推理日志。", default=False))},
        }

    RETURN_TYPES = ("MSST_STEMS", "STRING")
    RETURN_NAMES = ("分离音轨组", "音轨名称")
    OUTPUT_TOOLTIPS = ("预设链最终收集到的音轨组。", "预设链输出的音轨名称列表。")
    DESCRIPTION = "按 MSST WebUI 预设 JSON 依次运行多个模型。预设 flow 中的 input_to_next 会作为下一步输入，output_to_storage 会被收集到输出音轨组。"
    FUNCTION = "run"
    CATEGORY = CATEGORY

    def run(self, audio, preset_json_path, device, device_ids, force_vr_cpu, use_tta, cache_model, debug=False):
        preset_path = _resolve_runtime_path(preset_json_path)
        if not preset_path.is_file():
            raise FileNotFoundError(f"Preset JSON does not exist: {preset_path}")
        with preset_path.open("r", encoding="utf-8") as file:
            preset = json.load(file)
        flow = preset.get("flow") or []
        if not flow:
            raise ValueError("Preset contains no flow steps.")

        current_audio = audio
        collected: Dict[str, Dict[str, Any]] = {}
        for step in flow:
            model_name = step["model_name"]
            model_class = step["model_type"]
            input_to_next = step.get("input_to_next", "")
            output_to_storage = step.get("output_to_storage") or []
            if model_class == "UVR_VR_Models":
                stems = _separate_vr(
                    _catalog_vr_spec(model_name),
                    current_audio,
                    force_vr_cpu,
                    2,
                    512,
                    5,
                    use_tta,
                    False,
                    0.2,
                    False,
                    cache_model,
                    debug,
                )
            else:
                stems = _separate_msst(
                    _catalog_msst_spec(model_name, model_class),
                    current_audio,
                    device,
                    device_ids,
                    use_tta,
                    0,
                    0,
                    0,
                    "config",
                    cache_model,
                    debug,
                )
            for name in output_to_storage:
                try:
                    audio_out, resolved = ComfyMSSTGetStem().get(stems, name, fallback="error")
                    collected[resolved] = audio_out
                except ValueError:
                    pass
            if input_to_next:
                current_audio, _ = ComfyMSSTGetStem().get(stems, input_to_next, fallback="error")
            else:
                current_audio, _ = ComfyMSSTGetStem().get(stems, "", fallback="first")

        if not collected:
            collected["result"] = current_audio
        result = {"stems": collected, "sample_rate": current_audio["sample_rate"], "stem_names": list(collected.keys()), "meta": {"backend": "preset"}}
        return (result, ", ".join(result["stem_names"]))


class ComfyMSSTSomeVocalToMidi:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO", _ui("输入人声", "用于提取 MIDI 的干净人声音频。建议先去噪、去混响，SOME 不输出歌词。")),
                "tempo": ("FLOAT", _ui("BPM速度", "写入输出 MIDI 的速度值；提取结果通常还需要在编辑器中手动对齐节拍。", default=120.0, min=20.0, max=300.0, step=0.1)),
                "model_path": ("STRING", _ui("SOME模型路径", "SOME 权重路径。相对路径会从 models/MSST/SOME_weights 解析。", default="model_steps_64000_simplified.ckpt")),
                "output_prefix": ("STRING", _ui("输出前缀", "生成 MIDI 文件名的前缀。", default="some_midi")),
            },
            "optional": {
                "config_path": ("STRING", _ui("SOME配置路径", "SOME 配置文件路径，通常使用 configs_backup/config_some.yaml。", default="configs_backup/config_some.yaml")),
                "output_dir": ("STRING", _ui("输出目录", "MIDI 保存目录。留空时使用 output/MSST/SOME_MIDI。", default="")),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("MIDI路径", "全部MIDI路径")
    OUTPUT_TOOLTIPS = ("第一条生成的 MIDI 文件路径。", "所有生成的 MIDI 文件路径，每行一个。")
    DESCRIPTION = "使用 MSST WebUI 的 SOME 工具从人声音频提取 MIDI。文档建议输入干净、清晰、无噪声且没有混响的人声。"
    FUNCTION = "convert"
    CATEGORY = CATEGORY

    def convert(self, audio, tempo, model_path, output_prefix, config_path="configs_backup/config_some.yaml", output_dir=""):
        model = _resolve_some_weight_path(model_path)
        config = _resolve_runtime_path(config_path)
        _check_file(str(model), "SOME model")
        _check_file(str(config), "SOME config")

        waveform, sample_rate = _audio_to_numpy(audio)
        destination = Path(output_dir.strip().strip('"')) if output_dir else OUTPUT_ROOT / "SOME_MIDI"
        if not destination.is_absolute():
            destination = OUTPUT_ROOT / destination
        destination.mkdir(parents=True, exist_ok=True)

        midi_paths: List[str] = []
        with _msst_runtime():
            import soundfile as sf
            from tools.SOME.infer import infer

            with tempfile.TemporaryDirectory(prefix="comfy_msst_some_") as tmp:
                tmp_path = Path(tmp)
                for index, batch in enumerate(waveform):
                    name = _safe_name(output_prefix)
                    if waveform.shape[0] > 1:
                        name = f"{name}_{index + 1:03d}"
                    wav_path = tmp_path / f"{name}.wav"
                    sf.write(wav_path, batch.T, sample_rate, subtype="FLOAT")
                    midi_path = infer(str(model), str(config), str(wav_path), str(destination), float(tempo))
                    midi_paths.append(str(Path(midi_path).resolve()))

        return (midi_paths[0] if midi_paths else "", "\n".join(midi_paths))


class ComfyMSSTClearCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("状态",)
    OUTPUT_TOOLTIPS = ("清理模型缓存后的状态信息。",)
    DESCRIPTION = "清除当前扩展缓存的 MSST/VR 模型，并尝试释放 CUDA 显存。"
    FUNCTION = "clear"
    CATEGORY = CATEGORY

    def clear(self):
        for separator in list(_MODEL_CACHE.values()):
            try:
                separator.del_cache()
            except Exception:
                pass
        count = len(_MODEL_CACHE)
        _MODEL_CACHE.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return (f"Cleared {count} cached MSST model(s).",)


NODE_CLASS_MAPPINGS = {
    "ComfyMSSTModelFromCatalog": ComfyMSSTModelFromCatalog,
    "ComfyMSSTModelFromPaths": ComfyMSSTModelFromPaths,
    "ComfyMSSTVRModelFromCatalog": ComfyMSSTVRModelFromCatalog,
    "ComfyMSSTVRModelFromPath": ComfyMSSTVRModelFromPath,
    "ComfyMSSTSeparate": ComfyMSSTSeparate,
    "ComfyMSSTVRSeparate": ComfyMSSTVRSeparate,
    "ComfyMSSTGetStem": ComfyMSSTGetStem,
    "ComfyMSSTStemByPreset": ComfyMSSTStemByPreset,
    "ComfyMSSTListStems": ComfyMSSTListStems,
    "ComfyMSSTEnsembleAudio": ComfyMSSTEnsembleAudio,
    "ComfyMSSTSubtractAudio": ComfyMSSTSubtractAudio,
    "ComfyMSSTPresetChain": ComfyMSSTPresetChain,
    "ComfyMSSTSomeVocalToMidi": ComfyMSSTSomeVocalToMidi,
    "ComfyMSSTClearCache": ComfyMSSTClearCache,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyMSSTModelFromCatalog": "MSST 加载本地模型",
    "ComfyMSSTModelFromPaths": "MSST 手动加载模型",
    "ComfyMSSTVRModelFromCatalog": "MSST 加载本地 VR 模型",
    "ComfyMSSTVRModelFromPath": "MSST 手动加载 VR 模型",
    "ComfyMSSTSeparate": "MSST 分离音频",
    "ComfyMSSTVRSeparate": "MSST VR 分离音频",
    "ComfyMSSTGetStem": "MSST 获取指定音轨",
    "ComfyMSSTStemByPreset": "MSST 获取常用音轨",
    "ComfyMSSTListStems": "MSST 列出音轨",
    "ComfyMSSTEnsembleAudio": "MSST 音频合奏",
    "ComfyMSSTSubtractAudio": "MSST 音频相减",
    "ComfyMSSTPresetChain": "MSST 运行预设链",
    "ComfyMSSTSomeVocalToMidi": "MSST SOME 人声转 MIDI",
    "ComfyMSSTClearCache": "MSST 清理模型缓存",
}
