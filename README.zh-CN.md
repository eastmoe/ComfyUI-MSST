# ComfyUI-MSST 中文说明

ComfyUI-MSST 将 MSST-WEBUI 的推理能力封装成 ComfyUI 自定义节点，可以在 ComfyUI 的音频工作流中直接使用 MSST、UVR/VR、合奏和 SOME 人声转 MIDI 功能。

本扩展使用 ComfyUI 原生 `AUDIO` 输入/输出，适合做人声/伴奏分离、多音轨分离、降噪、去混响、去延迟、气声分离、音质修复和 MIDI 提取等任务。

## 目录结构

扩展目录：

```text
ComfyUI/custom_nodes/ComfyUI-MSST
```

MSST-WEBUI 推理运行时已经内置在：

```text
ComfyUI/custom_nodes/ComfyUI-MSST/msst_webui
```

默认不需要再单独安装或引用原 MSST-WEBUI 项目。如果你想指定外部 MSST-WEBUI 目录，可以设置环境变量：

```powershell
$env:COMFY_MSST_WEBUI_PATH="D:\path\to\MSST-WebUI"
```

## 模型放置

模型权重请放在 ComfyUI 的模型目录下：

```text
ComfyUI/models/MSST/pretrain
```

建议目录如下：

```text
ComfyUI/models/MSST/pretrain/vocal_models
ComfyUI/models/MSST/pretrain/multi_stem_models
ComfyUI/models/MSST/pretrain/single_stem_models
ComfyUI/models/MSST/pretrain/VR_Models
ComfyUI/models/MSST/SOME_weights
```

说明：

- `vocal_models`：人声/伴奏相关 MSST 模型。
- `multi_stem_models`：多音轨分离模型，例如 drums、bass、other、vocals。
- `single_stem_models`：单目标或两轨任务，例如降噪、去混响、音质修复。
- `VR_Models`：UVR/VR 的 `.pth` 模型。
- `SOME_weights`：SOME 人声转 MIDI 权重。

加载本地模型节点会自动扫描 `models/MSST/pretrain`，并在下拉列表中显示已存在的模型。MSST 模型会按模型文件名自动匹配：

```text
msst_webui/configs/<模型类别>/<模型文件名>.yaml
msst_webui/configs_backup/<模型类别>/<模型文件名>.yaml
```

只有找不到对应配置文件时才会报错。

可选环境变量：

```powershell
$env:COMFY_MSST_MODEL_ROOT="D:\ComfyUI\models\MSST\pretrain"
$env:COMFY_MSST_SOME_WEIGHT_ROOT="D:\ComfyUI\models\MSST\SOME_weights"
$env:COMFY_MSST_OUTPUT_ROOT="D:\ComfyUI\output\MSST"
```

## 节点说明

节点位于 ComfyUI 右键菜单：

```text
音频/Comfy-MSST
```

### 模型加载

- `MSST 加载本地模型`
  - 自动扫描 `models/MSST/pretrain` 下的 MSST 模型。
  - 选择模型后自动匹配 YAML 配置。

- `MSST 手动加载模型`
  - 手动填写模型架构、模型路径和配置路径。
  - 适合第三方模型或自训练模型。

- `MSST 加载本地 VR 模型`
  - 自动扫描 `models/MSST/pretrain/VR_Models`。

- `MSST 手动加载 VR 模型`
  - 手动填写 VR/UVR 模型路径。

### 音频分离

- `MSST 分离音频`
  - 使用 MSST 模型分离音频。
  - 常用于人声/伴奏、多音轨、降噪、去混响、Apollo 修复等。

- `MSST VR 分离音频`
  - 使用 UVR/VR 模型分离音频。
  - 常用于两轨任务，例如 Vocals/Instrumental、Dry/Reverb、Noise/No Noise。

### 音轨处理

- `MSST 获取指定音轨`
  - 从分离结果中取出指定 stem。
  - `stem_name` 提供中文注释的下拉候选，来源包括 MSST 配置和 VR 模型索引。
  - `custom_stem_name` 可手动输入特殊音轨名。

- `MSST 获取常用音轨`
  - 快速选择常见音轨，例如 `vocals`、`instrumental`、`drums`、`bass`、`reverb` 等。

- `MSST 列出音轨`
  - 输出当前分离结果包含的全部音轨名称。

- `MSST 音频合奏`
  - 将多个模型分离出的同一音轨融合。
  - 支持 `avg_wave`、`median_wave`、`min_fft` 等 MSST 合奏模式。

- `MSST 音频相减`
  - 从一段音频中减去另一段音频，可用于扣除已提取音轨。

### 预设和工具

- `MSST 运行预设链`
  - 执行 MSST-WEBUI 的预设 JSON。
  - 支持 `flow`、`input_to_next`、`output_to_storage` 等预设字段。

- `MSST SOME 人声转 MIDI`
  - 使用 SOME 从干净人声音频中提取 MIDI。
  - 建议输入清晰、无噪声、无混响的人声。
  - 输出 MIDI 通常仍需在 DAW 或 MIDI 编辑器中手动修正节拍和音符。

- `MSST 清理模型缓存`
  - 清除扩展缓存的分离模型，并尝试释放 CUDA 显存。

## 常用工作流

### MSST 模型分离

```text
Load Audio
  -> MSST 加载本地模型
  -> MSST 分离音频
  -> MSST 获取指定音轨
  -> Save Audio
```

### VR 模型分离

```text
Load Audio
  -> MSST 加载本地 VR 模型
  -> MSST VR 分离音频
  -> MSST 获取指定音轨
  -> Save Audio
```

### 多模型合奏

```text
同一首歌分别跑多个模型
  -> 各自用 MSST 获取指定音轨
  -> MSST 音频合奏
  -> Save Audio
```

### SOME 人声转 MIDI

```text
Load Audio
  -> 可选：先用 MSST/VR 得到干净人声
  -> MSST SOME 人声转 MIDI
```

## 参数提示

- `device`
  - `auto` 自动选择设备。
  - `cuda` 使用 NVIDIA GPU。
  - `cpu` 很慢，一般不建议用于 MSST 推理。

- `device_ids`
  - CUDA 编号，例如 `0` 或 `0,1`。

- `use_tta`
  - 测试时增强，可能稍微改善效果，但推理会变慢。

- `cache_model`
  - 开启后重复运行更快。
  - 显存紧张时可以关闭，或使用 `MSST 清理模型缓存`。

- `batch_size`
  - 批大小。越大可能越快，但更占显存。
  - MSST 分离节点中 `0` 表示使用配置文件默认值。

- `num_overlap`
  - 切片重叠次数。较大可能减少切片痕迹，但速度更慢。

- `chunk_size`
  - 切片长度。通常保持默认即可。
  - Apollo 音质修复模型建议使用较大的 chunk size。

- `aggression`
  - VR 模型主音轨提取强度，范围 `-100` 到 `100`。
  - 人声/伴奏模型通常保持 `5`。

## 依赖说明

ComfyUI 已经包含主要依赖，例如 `torch`、`numpy`、`transformers`、`safetensors`、`pyyaml`、`scipy` 等。

本扩展只需要安装补充依赖：

```powershell
pip install -r custom_nodes\ComfyUI-MSST\requirements.txt
```

请安装到启动 ComfyUI 的同一个 Python 环境中。

## 常见问题

### 下拉列表没有模型

检查模型是否放在：

```text
ComfyUI/models/MSST/pretrain/<模型类别>
```

例如：

```text
ComfyUI/models/MSST/pretrain/vocal_models/model_bs_roformer_ep_368_sdr_12.9628.ckpt
```

放好模型后重启或刷新 ComfyUI。

### 找不到配置文件

MSST 模型需要同名 YAML 配置，例如：

```text
model.ckpt
model.ckpt.yaml
```

配置会从 `msst_webui/configs` 和 `msst_webui/configs_backup` 查找。

### 显存占用没有释放

使用 `MSST 清理模型缓存`，或关闭分离节点里的 `cache_model`。

### SOME 输出 MIDI 不准

这是 SOME 工具本身的常见情况。建议输入更干净的人声，并在 MIDI 编辑器中手动修正节奏、断音和音符。

## 更多资料

- 英文说明：`README.md`
- 原 MSST-WEBUI 文档备份：`old-doc`
- 原项目运行时：`msst_webui`
