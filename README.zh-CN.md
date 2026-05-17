# ComfyUI-MSST 中文说明

ComfyUI-MSST 将 MSST-WEBUI 的推理能力封装成 ComfyUI 自定义节点，可以在 ComfyUI 的音频工作流中直接使用 MSST、UVR/VR、合奏和 SOME 人声转 MIDI 功能。

本扩展使用 ComfyUI 原生 `AUDIO` 输入/输出，适合做人声/伴奏分离、多音轨分离、降噪、去混响、去延迟、气声分离、音质修复和 MIDI 提取等任务。

![ComfyUI-MSST 工作流预览](imgs/msst-1.png)

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

## MSST-WebUI官方模型推荐

下面的模型推荐列表来自于MSST官方飞书: `https://my.feishu.cn/wiki/Dy0bwG4XIizBgJkePDucILaMnlf`.可从[huggingface](https://huggingface.co/Sucial/MSST-WebUI)下载。

### MSST模型

| 模型 | 分类 | Stems | 体积 | 备注 | 推荐星级 |
| --- | --- | --- | --- | --- | --- |
| `aufr33-jarredou_DrumSep_model_mdx23c_ep_141_sdr_10.8059.ckpt` | `multi_stem_models` | kick, snare, toms, hh, ride, crash | 417.38 MB | 用来细分提取出来的 drums | ⭐⭐⭐⭐ |
| `model_bandit_plus_dnr_sdr_11.47.chpt` | `multi_stem_models` | speech, music, effects | 141.99 MB | 可以提取带有背景音乐的说话声，以及特效音；提出来的说话声效果一般，但是能去掉特效音 | ⭐⭐⭐⭐ |
| `model_drumsep.th` | `multi_stem_models` | kick, snare, cybals, toms | 159.65 MB | 用来细分提取出来的 drums |  |
| `model_mdx23c_ep_168_sdr_7.0207.ckpt` | `multi_stem_models` | vocals, bass, drums, other | 427.36 MB | 提取多轨的 MDX23C |  |
| `scnet_checkpoint_musdb18.ckpt` | `multi_stem_models` | drums, bass, other, vocals | 40.47 MB |  |  |
| `HTDemucs4.th` | `multi_stem_models` | drums, bass, other, vocals | 80.24 MB | 提取多轨可以使用 | ⭐⭐⭐⭐ |
| `HTDemucs4_6stems.th` | `multi_stem_models` | drums, bass, other, vocals, guitar, piano | 52.45 MB | 提取多轨可以使用，拆的最多，一共 6 轨，最推荐 | ⭐⭐⭐⭐⭐ |
| `model_scnet_sdr_9.3244.ckpt` | `multi_stem_models` | drums, bass, other, vocals | 161.05 MB |  |  |
| `bs_roformer_4stems_ft.ckpt` | `multi_stem_models` | drums, bass, other, vocals | 502.82 MB | 微调后的 4 轨拆分 bs_roformer 模型 |  |
| `HTDemucs4_FT_bass.th` | `single_stem_models` | drums, bass, other, vocals | 80.24 MB | 输出结果中 bass 质量最高 |  |
| `HTDemucs4_FT_drums.th` | `single_stem_models` | drums, bass, other, vocals | 80.24 MB | 输出结果中 drums 质量最高 |  |
| `HTDemucs4_FT_other.th` | `single_stem_models` | drums, bass, other, vocals | 80.24 MB | 输出结果中 other 质量最高 |  |
| `HTDemucs4_FT_vocals_official.th` | `single_stem_models` | drums, bass, other, vocals | 80.24 MB | 输出结果中 vocals 质量最高 |  |
| `mel_band_roformer_crowd_aufr33_viperx_sdr_8.7144.ckpt` | `single_stem_models` | crowd, other | 870.8 MB | 此处的 crowd 指嘈杂声 |  |
| `model_bs_roformer_ep_937_sdr_10.5309.ckpt` | `single_stem_models` | other, vocals | 374.86 MB | 一般，但是比 UVR 好 | ⭐⭐⭐ |
| `deverb_mel_band_roformer_ep_27_sdr_10.4567.ckpt` | `single_stem_models` | noreverb, reverb | 466.9 MB | 去混响效果不如下面的 deverb_bs_roformer | ⭐⭐⭐⭐ |
| `deverb_bs_roformer_8_256dim_8depth.ckpt` | `single_stem_models` | noreverb, reverb | 162.86 MB | 去混响首选；也能去除部分和声；但是无法去延迟 | ⭐⭐⭐⭐⭐ |
| `deverb_bs_roformer_8_384dim_10depth.ckpt` | `single_stem_models` | noreverb, reverb | 344.75 MB | 比上面那个激进一点点，不过基本听不出差别 | ⭐⭐⭐⭐⭐ |
| `denoise_mel_band_roformer_aufr33_sdr_27.9959.ckpt` | `single_stem_models` | dry, other | 870.8 MB | 最新的降噪模型，推荐 | ⭐⭐⭐⭐⭐ |
| `denoise_mel_band_roformer_aufr33_aggr_sdr_27.9768.ckpt` | `single_stem_models` | dry, other | 870.8 MB | 比上面的降噪激进一点 | ⭐⭐⭐⭐⭐ |
| `dereverb_mdx23c_sdr_6.9096.ckpt` | `single_stem_models` | dry, other | 427.34 MB | 去混响，比 bs_roformer 那两个少激进得多 |  |
| `dereverb_mel_band_roformer_anvuew_sdr_19.1729.ckpt` | `single_stem_models` | noreverb, reverb | 870.81 MB | 分离混响推荐；不支持分离单声道混响 | ⭐⭐⭐⭐⭐ |
| `dereverb_mel_band_roformer_less_aggressive_anvuew_sdr_18.8050.ckpt` | `single_stem_models` | noreverb, reverb | 870.81 MB | 比上面那个少激进一点；不支持分离单声道混响 | ⭐⭐⭐⭐⭐ |
| `Apollo_LQ_MP3_restoration.ckpt` | `single_stem_models` | restored, addition | 63.46 MB | 用于修复 mp3 格式音频的音质，修复至 44.1kHz | ⭐⭐⭐⭐ |
| `aspiration_mel_band_roformer_sdr_18.9845.ckpt` | `single_stem_models` | aspiration, other | 797.26 MB | 气声分离模型，是气声不是呼吸声 | ⭐⭐⭐⭐ |
| `aspiration_mel_band_roformer_less_aggr_sdr_18.1201.ckpt` | `single_stem_models` | aspiration, other | 797.26 MB | 气声分离模型，比上面一个少激进，但效果会略差 | ⭐⭐⭐⭐ |
| `apollo_model_uni.ckpt` | `single_stem_models` | restored, addition | 140.06 MB | 针对人声的 apollo 音频修复模型，但实际效果一般 |  |
| `dereverb_echo_mbr_fused_0.5_v2_0.25_big_0.25_super.ckpt` | `single_stem_models` | dry, other | 434.66 MB | 同时去除混响和延迟的模型；效果略差于 anvuew 模型；不支持分离单声道混响 | ⭐⭐⭐⭐ |
| `de_big_reverb_mbr_ep_362.ckpt` | `single_stem_models` | dry, other | 434.66 MB | 去除大混响的模型，遇到大混响去除不了时可以尝试；不支持分离单声道混响 |  |
| `model_mdx23c_ep_271_l1_freq_72.2383.ckpt` | `single_stem_models` | similarity, difference | 417.34 MB | 中置声道提取，可以提取歌曲的 mid 和 side，实际用途不大 |  |
| `model_mel_band_roformer_ep_3005_sdr_11.4360.ckpt` | `vocal_models` | vocals, instrumental | 961.13 MB | 可以用来做微调训练的底模 | ⭐⭐⭐⭐ |
| `model_swin_upernet_ep_56_sdr_10.6703.ckpt` | `vocal_models` | vocals, instrumental | 896.36 MB | 第一次使用此模型会从 Hugging Face 下载预训练底模 |  |
| `model_vocals_htdemucs_sdr_8.78.ckpt` | `vocal_models` | vocals, instrumental | 160.34 MB |  |  |
| `model_vocals_mdx23c_sdr_10.17.ckpt` | `vocal_models` | vocals, instrumental | 427.34 MB | MDX23C 老人声提取模型，已经不如 bs_roformer | ⭐⭐⭐⭐ |
| `model_vocals_mel_band_roformer_sdr_8.42.ckpt` | `vocal_models` | vocals, instrumental | 128.67 MB |  |  |
| `model_vocals_segm_models_sdr_9.77.ckpt` | `vocal_models` | vocals, instrumental | 823.67 MB | 第一次使用此模型会从 Hugging Face 下载预训练底模 |  |
| `model_bs_roformer_ep_368_sdr_12.9628.ckpt` | `vocal_models` | vocals, instrumental | 609.7 MB | 分离人声伴奏推荐使用 | ⭐⭐⭐⭐⭐ |
| `model_bs_roformer_ep_317_sdr_12.9755.ckpt` | `vocal_models` | vocals, instrumental | 609.71 MB | 1297 提取出来的音频极高频有点问题，但是 SDR 值高，可以使用 | ⭐⭐⭐⭐⭐ |
| `model_mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt` | `vocal_models` | karaoke, other | 870.8 MB | 分离和声模型；如果是原曲放进去，出来的就是带和声伴奏 | ⭐⭐⭐⭐⭐ |
| `Kim_MelBandRoformer.ckpt` | `vocal_models` | vocals, instrumental | 870.81 MB | Kim 的模型，效果比 1296 差一点，但是速度更快 | ⭐⭐⭐⭐⭐ |
| `big_beta5e.ckpt` | `vocal_models` | vocals, other | 1411.2 MB | 超级大模型，用于提取人声，处理速度和质量都不错，但人声存在少量噪声 | ⭐⭐⭐⭐⭐ |
| `bs_roformer_male_female_by_aufr33_sdr_7.2889.ckpt` | `vocal_models` | male, female | 502.7 MB | 用于分离男女声合唱，只能分合唱，间隔唱不行，效果一般，能用 |  |
| `BS-Roformer_LargeV1.ckpt` | `vocal_models` | vocals, other | 705.97 MB | 大模型，用于提取人声，处理速度和质量都不错，能比得上 bsr1296 | ⭐⭐⭐⭐⭐ |
| `inst_v1e.ckpt` | `vocal_models` | other, vocals | 870.8 MB | 针对提取伴奏训练的模型，出来的伴奏非常接近原版伴奏 | ⭐⭐⭐⭐⭐ |
| `kimmel_unwa_ft.ckpt` | `vocal_models` | vocals, other | 870.8 MB | Kim_MelBandRoformer 的微调模型，速度较快，分离表现不错 | ⭐⭐⭐⭐⭐ |
| `mel_band_roformer_instrumental_becruily.ckpt` | `vocal_models` | other, vocals | 870.81 MB | 针对提取伴奏训练的模型，伴奏 SDR 值和 1296 一样，但是推理速度更快 | ⭐⭐⭐⭐⭐ |
| `mel_band_roformer_vocals_becruily.ckpt` | `vocal_models` | other, vocals | 870.81 MB | 针对提取人声训练的模型，速度和质量都不错，但人声存在少量噪声 | ⭐⭐⭐⭐⭐ |
| `melband_roformer_inst_v2.ckpt` | `vocal_models` | other, vocals | 1501.54 MB | 超级大模型，如果 `inst_v1e` 不满意，可以尝试这个 v2 伴奏提取模型 | ⭐⭐⭐⭐⭐ |
| `melband_roformer_instvox_duality_v2.ckpt` | `vocal_models` | Vocals, Instrumental | 1639.48 MB | 平衡了人声和伴奏提取质量，并且处理速度较快，虽然模型最大 | ⭐⭐⭐⭐⭐ |

### VR模型

| 模型 | Primary Stem | Secondary Stem | 备注 | 推荐星级 |
| --- | --- | --- | --- | --- |
| `1_HP-UVR.pth` | Instrumental | Vocals |  |  |
| `2_HP-UVR.pth` | Instrumental | Vocals |  |  |
| `3_HP-Vocal-UVR.pth` | Vocals | Instrumental |  |  |
| `4_HP-Vocal-UVR.pth` | Vocals | Instrumental | 用于提取人声/伴奏，但是不如上面的 bs_roformer 模型 | ⭐⭐ |
| `5_HP-Karaoke-UVR.pth` | Instrumental | Vocals | 去和声 | ⭐⭐⭐ |
| `6_HP-Karaoke-UVR.pth` | Instrumental | Vocals | 去和声，没有 5HP 激进 | ⭐⭐⭐ |
| `7_HP2-UVR.pth` | Instrumental | Vocals |  |  |
| `8_HP2-UVR.pth` | Instrumental | Vocals |  |  |
| `9_HP2-UVR.pth` | Instrumental | Vocals |  |  |
| `10_SP-UVR-2B-32000-1.pth` | Instrumental | Vocals |  |  |
| `11_SP-UVR-2B-32000-2.pth` | Instrumental | Vocals |  |  |
| `12_SP-UVR-3B-44100.pth` | Instrumental | Vocals |  |  |
| `13_SP-UVR-4B-44100-1.pth` | Instrumental | Vocals |  |  |
| `14_SP-UVR-4B-44100-2.pth` | Instrumental | Vocals |  |  |
| `15_SP-UVR-MID-44100-1.pth` | Instrumental | Vocals |  |  |
| `16_SP-UVR-MID-44100-2.pth` | Instrumental | Vocals |  |  |
| `17_HP-Wind_Inst-UVR.pth` | No Woodwinds | Woodwinds | 可以提取木管乐器 |  |
| `MGM_HIGHEND_v4.pth` | Instrumental | Vocals |  |  |
| `MGM_LOWEND_A_v4.pth` | Instrumental | Vocals |  |  |
| `MGM_LOWEND_B_v4.pth` | Instrumental | Vocals |  |  |
| `MGM_MAIN_v4.pth` | Instrumental | Vocals |  |  |
| `UVR-BVE-4B_SN-44100-1.pth` | Vocals | Instrumental | 去和声，但是 BVE 的 Vocals 和 Instrumental 是相反的，要注意；输出音频频率砍到了 18k | ⭐⭐⭐ |
| `UVR-De-Echo-Aggressive.pth` | No Echo | Echo | 去混响延迟，比 Normal 激进 | ⭐⭐⭐ |
| `UVR-DeEcho-DeReverb.pth` | No Reverb | Reverb | 去混响延迟，大混响用这个 | ⭐⭐⭐ |
| `UVR-De-Echo-Normal.pth` | No Echo | Echo | 去混响延迟，没 Aggressive 激进 | ⭐⭐⭐ |
| `UVR-DeNoise.pth` | Noise | No Noise | 降噪，不如上面的 MSST 模型 | ⭐⭐ |
| `UVR-DeNoise-Lite.pth` | Noise | No Noise | 降噪轻量版，更不行了 | ⭐ |
| `UVR-DeReverb-aufr33-jarredou_4band_v4_ms_fullband.pth` | Dry | Reverb | 新的去混响模型 | ⭐⭐⭐⭐ |
| `Harmonic_Noise_Separation_yxlllc.pth` | No Aspiration | Aspiration | yxlllc 大佬训练的气声分离模型，速度更快，效果更好 | ⭐⭐⭐⭐⭐ |



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

示例工作流：[workflows/Separate_vocals.json](workflows/Separate_vocals.json)

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

ComfyUI 已经包含主要依赖，例如 `torch`、`numpy`、`transformers`、`safetensors`、`pyyaml`、`scipy` 等。本扩展支持 `numpy>=1.25,<3` 和 `transformers>=4.35,<6`，可兼容 NumPy 1.x/2.x 与 Transformers 4.x/5.x。

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

## 许可证

ComfyUI-MSST 使用 GNU Affero General Public License v3.0 发布，完整条款见 [LICENSE](LICENSE)。
内置的 MSST-WEBUI 运行时位于 `msst_webui`，同样按 AGPL-3.0 分发；如果重新分发修改版本，请遵守相应许可证要求。
