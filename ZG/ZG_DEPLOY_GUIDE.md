# SuperPoint ONNX → iCraft ZhuGe (ZG) NPU 编译项目

> 日期：2026-07-19  
> iCraft 版本：v3.33.1  
> 目标硬件：ZhuGe (ZG) NPU

---

## 1. 板端部署指南

### 交付文件

| 文件 | 路径 | 用途 |
|------|------|------|
| IR | `ZG/output/superpoint_zg_ZG.json` | 编译后计算图 |
| 权重 | `ZG/output/superpoint_zg_ZG.raw` | INT8 量化参数 |

### 模型规格

| 属性 | 值 |
|------|-----|
| 输入 | 单通道灰度图，480×640，NHWC 格式 `[1, 480, 640, 1]`，dtype=FP32 |
| 输入预处理 | 无（`pre_method = nop`），直接喂原始像素值 |
| 输出 0 (semi) | 关键点响应概率 `[1, 60, 80, 65]`（**已含 Softmax**） |
| 输出 1 (desc) | 特征描述子 `[1, 60, 80, 256]` |
| 量化精度 | INT8（逐通道 per-channel，KLD 饱和度校准） |
| 量化校准集 | 8 张灰度 BMP，`qtset/` |

### 部署步骤

#### 步骤 1：拷贝文件到板端

```
superpoint_zg_ZG.json  →  板端任意目录
superpoint_zg_ZG.raw   →  板端同一目录
```

#### 步骤 2：使用 iCraft Runtime API 加载模型

```cpp
// C++ 伪代码
#include <icraft/xrt/session.hpp>

icraft::xrt::Session session;
session.load("superpoint_zg_ZG.json", "superpoint_zg_ZG.raw");
```

#### 步骤 3：推理

```python
# Python 伪代码（host 侧）
import numpy as np
from PIL import Image

# 加载图片 → 单通道 480×640
img = Image.open("image.bmp").convert('L').resize((640, 480))
input_data = np.array(img, dtype=np.float32).reshape(1, 480, 640, 1)  # NHWC

# 推理
semi, desc = session.run(input_data)

# semi 已经是 Softmax 概率，直接解码关键点
keypoints = decode_keypoints(semi)       # [1, 60, 80, 65] → N×2 坐标
# desc 用于特征匹配
matches = match_descriptors(desc1, desc2) # [1, 60, 80, 256]
```

### 输出解码

- **semi** `[1, 60, 80, 65]`：65 通道 = 8×8 网格 (64) + dustbin (1)，已过 Softmax，直接是概率分布。按 SuperPoint 论文的 `decode_keypoints()` 即可提取关键点坐标。
- **desc** `[1, 60, 80, 256]`：256 维特征描述子（FP32），L2 归一化后用于关键点匹配。

---

## 2. 项目概述

将 SuperPoint 特征点检测网络从 PyTorch 权重导出为 ONNX，通过 iCraft 工具链完成面向 **ZhuGe (ZG)** NPU 板的完整编译流水线。

### 完整流水线

```
PyTorch .pt → ONNX (opset12, Softmax) → parse → optimize → quantize (INT8) → adapt → generate → ZG.json/raw
```

### ZG vs BY 平台差异

| 项目 | BY (BuYi) | ZG (ZhuGe) |
|------|-----------|------------|
| iCraft target | `buyi` | `zhuge` |
| `inputs_dtype` | 不需要 | 必须填 `FP32` |
| 最终输出 | `_BY.json/raw` | `_ZG.json/raw` |
| 输出大小 | ~2.1 MB | ~5.0 MB |

---

## 3. 文件清单

### 🎯 最终产出（板端直接使用）

| 文件 | 说明 | 大小 |
|------|------|------|
| `ZG/plin/superpoint_zg_ZG.json` | 最终编译 IR（ZhuGe 目标） | ~88 KB |
| `ZG/plin/superpoint_zg_ZG.raw` | 最终 INT8 量化权重 | ~5.0 MB |

### 🔁 ZG 编译配置

| 文件 | 说明 |
|------|------|
| `ZG/superpoint_zg_compile.toml` | ZG 平台 iCraft 编译配方 |

### 📦 共享模型源

| 文件 | 路径 | 说明 |
|------|------|------|
| ONNX | `../superpoint_test_softmax.onnx` | 含 Softmax，ZG/BY 共用 |

### 🗑️ ZG 中间产物

| 文件 | 阶段 |
|------|------|
| `ZG/plin/superpoint_zg_parsed.json/raw` | Parse |
| `ZG/plin/superpoint_zg_optimized.json/raw` | Optimize |
| `ZG/plin/superpoint_zg_quantized.json/raw` | Quantize |
| `ZG/plin/superpoint_zg_adapted.json/raw` | Adapt |
| `ZG/logs/*` | 编译日志 |

---

## 4. ZG 编译配置 (TOML)

```toml
[parse]
net_name = "superpoint_zg"
framework = "onnx"
inputs = [ 1, 480, 640, 1]
inputs_layout = "NHWC"
inputs_dtype = ["FP32"]          # ← ZG 平台必须
pre_method = "nop"
pre_scale = [ 1.0]
pre_mean = [ 0.0]
channel_swap = [ 0]
network = "../superpoint_test_softmax.onnx"
jr_path = "plin/"
target = "zhuge"                # ← 小写 zhuge

[optimize]
target = "zhuge"
...

[quantize]
target = "zhuge"
forward_mode = "image"
saturation = "kld"
forward_dir = "../icraft_superpoint/qtset"
forward_list = "../icraft_superpoint/qtset/list.txt"
qdtype = "int8"
per = "channel"
...

[adapt]
target = "zhuge"
...

[generate]
target = "zhuge"
no_mergeops = true
etmopt = 0
...
```

---

## 5. 复现流程（换真实权重后）

```bash
cd ZG/

# 步骤 1：导出 ONNX（在父目录）
cd ..
python export_superpoint_onnx.py --weights real_superpoint.pt --height 480 --width 640 --output superpoint_test_softmax.onnx
cd ZG/

# 步骤 2：全流水线
icraft parse superpoint_zg_compile.toml
icraft optimize superpoint_zg_compile.toml
icraft quantize superpoint_zg_compile.toml
icraft adapt superpoint_zg_compile.toml
icraft generate superpoint_zg_compile.toml

# 输出：plin/superpoint_zg_ZG.json + plin/superpoint_zg_ZG.raw
```

---

## 6. 网络架构

> 与 BY 版本共用同一个 ONNX 模型，架构完全一致。

| 层 | 类型 | 输入 H×W | 通道 |
|----|------|----------|------|
| 1-2 | Conv2d 3×3 + ReLU | 480×640 | 1→64→64 |
| 3 | MaxPool /2 | →240×320 | 64 |
| 4-5 | Conv2d 3×3 + ReLU | 240×320 | 64→64→64 |
| 6 | MaxPool /2 | →120×160 | 64 |
| 7-8 | Conv2d 3×3 + ReLU | 120×160 | 64→128→128 |
| 9 | MaxPool /2 | →60×80 | 128 |
| 10-11 | Conv2d 3×3 + ReLU | 60×80 | 128→128→128 |
| **Semi** | Conv2d 3×3→ReLU→Conv2d 1×1→**Softmax** | 60×80 | 128→256→65 |
| **Desc** | Conv2d 3×3→ReLU→Conv2d 1×1 | 60×80 | 128→256→256 |

### 算子统计 (ONNX)

```
Conv2d     ×12
MaxPool    ×3
ReLU       ×10
Softmax    ×1       ← NPU 硬件执行
Transpose  ×2       ← PyTorch 导出附带
```

---

## 7. 遇到的问题

| # | 阶段 | 问题 | 修复 |
|---|------|------|------|
| 1 | parse | `zg is not a valid target` | 目标名改为 `zhuge` |
| 2 | parse | `input_dtype num error` | 添加 `inputs_dtype = ["FP32"]` |
| 3 | parse | `FLoat32 is not a valid dtype` | 大小写改为 `FP32` |
| 4 | optimize | `Invalid target: ZG` | `[optimize]` target 改为小写 `zhuge` |

---

## 8. 当前状态

| 阶段 | 状态 |
|------|------|
| Parse | ✅ |
| Optimize | ✅ |
| Quantize (INT8) | ✅ |
| Adapt | ✅ |
| Generate | ✅ |
| Softmax | ✅ NPU 硬件算子 |
