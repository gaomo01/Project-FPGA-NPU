# SuperPoint ONNX → iCraft 多平台 NPU 编译项目

> 日期：2026-07-19 | iCraft v3.33.1 | ONNX opset 12

---

## 目录结构

```
Project-YiyuanZhang/
├── superpoint_dummy.pt                 ← 随机权重（仅用于流程验证）
├── superpoint_test_softmax.onnx        ← ONNX 模型（含 Softmax, 480×640）
├── export_superpoint_onnx.py           ← PyTorch → ONNX 导出脚本
├── inspect_superpoint_onnx.py          ← ONNX 结构体检脚本
├── SUPERPOINT_PROJECT_SUMMARY.md       ← 本文件
├── qtset/                              ← 共享 INT8 校准数据（8 张灰度 BMP）
│
├── BY/                                 ← BuYi 平台
│   ├── superpoint_by_compile.toml      ← 编译配置
│   ├── BY_DEPLOY_GUIDE.md             ← 部署指南
│   └── output/
│       ├── superpoint_test_softmax_BY.json   ← 最终 IR
│       └── superpoint_test_softmax_BY.raw    ← 最终权重
│
└── ZG/                                 ← ZhuGe 平台
    ├── superpoint_zg_compile.toml      ← 编译配置
    ├── ZG_DEPLOY_GUIDE.md             ← 部署指南
    └── output/
        ├── superpoint_zg_ZG.json       ← 最终 IR
        └── superpoint_zg_ZG.raw        ← 最终权重
```

---

## 平台对比

| | BuYi (BY) | ZhuGe (ZG) |
|------|-----------|------------|
| iCraft target | `buyi` | `zhuge` |
| `inputs_dtype` | 不需要 | `["FP32"]` 必填 |
| 最终输出大小 | ~2.1 MB | ~5.0 MB |
| 部署文档 | `BY/BY_DEPLOY_GUIDE.md` | `ZG/ZG_DEPLOY_GUIDE.md` |

---

## 模型规格

| 属性 | 值 |
|------|-----|
| 输入 | 单通道灰度 480×640，NHWC `[1, 480, 640, 1]` |
| 输出 semi | Softmax 概率 `[1, 60, 80, 65]` |
| 输出 desc | 描述子 `[1, 60, 80, 256]` |
| 量化 | INT8 逐通道 KLD 校准 |
| Softmax | ✅ 内嵌，NPU 硬件执行 |

---

## 网络架构

| 层 | 操作 | 尺寸 | 通道 |
|----|------|------|------|
| 1-2 | Conv2d 3×3 + ReLU | 480×640 | 1→64→64 |
| 3 | MaxPool /2 | →240×320 | |
| 4-5 | Conv2d 3×3 + ReLU | 240×320 | 64→64→64 |
| 6 | MaxPool /2 | →120×160 | |
| 7-8 | Conv2d 3×3 + ReLU | 120×160 | 64→128→128 |
| 9 | MaxPool /2 | →60×80 | |
| 10-11 | Conv2d 3×3 + ReLU | 60×80 | 128→128→128 |

双输出头：

| 分支 | 结构 | 输出通道 |
|------|------|----------|
| Semi (关键点) | Conv2d 3×3 → ReLU → Conv2d 1×1 → **Softmax** | 65 |
| Desc (描述子) | Conv2d 3×3 → ReLU → Conv2d 1×1 | 256 |

算子：Conv2d×12 + MaxPool×3 + ReLU×10 + Softmax×1 + Transpose×2

---

## 复现编译

```bash
# 导出 ONNX（如更换权重）
python export_superpoint_onnx.py --weights your.pt -H 480 -W 640 -o superpoint_test_softmax.onnx

# BuYi
cd BY/
icraft parse   superpoint_by_compile.toml
icraft optimize superpoint_by_compile.toml
icraft quantize superpoint_by_compile.toml
icraft adapt    superpoint_by_compile.toml
icraft generate superpoint_by_compile.toml
# → output/superpoint_test_softmax_BY.json + .raw

# ZhuGe
cd ../ZG/
icraft parse   superpoint_zg_compile.toml
icraft optimize superpoint_zg_compile.toml
icraft quantize superpoint_zg_compile.toml
icraft adapt    superpoint_zg_compile.toml
icraft generate superpoint_zg_compile.toml
# → output/superpoint_zg_ZG.json + .raw
```

> 中间产物在 `build/` 目录（已 gitignore），可随时删除。

---

## 问题记录

| # | 平台 | 阶段 | 问题 | 修复 |
|---|------|------|------|------|
| 1 | BY | parse | `inputs_layout only support NHWC` | TOML 改为 NHWC, inputs=[1,480,640,1] |
| 2 | BY | quantize | shape[3] mismatch (RGB vs 灰度) | 校准 BMP 转灰度 |
| 3 | ZG | parse | `zg is not a valid target` | target 改为 `zhuge` |
| 4 | ZG | parse | `input_dtype num error` | 添加 `inputs_dtype = ["FP32"]` |
| 5 | ZG | parse | `FLoat32 is not a valid dtype` | 改为 `FP32`（大写） |
| 6 | ZG | optimize | `Invalid target: ZG` | 统一为小写 `zhuge` |
