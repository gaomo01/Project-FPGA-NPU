# SuperPoint → iCraft BuYi (BY) NPU 部署指南

> 日期：2026-07-19 | iCraft v3.33.1 | 目标：BuYi NPU

---

## 1. 交付文件

| 文件 | 路径 | 说明 |
|------|------|------|
| IR | `BY/output/superpoint_test_softmax_BY.json` | 编译后计算图 |
| 权重 | `BY/output/superpoint_test_softmax_BY.raw` | INT8 量化参数 |

## 2. 模型规格

| 属性 | 值 |
|------|-----|
| 输入 | 单通道灰度图 480×640，NHWC `[1, 480, 640, 1]` |
| 输入预处理 | 无（pre_method = nop），直接喂原始像素值 |
| 输出 0 (semi) | Softmax 概率 `[1, 60, 80, 65]` |
| 输出 1 (desc) | 描述子 `[1, 60, 80, 256]` |
| 量化 | INT8 逐通道 KLD 校准 |
| 大小 | IR ~55KB + W ~2.1MB |

## 3. 部署步骤

```bash
# 1. 拷贝到板端
#    BY/output/superpoint_test_softmax_BY.json
#    BY/output/superpoint_test_softmax_BY.raw

# 2. 加载（C++ 伪代码）
Session session;
session.load("superpoint_test_softmax_BY.json", "superpoint_test_softmax_BY.raw");

# 3. 推理（host 侧伪代码）
input = load_gray("image.bmp", 480, 640).reshape(1, 480, 640, 1)  # NHWC
semi, desc = session.run(input)
keypoints = decode_keypoints(semi)   # semi 已含 Softmax
```

## 4. 复现编译

```bash
cd BY/
icraft parse   superpoint_by_compile.toml
icraft optimize superpoint_by_compile.toml
icraft quantize superpoint_by_compile.toml
icraft adapt    superpoint_by_compile.toml
icraft generate superpoint_by_compile.toml
# 输出：output/superpoint_test_softmax_BY.json + .raw
```

## 5. 算子

Conv2d×12 + MaxPool×3 + ReLU×10 + Softmax×1 + Transpose×2

全部由 NPU 硬件执行，Softmax 已内嵌无需后处理。
