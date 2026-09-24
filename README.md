---
license: apache-2.0
base_model: allenai/Molmo2-ER
language: [en, zh]
tags: [spatial-reasoning, multimodal, classification, pointing]
---
# Jev-Spatial

[Code / 代码](https://github.com/Fr0zenCrane/jev-spatial) · [Weights / 权重](https://huggingface.co/Fr0zencr4nE/jev-spatial)

**Molmo2-ER 提供空间理解能力，Jev 启发从有限选项中快速判断。** Jev-Spatial 在 [Molmo2-ER](https://huggingface.co/allenai/Molmo2-ER) 上使用同一个分类头，处理分类、数值估计和 pointing（选点），直接返回选项、数值或坐标。

**Molmo2-ER provides spatial understanding; Jev inspires fast choices among given options.** Jev-Spatial uses one shared classification head for categorical questions, numerical estimates and pointing, returning option IDs, values or coordinates.

## Motivation / 动机

许多常见空间理解任务更接近 System 1（快思考）：捕捉视觉模式，利用已有的多模态对齐理解语义，然后作出判断。这些任务主要依赖感知，而非大量推理。受 [Jev / System One](https://typesafe.ai/blog/introducing-system-one-models-and-jev) 启发，我们希望直接读出判断结果，减少把视觉模式逐 token 描述成自然语言的开销。

Many everyday spatial understanding tasks resemble System 1 perception: recognize a visual pattern, connect it to language, and decide. Inspired by Jev, we expose these decisions directly instead of generating a verbal description of each pattern.

## Method / 方法

把现成的空间理解 VLM（视觉语言模型）改成 Jev 式快速决策模型，关键是统一不同任务的输入输出：输入图像、问题和对应的候选项，让模型通过分类作出选择。数值和位置通过多轮选择逐步细化。三类任务的约定如下：

To turn an existing vision-language model (VLM) into a Jev-like decision model, we give each task the same interface: images, a question and candidate options. The model selects an option; repeated selections refine numerical estimates and point locations.

`image(s) + question + candidates → Molmo2-ER → shared classifier → option / value / (x, y)`

| Task / 任务 | Classification / 分类方式 | Output / 输出 |
|---|---|---|
| Relations, directions, yes/no / 空间关系、方向、是否 | Choose an option / 选择候选项 | Option ID / 选项 ID |
| Height, size / 高度、尺寸 | Choose a range, then refine it / 先选数值区间，再细分 | Value in meters / 米制数值 |
| Object and free-space pointing / 物体定位、自由空间选点 | Three rounds of 3×3 selection / 三轮九宫格选区（九分法） | Normalized `(x, y)` / 归一化坐标 |

所有任务使用同一个分类头，以交叉熵训练。Pointing 每轮把当前区域分成九宫格并选择一格，连续三轮，输出最后一格的中心坐标；坐标用 0–1 表示在原图中的相对位置。前两轮选定后，会从原图裁出选区、重新编码，帮助下一轮判断，这就是 **crop refill**。评测看最终点是否命中目标物体或有效放置区域。

All tasks share one classification head and a cross-entropy objective. Pointing selects a cell in a 3×3 grid for three rounds, then returns the final cell's center as coordinates from 0 to 1 in the original image. After each of the first two rounds, the selected region is cropped from the original image and re-encoded for the next decision (**crop refill**). Evaluation checks whether the point hits the target object or a valid placement region.

**Molmo2-ER 的图像处理 / Image processing：** Molmo2-ER 会把输入图像切成局部图块，并保留一张全图缩略图，兼顾细节和整体布局。**24-crop** 指最多 24 个局部图块，再加全图缩略图；实际数量取决于图像尺寸和比例。Pointing 裁出的选区图也走这套处理，因此会叠加计算开销。

Molmo2-ER combines local image tiles with a global thumbnail to capture both detail and layout. **24-crop** means up to 24 local tiles plus the thumbnail; the actual count depends on image dimensions and aspect ratio. Pointing crops go through the same preprocessing, adding to the computation.

**Training / 训练：** **72,061 QA** — SAT 24,988 + VST-P 22,073 + RefSpatial 25,000. 使用 8×A800，以 LoRA 微调语言部分并训练分类头，冻结视觉编码器和图文连接层（projector）。 / Fine-tune the language model with LoRA and train the classifier on 8×A800; freeze the vision encoder and the projector that connects visual features to the language model.

## Use / 使用

代码从 GitHub 安装，合并权重从 HF 下载。代码仓库目前仍为私有，取得访问权限后可按下列步骤使用： / Install the code from GitHub and download the merged weights from HF. The code repository is currently private; the steps below require access:

```bash
git clone https://github.com/Fr0zenCrane/jev-spatial
cd jev-spatial
pip install -e '.[inference]'
hf download Fr0zencr4nE/jev-spatial --local-dir models/jev-spatial
jev-spatial --model models/jev-spatial --input examples/requests.jsonl
```

## Results / 结果

我们先用三个独立头分别做分类、预测数值和预测坐标，作为 **three-head naive baseline**，随后改为统一分类头。Pointing 比较了始终使用原图、在注意力计算中屏蔽未选区域、裁出选区重新编码三种方案，当前采用第三种（crop refill）。

We began with a **naive three-head baseline** for classification, numeric regression and coordinate regression, then unified the tasks under one classifier. Pointing experiments compared reusing the original image, masking attention to unselected regions, and re-encoding the selected crop. The current model uses the third approach (crop refill).

| Local evaluation / 本地评测 | Native Molmo2-ER | Naive three-head | Shared head, merged / 统一头 |
|---|---:|---:|---:|
| SAT real ↑ | 79.3 | 77.7 | 75.3 |
| CV-Bench ↑ | 87.3 | 87.0 | 86.3 |
| RefSpatial-Bench ↑ | 52.5 | 9.0 | 54.5 |
| Where2Place ↑ | 57.0 | 26.0 | 64.0 |
| RoboSpatial-Poi † ↑ | 29.5 | 4.1 | 59.8 |
| RoboSpatial-VQ † ↑ | 58.0 | 58.3 | 64.2 |
| VST internal dev MAE, m ↓ | 0.520 | **0.428** | 0.472 |

图像评测使用 24-crop，表中分数为百分比；VST 在 300 条内部开发样本上报告平均绝对误差（MAE，单位米）。Three-head 使用了更少的数据与训练步数。统一头改善了部分选点结果，但分类有回退，**数值估计仍不如 naive baseline**。

Image scores are local 24-crop percentages; VST reports mean absolute error (MAE, in meters) on 300 internal development examples. The three-head baseline used less data and training. The shared head improves some pointing scores, while classification accuracy drops on some tests and **numeric regression still trails the naive baseline**.

† **RoboSpatial 待复核 / unresolved reproduction:** native VQ **58.0** vs paper **73.4**; always-Yes **72.3** on the current labels.

**Latency / 延迟（ms/request ↓）**

每个 benchmark 固定抽 20 条，在同一张 A800 上预热后逐条测量，每条运行 3 次取中位数。计入读图、预处理、推理和结果解析；LoRA 权重已合并，pointing 每题请求一个点。图像任务使用 24-crop，VST 数值任务使用 2-crop。

After warmup, run 20 fixed samples per benchmark one at a time on the same A800, taking the median of 3 runs per sample. Timing includes image loading, preprocessing, inference and parsing. Both LoRA adapters are merged, and pointing requests one point. Image tasks use 24-crop; VST uses 2-crop.

| Benchmark / 统计 | Native Molmo2-ER | Naive three-head | Shared head / 统一头 |
|---|---:|---:|---:|
| SAT real | 504.3 | 461.2 | 484.3 |
| CV-Bench | 202.8 | 161.8 | 158.2 |
| RefSpatial-Bench | 741.4 | 127.9 | 306.1 |
| Where2Place | 1083.2 | 121.8 | 299.9 |
| RoboSpatial-Poi | 986.8 | 464.8 | 772.9 |
| RoboSpatial-VQ | 542.5 | 463.2 | 471.1 |
| VST numeric dev | 300.1 | 111.8 | 187.4 |
| Overall mean / 总体均值 | 623.0 | 273.2 | 382.9 |
| Overall P50 | 544.0 | 127.0 | 301.2 |
| Overall P95 | 1267.2 | 467.0 | 776.0 |

统一头总体平均比原生快 **1.63×**，比 three-head 耗时高约 **40%**。 / The shared head averages **1.63×** faster than native, with **40%** higher latency than three-head.

各 benchmark 行为平均值，总体按 benchmark 等权。保留了 1 条原生解析失败，以及 1 条 Where2Place 原生输出达到 256-token 上限（约 9.5 秒）。复现脚本：`scripts/benchmark_latency.py`。 / Rows show means with equal benchmark weighting. All requests are included: one native parse failure and one native Where2Place response reaching the 256-token cap (~9.5 s).

## Limitations / 局限

- **算力与数据 / Compute and data：** 短训练和有限数据保留了部分基模能力，鲁棒性仍不足。 / Limited training preserves some base-model capability, with robustness still to improve.
- **数值设计 / Numeric design：** 当前区间划分依赖小样本，仍偏 toy；需要根据大规模空间数值数据中的常见范围和极值，重新设计通用的两级类别。 / The current bins come from a small sample. A general two-level scheme needs larger spatial regression datasets, representative value ranges and explicit treatment of extreme values.
- **消融范围 / Ablation scope：** 开发集较小，实验只使用一个随机种子；定位网格较粗，训练主要使用单个标注点，这些都限制了结论和定位精度。 / Small development sets, one random seed, coarse grids and training with one annotated point per example limit the conclusions and pointing precision.
- **合并差异 / Merge differences：** 合并后部分 pointing 输出会变化，24-crop 配置下更明显。 / Some pointing outputs change after merging, especially at 24 crops.

**TODO / 待办**

- [ ] 保留当前设置和至少三轮九分法，检查 24-crop 切块与逐轮选区裁剪的叠加开销及潜在冲突，并评估是否需要更多轮细化。 / Keep the current settings and at least three 3×3 selection rounds. Check the combined cost and possible conflicts between 24-crop preprocessing and per-round cropping, and assess whether more refinement rounds are needed.

## Acknowledgements / 致谢

首先感谢 **Molmo2-ER 与 Jev**，分别提供本项目的能力基础和核心思路。 / We especially thank **Molmo2-ER and Jev** for the model foundation and decision-oriented inspiration.

感谢 / Thanks to [Molmo2](https://github.com/allenai/molmo2), [MolmoAct2](https://github.com/allenai/molmoact2), [Qwen](https://github.com/QwenLM/Qwen3), [SigLIP 2](https://huggingface.co/google/siglip2-so400m-patch14-384); and the community references [jev-visual](https://github.com/hr98w/jev-visual), [Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni), [Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD), [OpenJev](https://github.com/razorback16/openjev), [OmniJev](https://github.com/shapsider/OmniJev), [OpenJev-Vision](https://github.com/IamBusy/OpenJev-Vision), [SemIf](https://github.com/TheoLeeCJ/SemIf).

感谢数据、评测与工具作者 / Thanks to the authors of [SAT](https://huggingface.co/datasets/array/SAT), [VST](https://huggingface.co/datasets/rayruiyang/vst_500k), [RefSpatial/RoboRefer](https://github.com/Zhoues/RoboRefer), [CV-Bench](https://huggingface.co/datasets/nyu-visionx/CV-Bench), [RoboPoint/Where2Place](https://github.com/wentaoyuan/RoboPoint), [RoboSpatial](https://github.com/chanhee-luke/RoboSpatial-Eval), [VSI-Bench](https://github.com/vision-x-nyu/thinking-in-space), the original scene datasets, PyTorch, Transformers, PEFT and Safetensors.

**License / 许可：Apache-2.0.**
