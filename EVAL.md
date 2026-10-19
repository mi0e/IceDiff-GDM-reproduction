# IceDiff-GDM 降尺度评测（单卡本地运行）

有真值评测：把 25 km 观测图（448×304）做 4×4 平均池化，得到 100 km 图（112×76）作为输入 y；用 GDM 引导采样降尺度回 25 km，再与真实 25 km 图比较，同时与最近邻、双线性、双三次插值对比。指标定义见 `METRICS.md`。

## 目录
```
run_eval.py               一键入口
GuidedDiffusionModule/    eval_gdm.py（GPU 采样）、eval_metrics.py（指标）、guided_diffusion/
data/                     sic_val.npz（2014–2015）、sic_test.npz（2016–2023）、ocean_mask.npy
models/                   自己放入 EMA 权重 *.pt（fp16 或 fp32 都可以）
METRICS.md                指标公式与口径
```

在本仓库中使用时，`data/` 需自己准备：把 `prepare_sic.py` 生成的（或 ModelScope `kiiisara/nsidc_1978-2023` 中的）`sic_val.npz`、`sic_test.npz`、`ocean_mask.npy` 放进仓库根目录的 `data/`。

## 1. 环境（RTX 50 系 = Blackwell，必须用 CUDA 12.8 及以上版本的 PyTorch）
```
python -m venv venv          # Python 3.10+；Windows 激活：venv\Scripts\activate，Linux：source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install numpy pillow scipy pytorch-fid
python -c "import torch;print(torch.__version__, torch.cuda.get_device_name(0))"
```
- 若出现 `no kernel image is available` 或 `sm_120 is not compatible`，说明 torch 版本太旧，需重装 cu128 版本。
- 第一次算 FID 时，pytorch-fid 会从 GitHub 下载约 100 MB 的 Inception 权重。

## 2. 放权重
把服务器上的 EMA 权重放进 `models/`，例如：
- `ema_070000.pt`：服务器 `/mnt/workspace/keep/ema_0.9995_070000.pt`
- `ema_100000.pt`：训练结束后 `download/fp16_ema_0.9995_100000.pt`

文件名就是结果中的模型标签。tune 模式只用**按文件名排序的最后一个**权重。

## 3. 运行
```
python run_eval.py tune                     # 第 1 步：在验证集上选引导参数
python run_eval.py final --configs lr:8     # 第 2 步：用选出的参数在测试集上跑 models/ 下所有权重
```
| 模式 | 数据 | 张数 | 设置 | 显存 |
|---|---|---|---|---|
| tune | 验证集，每 61 天取 1 张 | 12 | hr:0.5、lr:2、lr:8、lr:32 | 约 3–5 GB |
| final | 测试集，每 30 天取 1 张 | 98 | 由 `--configs` 指定 | 同上 |

- 先跑 tune。看终端打印的每张耗时，乘以总张数估算 final 要多久。50 系显卡上，1000 步预计每张 1–3 分钟。
- 可以在模式后追加 `eval_gdm.py` 的参数，例如 `--timestep_respacing 250` 把采样压到 250 步（快 4 倍，但偏离论文的 1000 步），或用 `--every 10`、`--limit 20` 调整张数。
- 中断后重跑会跳过已完成的 `<模型>__<设置>.npz`；只想重算指标时用 `python run_eval.py metrics --out eval_tune`。
- 引导设置的写法是 `空间:系数`：`hr` = 输入双三次放大后在 25 km 上比较；`lr` = 输出 4×4 平均池化后在 100 km 上比较。lr 空间每像素的梯度约为 hr 的 1/16，所以 `lr:8` 的强度大致相当于 `hr:0.5`。

## 4. 结果（把整个 `eval_tune/` 或 `eval_final/` 发回来，但 `*.npz` 很大的话只发下面三个文件）
- `metrics.md` / `metrics.csv`：各方法的指标
  - RMSE%、MAE%：海洋像素，SIC 百分点
  - SSIM
  - IIEE：15% 冰缘积分误差，单位 10³ km²
  - Cons255：0–255 尺度的 Consistency
  - Cons255_ocean：只算全海洋 4×4 块（推荐看这个，全图版本会被海岸陆地置零放大）
  - Cons01：[0,1] 尺度，即 GDP 口径
  - FID：256×256 patch，patch 数少时数值偏高，只作相对比较
- `compare.png`：几天的对比图，列依次为 LR（最近邻放大）、插值基线、模型输出、真值。

判读：模型的 RMSE/SSIM 不一定能赢双三次（生成模型会补细节，像素级误差往往更大），关键看 FID 和 IIEE 是否更好、Cons255_ocean 是否接近 0，以及 compare.png 里冰缘和水道是否更真实。
