# IceDiff-GDM 复现说明

本仓库基于官方代码 [EtronTech/IceDiff](https://github.com/EtronTech/IceDiff)（commit `4187f43`），
目标是复现论文 *IceDiff: High Resolution and High-Quality Arctic Sea Ice Forecasting with Generative
Diffusion Prior*（CVPR 2025，arXiv:2410.09111）中的 **Stage-2：IceDiff-GDM（扩散降尺度）**。
原仓库未声明许可证，代码版权归原作者所有，本仓库仅用于学术复现。

## 为什么需要修改

官方仓库的 GDM 部分无法直接复现：

| 问题 | 说明 |
|---|---|
| 无预训练权重 | `downscale.py` 需要 `--model_path`，但权重未公开（见官方 issue #1，作者未回复） |
| 无训练代码 | 只有推理脚本，缺少 guided-diffusion 的训练循环（`train_util.py` 等） |
| 无数据转换脚本 | README 要求 npz 输入，但没有生成脚本 |
| 推理代码有语法错误 | `gaussian_diffusion_x0.py` 中 `scale=scale` 重复传参，`import` 即报 `SyntaxError` |
| 推理代码逻辑缺陷 | `condition_mean` 未把 `scale` 传给 `cond_fn`（`1/None`）；`corner` 只有 3 个元素却访问 `corner[3]`；`light_mask` 固定 256×256 却按整图坐标索引 |
| 数据集版本下架 | 论文用 NSIDC G02202 **V4**，已退役，现只能下载 **V6** |


## 修改内容

### 修复
- `guided_diffusion/gaussian_diffusion_x0.py`：删除重复的 `scale=scale` 关键字参数（唯一改动原文件之处）。
  原 `downscale.py` 的其余缺陷未逐个修补，改为新写 `sample_gdm.py` 替代。

### 新增（均在 `GuidedDiffusionModule/`）
| 文件 | 作用 |
|---|---|
| `prepare_sic.py` | NSIDC G02202 V6 年度汇总文件 → `sic_{train,val,test}.npz` + `ocean_mask.npy` |
| `sic_dataset.py` | 训练数据集：在原始 25 km 日图上随机裁 256×256，复制为 3 通道，归一化到 [-1,1] |
| `train_gdm.py` | 无条件扩散先验训练（单卡 / `torchrun` 多卡，fp16，EMA，断点续训，预览图） |
| `sample_gdm.py` | 论文 Algorithm 1 引导采样 + Fig. 3 patch 策略，25 km → 6.25 km |
| `eval_metrics.py` | Table 2 指标：Consistency、FID；对比最近邻 / 双线性 / 双三次 |

## 与论文对应关系

**论文明确给出的设置（已照做）**
- 数据划分：训练 1978-10-25 至 2013，验证 2014–2015，测试 2016–2023
- 训练图像 256×256；AdamW β=(0.9, 0.999)；fp16 + loss scaling，fp32 主权重/EMA/优化器状态
- 1000 步扩散，采样与训练使用相同噪声表
- Algorithm 1：每步由先验估计 x̃₀，按 s·∇L 修正 x̃₀，再由修正后的 x̃₀ 计算后验均值和方差并采样 x_{t-1}，
  同时以学习率 l 更新卷积核
- patch 大小 256，stride 128；每个 patch 独立可优化卷积核；重叠区域平均
- 目标：25 km（448×304）→ 6.25 km（1792×1216），即 4×
- 评测指标：Consistency、FID；对比方法：最近邻、双线性、双三次、DGP、GDP

**论文未给出、由本仓库取定的设置（需在报告中注明）**

| 项目 | 取值 | 依据 |
|---|---|---|
| UNet 结构 | ch=128, res_blocks=2, attn=32,16,8, head_ch=64, resblock_updown, learn_sigma（约 1.38 亿参数） | OpenAI 256×256 无条件模型结构，通道数受算力限制由 256 减为 128 |
| 学习率 / batch | 1e-4 / 32 | OpenAI 默认学习率 |
| 训练步数 | 104765 步 | 论文未给出，受机时限制 |
| 距离函数 L | x̃₀ 退化后与 y 的均方误差 | 官方 `downscale.py` |
| 引导系数 s | 1000 步 DDPM 取 2.5e5（脚本默认值仍为官方 3800） | 官方 3800 在核归一化后几乎无引导；在验证集上按 Consistency 选取，见"引导系数选取" |
| 卷积核初始化 / 学习率 l | `rand/100` / 0.01 | 官方 `downscale.py` |
| 重叠区融合 | 默认等权平均（论文）；`--blend ramp` 为中心高、边缘低的帐篷窗加权，**非论文设置**，仅用于对比接缝 | — |
| 卷积核约束 | 前向中保持非负并归一化为和 1 | 官方未归一化；不归一化时在全水 / 全冰 patch 上核尺度不受约束，引导退化（见"DDIM 快速预览"） |
| 退化函数 D 中的下采样 | 3×3 卷积后 4×4 平均池化，在 25 km 与 y 比较 | 论文称 D 为 down-scale function；块平均为 GDP 超分退化算子 |
| Consistency | 论文附录：低分辨率图与"经过卷积函数的降尺度图"之间的 MSE。实现：每个 6.25 km patch 用该 patch 采样结束时的卷积核卷积，再 4×4 平均池化，重叠区在 25 km 平均后与 y 比较，0–255 尺度 | 附录未说明插值基线用哪个核，本仓库所有方法统一使用 GDM 的最终卷积核（`samples.npz` 中的 `kernels`、`corners`） |
| FID | pytorch-fid（Inception-v3 pool3），输出切 256×256 patch（stride 128），参考集为 25 km 输入切 patch | 论文仅引用 Heusel et al. 2017；6.25 km 无真实参考图 |
| 测试样本数 | 标准时间样点：论文附录示例的 2021-07-08、2021-03-01、2018-04-15（`--dates paper`，`sample_gdm.PAPER_DATES`） | 论文附录 |

## 已知偏离

| 项目 | 论文 / 官方 | 本仓库 | 原因 |
|---|---|---|---|
| 数据集版本 | G02202 V4 | G02202 V6 | V4 已下架 |
| 缺测日期 | 未剔除，>100 的值置 0 | 剔除海洋缺测 >1% 的 112 天（集中在 1984、1986–1988，含 1987-12 至 1988-01 的 F8 卫星中断），其余 >100 的值置 0 | 避免整幅缺测图被当作"全无冰"；仅影响训练集，验证/测试集无剔除 |
| 训练裁剪 | "resized and cropped to 256×256" | 原始 25 km 图上随机裁 256×256，要求海洋占比 ≥25% | 受机器限制的折中 |
| EMA | 0.9999 | 0.9995 | 训练步数有限，0.9999 在短训练下未收敛 |
| 模型规模 | 未给出 | 128 通道 | 算力限制 |
| 引导的施加位置 | 论文 Algorithm 1 修正 x̃₀；官方代码直接把梯度加到后验均值 | 按论文 Algorithm 1 | 官方代码与论文不一致且无法运行，以论文为准 |
| 卷积核 | 论文：每个 patch 独立；官方代码：所有 patch 共享并取平均 | 按论文，每个 patch 独立 | 同上 |
| 降尺度输入 | IceDiff-FM 的 7 天 / 8 周平均 / 6 个月平均预报 | 25 km 观测（测试集） | 只复现 GDM，不做 FM（见"复现范围"） |

## 数据处理（V4 → V6）

同日（2023-12-31）V4 样例与 V6 对比：网格同为 448×304、变量同为 `cdr_seaice_conc`（uint8，×0.01）；
海洋像素平均绝对差 0.0037，冰范围（>15%）相差约 1%。差异点：
- 文件名 `seaice_conc_daily_nh_*_v04r00.nc` → `sic_psn25_*_v06r00.nc`，并提供按年汇总文件
- V4 陆地/海岸/湖泊分别编码 254/253/252，V6 统一为 255；掩膜改用 `G02202-ancillary-psn25-v06r00.nc` 的
  `surface_type`（50 海洋 / 75 湖泊 / 200 海岸 / 250 陆地）
- 预处理将 >100 的值置 0（与官方 FM 预处理一致）；剔除缺测日期见"已知偏离"

处理结果：train 12740 天，val 730 天，test 2922 天，共约 2.1 GB。

## 使用方法

```bash
# 1. 下载数据（约 4 GB，无需登录）
mkdir -p data/raw/aggregate && cd data/raw/aggregate
U=https://noaadata.apps.nsidc.org/NOAA/G02202_V6/north
curl -sL $U/aggregate/ | grep -oE 'sic_psn25_(19[0-9]{2}|20(0[0-9]|1[0-9]|2[0-3]))[0-9]{4}-[0-9]{8}_v06r00\.nc' \
  | sort -u | while read f; do curl -sfLO $U/aggregate/$f; curl -sfLO $U/checksums/aggregate/$f.mnf; done
curl -sfL -o ../G02202-ancillary-psn25-v06r00.nc https://noaadata.apps.nsidc.org/NOAA/G02202_V6/ancillary/G02202-ancillary-psn25-v06r00.nc
cd -

# 2. 依赖（AMD 上安装 ROCm 版 PyTorch；RTX 50 系需 cu128 版）
pip install numpy netCDF4 scipy pillow pytorch-fid

# 3. 生成 npz
cd GuidedDiffusionModule
python prepare_sic.py --raw_dir ../data/raw/aggregate \
  --ancillary ../data/raw/G02202-ancillary-psn25-v06r00.nc --out_dir ../data

# 4. 训练（实际使用的配置见 start_train.sh）
python train_gdm.py --train_npz ../data/sic_train.npz --val_npz ../data/sic_val.npz \
  --ocean_mask ../data/ocean_mask.npy --log_dir runs/gdm --num_channels 128 --batch_size 32 --ema_rate 0.9995

# 5. 降尺度采样（1000 步；--every/--limit 选取测试日期）
python sample_gdm.py --model_path ../models/ema_083903.pt --lr_npz ../data/sic_test.npz \
  --every 30 --save_dir runs/gdm_samples

# 6. Table 2 指标
python eval_metrics.py --samples runs/gdm_samples/samples.npz
```

ROCm 版 PyTorch 通过 `torch.cuda` 接口暴露 AMD GPU，代码无需改动；`nccl` 后端在 ROCm 上即 RCCL。

## 引导系数选取（1000 步 DDPM）

验证集 2014-01-01、2014-07-03 各取冰缘最密集的 64×76 裁块（6.25 km 下 256×304），1000 步 DDPM、l=0.01、
104765 步权重，每组 1 个随机种子：

| s | 3e4 | 1e5 | **2.5e5** | 5e5 | 1e6 | 4e6 |
|---|---|---|---|---|---|---|
| Consistency 冬 / 夏 | 4735 / 7800 | 159 / 435 | **29.3 / 92.3** | 71.9 / 67.0 | 84.6 / 64.4 | 402 / 1106 |
| 均值 | — | 297 | **60.8** | 69.4 | 74.5 | — |

双三次在这两块上为 58.2 / 72.6。按均值最低取 **s=2.5e5**。

## DDIM 快速预览（非论文设置）

1000 步 DDPM 在 RTX 5060 Ti 上约 80 分钟一张，为先看结果加了 `--sampler ddim --timestep_respacing N`：
Algorithm 1 其余不变，只把"由 x̃₀ 和 x_t 计算后验并采样"换成 DDIM 步 q_σ(x_{t-1} | x_t, x̃₀)（`--eta 0` 为确定性）。
50 步约 4 分钟一张。**预览结果不能作为 Table 2 复现结果。**

官方取值（s=3800、核 `rand/100`、l=0.01）在 50 步下几乎没有引导：核从约 0.005 起步，D(x̃₀) 只有 x̃₀ 的几个百分点，
Consistency 约 16000（双三次约 45）。只加大 s、l 也不行：在全水（y 全 0）的 patch 上，把核缩到 0 就能让误差为 0，
引导失效，先验在开阔水域随机画出冰块；在全冰 patch 上核尺度与 x̃₀ 亮度互相补偿，出现灰色方块。
因此核在 D 内保持非负并归一化为和 1（见上表"卷积核约束"）。

s 论文未给出，在验证集（2014-01-01、2014-07-03 两张图的 128×152 裁块，含密集冰、冰缘和开阔水域）上按
Consistency 选取，50 步 DDIM、核归一化后：

| s | l | Consistency（冬 / 夏） |
|---|---|---|
| 2.5e5 | 0.01 | 33.6 / 26.4 |
| 5e5 | 0.01 | 12.3 / 13.9 |
| **1e6** | **0.01** | **6.0 / 12.2** |
| 1e6 | 0 | 7.5 / 13.2 |
| 1e6 | 0.1 | 61.5 / 41.3 |
| 4e6 | 0.01 | 6.4 / 422.6 |
| 1.6e7 | 0.01 | 2237 / 2625 |
| 双三次 | — | 58.8 / 36.1 |

预览取 **s=1e6、l=0.01**（l 与官方一致）。每组只用 1 个随机种子；1000 步 DDPM 正式运行时需在验证集上重新选取 s。

## 验证状态

**训练**：
- 早期在 AMD Radeon（gfx1100，48 GB）上用 4× 双三次放大后裁剪的数据训练 16788 步，先验只学到插值纹理，已弃用。
- PAI-DSW（AMD 约 206 GB 显存）上用原始 25 km 数据训练：128 通道、batch 32、lr 1e-4、EMA 0.9995、约 1.04 s/step，
  共 104765 步（到机时截止），loss 自约 6.5 万步起稳定在 0.0027 左右（mse ≈ 0.0022）。入口 `boot.sh` → `start_train.sh`。

**采样**：Colab A100（`colab` CLI）上 1000 步 DDPM，每张 6.25 km 图约 22.5 分钟；RTX 5060 Ti 约 80 分钟。

## 复现结果

### 标准时间样点（论文附录的 3 个日期，Consistency 按附录定义）

104765 步 EMA 权重，1000 步 DDPM，s=2.5e5，l=0.01，等权平均融合；2021-07-08、2021-03-01、2018-04-15，
A100 每张约 22.6 分钟，共约 6.4 个计算单元。结果在 `runs/paper_mean/`。

| 方法 | Consistency ↓ | FID ↓ |
|---|---|---|
| 最近邻 | 8.43 | 317.81 |
| 双线性 | 73.33 | 311.59 |
| 双三次 | 48.27 | 285.12 |
| **IceDiff-GDM** | **6.57** | **215.60** |

两项指标 IceDiff-GDM 均为最优，排序与论文 Table 2 一致（论文 7 天列：最近邻 20.01 / 78.45，双三次 13.13 / 54.71，
IceDiff 6.84 / 29.34）。Consistency 绝对值与论文同量级；FID 因参考集为 25 km 图、样本仅 3 张，绝对值不可比。
采样结束时每个 patch 的卷积核基本塌缩为平移核（权重集中在 1–2 个位置），见 `samples.npz` 中的 `kernels`。

### 16 张测试图（旧 Consistency 口径）

下表的 Consistency 为旧口径（输出 4×4 平均池化后直接与 y 比较，不经过卷积），当时未保存卷积核，无法按附录定义重算；
FID 不受影响。

104765 步 EMA 权重，1000 步 DDPM，s=2.5e5，l=0.01，25 km → 6.25 km；测试集 2016-01-01 至 2023-07-08 每 183 天取 1 张，
共 16 张（冬夏交替），共约 6 小时、32 个 Colab 计算单元。

| 方法 | Consistency ↓ | FID ↓ |
|---|---|---|
| 最近邻 | 0.00 | 317.19 |
| 双线性 | 57.36 | 310.38 |
| 双三次 | 34.31 | 286.17 |
| **IceDiff-GDM** | **9.96** | **211.46** |

方法排序与论文 Table 2 一致。绝对数值不可与论文直接比较：FID 协议论文未给出（参考集为 25 km 测试图 patch，
与 6.25 km 输出分辨率不同，所有方法 FID 偏高）；输入为观测而非 FM 预报；样本数 16。最近邻按定义 Consistency 为 0。

图：`python plot_results.py --samples runs/ddpm1000/samples.npz`，输出 `compare_gray.png`（单通道）、
`compare_color.png`（论文配色：蓝白 SIC、黄色陆地）、`table.png`。

已知现象：输出在 patch 边界处有接缝（冰区竖直方向梯度在边界处约为其他位置的 3.5 倍；宽度 1216 不能被 128 整除，
最后一列 patch 只重叠 64 像素），来自论文的重叠区等权平均；密集冰区有零星暗点。

## 复现范围
- 只复现 Stage-2 IceDiff-GDM，不复现 Stage-1 IceDiff-FM。降尺度输入为 25 km 观测；
  如需 FM 预报作输入，保存为 `arr_0` [N,448,304]（uint8 0–255 或 float 0–1）传给 `sample_gdm.py --lr_npz` 即可。
- 对比方法 DGP、GDP 未实现，目前只与插值方法对比。
