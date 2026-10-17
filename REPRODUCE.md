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
| `sic_dataset.py` | 训练数据集：在原始 25 km 日图上随机裁 256×256（`--upscale` 可选先放大），复制为 3 通道，归一化到 [-1,1] |
| `train_gdm.py` | 无条件扩散先验训练（单卡 / `torchrun` 多卡，fp16，EMA，断点续训，验证 loss，预览图） |
| `sample_gdm.py` | 论文 Algorithm 1 引导采样 + Fig. 3 patch 拼接，输出 4× 高分辨率 SIC 与 Consistency |

## 与论文对应关系

**论文明确给出的设置（已照做）**
- 数据划分：训练 1978-10-25 至 2013，验证 2014–2015，测试 2016–2023
- 训练图像 256×256；AdamW β=(0.9, 0.999)；fp16 + loss scaling，fp32 主权重/EMA/优化器状态；EMA 0.9999
- 1000 步扩散，采样与训练使用相同噪声表
- patch 大小 256，stride 128；重叠区域均值平均；每个 patch 独立可优化卷积核
- 目标：25 km（448×304）→ 6.25 km（1792×1216），即 4×

**论文未给出、由本仓库假设的设置（需在报告中注明）**

| 项目 | 取值 | 依据 |
|---|---|---|
| UNet 结构 | ch=256, res_blocks=2, attn=32,16,8, head_ch=64, resblock_updown, learn_sigma | OpenAI 256×256 无条件模型（guided-diffusion 原版配置，GDP 亦基于此） |
| 学习率 / batch | 1e-4 / 8 每卡 | OpenAI 默认 |
| 训练步数 | 未定（`--max_steps 0` 表示不限） | 论文未给出 |
| "resized and cropped" 的方式 | 原始 25 km 图上随机裁 256×256（`--upscale 1`，默认） | 先验只在真实 25 km 数据上学海冰纹理，推理时 zero-shot 用到 6.25 km。早期版本默认先 4× 双三次放大再裁，模型只学到插值纹理（见"AMD 训练结果"），已弃用 |
| 裁剪位置过滤 | 海洋占比 ≥ 25%（`--min_ocean`，设 0 关闭） | 避免大量纯陆地样本 |
| 引导 | 平方误差**求和**，`guidance_scale=0.5`，核初始化为单位核，`kernel_lr=1e-8` | 官方取均值误差 + 3800 + 随机核 /100，实测几乎无引导（见下方"引导修正"） |
| 引导空间 | 默认 `hr`（与官方代码一致）；可选 `lr`（卷积后平均池化，在 25 km 比较） | — |
| Consistency 指标 | 输出 4×4 平均池化后与 LR 输入的 MSE（0–255 尺度）；`sample_gdm.py` 打印的是全图均值（陆地误差为 0，会拉低数值） | **论文与 GDP 均未给公式，本口径为自定，与论文 Table 2 数值不可比**。该口径下最近邻插值得 0 分，只能作"不偏离输入"的合格线，不能衡量细节 |
| FID | 未实现 | 论文未说明 Inception 特征如何作用于 SIC 图 |

## 数据处理（V4 → V6）

同日（2023-12-31）V4 样例与 V6 对比：网格同为 448×304、变量同为 `cdr_seaice_conc`（uint8，×0.01）；
海洋像素平均绝对差 0.0037，冰范围（>15%）相差约 1%。差异点：
- 文件名 `seaice_conc_daily_nh_*_v04r00.nc` → `sic_psn25_*_v06r00.nc`，并提供按年汇总文件
- V4 陆地/海岸/湖泊分别编码 254/253/252，V6 统一为 255；掩膜改用 `G02202-ancillary-psn25-v06r00.nc` 的
  `surface_type`（50 海洋 / 75 湖泊 / 200 海岸 / 250 陆地）
- 预处理将 >100 的值置 0（与官方 FM 预处理一致）
- **剔除海洋缺测 >1% 的日期**（共 112 天，集中在 1984、1986–1988，含 1987-12 至 1988-01 的 F8 卫星中断），
  否则这些整幅缺测图会被当成"全无冰"污染训练

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

# 2. 依赖（AMD 上安装 ROCm 版 PyTorch：pip install torch --index-url https://download.pytorch.org/whl/rocm6.x）
pip install numpy netCDF4 scipy opencv-python-headless pillow

# 3. 生成 npz
cd GuidedDiffusionModule
python prepare_sic.py --raw_dir ../data/raw/aggregate \
  --ancillary ../data/raw/G02202-ancillary-psn25-v06r00.nc --out_dir ../data/gdm

# 4. 训练（单卡；多卡用 torchrun --nproc_per_node=8）
python train_gdm.py --train_npz ../data/gdm/sic_train.npz --val_npz ../data/gdm/sic_val.npz \
  --ocean_mask ../data/gdm/ocean_mask.npy --log_dir runs/gdm --batch_size 16

# 5. 降尺度采样（测试集前 4 天）
# 训练步数少时 EMA 未收敛，改用 runs/gdm/modelXXXXXX.pt
python sample_gdm.py --model_path runs/gdm/ema_0.9999_XXXXXX.pt \
  --lr_npz ../data/gdm/sic_test.npz --ocean_mask ../data/gdm/ocean_mask.npy \
  --num_samples 4 --save_dir runs/gdm_samples
```

ROCm 版 PyTorch 通过 `torch.cuda` 接口暴露 AMD GPU，代码无需改动；`nccl` 后端在 ROCm 上即 RCCL。

## 引导修正

按官方 `downscale.py` 的写法（整个 patch 的**平均** MSE × 3800，卷积核 `rand/100`），每像素梯度约 1e-4，
D(x) 只有 x 的约 4%，采样实际上是无条件生成。在训练 2000 步的模型上（测试集单日，25 步采样）对比：

| s | 卷积核 | kernel_lr | Consistency |
|---|---|---|---|
| 官方设置 | 随机 /100 | 0.01 | ≈7800 |
| 0.5 | 单位核 | 1e-5 | 4496 |
| 2 | 单位核 | 1e-5 | 9274 |
| 0.5 | 随机 /100 | 1e-5 | 9562 |
| 0.5 | 单位核 | 1e-7 | 233 |
| 0.5 | 单位核 | 0 | 119 |
| **0.5** | **单位核** | **1e-8** | **107** |

改为平方误差求和后核梯度大幅增大，`kernel_lr` 稍大就会让核漂移、引导失效；采用最后一行为默认值。
每组仅 1 张图。注意：该扫参以 Consistency 最低为目标，而双三次/最近邻插值本身就是该口径下的最优解之一，
所以这组参数偏向"贴近插值"，在原始尺度训练的模型上需重新评估。

## 验证状态

**AMD Radeon（gfx1100，48 GB，ROCm 7.2，PyTorch 2.9.1）实际训练**（2026-10-17）：
- 配置：`num_channels=128`、`attention_resolutions=32,16,8`，约 1.38 亿参数（论文同款 256 通道约 5.5 亿，在该卡上
  18 小时内无法充分训练）；batch 16，fp16，约 1.09 s/step；**训练数据为 4× 双三次放大后裁剪（旧默认）**
- 共 16788 步，验证 loss ≈ 0.0007；fp16 下 model/EMA/optimizer 断点加载已在服务器上用真实 checkpoint 验证
- EMA 0.9999 在 1.7 万步时仍有约 19% 来自随机初始化，采样只产生噪声，改用原始权重；短训练建议 `--ema_rate 0.999`

**AMD 训练结果：降尺度未产生真实细节**（测试集 4 天，25 步采样，仅统计海洋像素）：

| 指标 | 扩散模型 | 双三次插值 | 最近邻 |
|---|---|---|---|
| Consistency（本文口径） | 42.3 | 42.3 | 0.0 |
| 与双三次结果平均差异 | 0.30 / 255 | — | — |
| 高频能量（4×4 块内残差均值） | 2.05 | 2.07 | 0.00 |
| 中间密集度像素占比（10–90%） | 5.0% | 5.1% | — |

输出与双三次插值几乎相同。原因：(1) 训练数据是双三次放大图，先验只学到插值纹理；(2) 引导以双三次放大的
LR 为目标且参数按 Consistency 选取，把结果拉向插值；(3) 训练量远小于论文。下次训练应使用原始尺度（现默认）、
`--guidance_space lr`，并以有真值的评测（25 km 降到 100 km 再 4× 放大，与 25 km 真值比 RMSE/SSIM，并对比双三次）
代替单看 Consistency。

**CPU 冒烟测试**：
在 CPU（4 核）上用缩小模型（5.9M 参数）完成冒烟测试：
- `prepare_sic.py`：全量数据处理完成，46 个文件 MD5 校验通过
- `train_gdm.py`：loss 正常下降，checkpoint 保存、断点续训、验证 loss、预览图均正常
- `sample_gdm.py`：4× 输出 1792×1216，`hr`/`lr` 两种引导空间与 `--start_t` 模式均可运行

**尚未完成**：原始尺度下的训练、有真值的降尺度评测、FID。
冒烟测试模型仅训练 15 步，Consistency 数值无意义。

## 未复现部分
- Stage-1 IceDiff-FM 未改动；`sample_gdm.py` 默认以真实 25 km 观测作为低分辨率输入，
  若要用 FM 预测结果作输入，将其保存为 `arr_0` [N,448,304]（uint8 0–255 或 float 0–1）即可。
- 对比方法（DGP、GDP）与 FID 未实现。
