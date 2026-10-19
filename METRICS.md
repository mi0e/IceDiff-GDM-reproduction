# GDM 降尺度评测指标

IceDiff 论文（§4.3、Table 2）只说用 Consistency 和 FID 评价降尺度，**没有给公式**：FID 引用 Heusel et al. 2017，Consistency 没有引用。下面是沿引用链查到的原始定义，以及本仓库下一步推理统一采用的口径。

## 1. Consistency（一致性，越低越好）

**出处链**：IceDiff → GDP（Fei et al., CVPR 2023, arXiv 2304.01247）附录 C → SR3（Saharia et al., TPAMI 2022, arXiv 2104.07636）§4.2.1。

- SR3 原文："we compute MSE between the downsampled outputs and the low resolution inputs"，Table 1 单位 ×10⁻⁵。
- GDP 原文："measures MSE between the degraded inputs and the outputs undergoing the same degradation"。4× 超分的退化算子是 4×4 块平均（"block averaging filter ... 4 times"），像素范围 [0,1]。

**公式**：设 y ∈ ℝ^{H×W} 为低分辨率输入，x̂ ∈ ℝ^{sH×sW} 为降尺度输出，s 为放大倍数（本任务 s = 4），A 为 s×s 块平均下采样：

$$
(A\hat{x})_{ij} = \frac{1}{s^2}\sum_{p=0}^{s-1}\sum_{q=0}^{s-1}\hat{x}_{si+p,\;sj+q}
$$

$$
\mathrm{Consistency} = \frac{1}{N}\sum_{n=1}^{N}\frac{1}{HW}\sum_{i,j}\big((A\hat{x}_n)_{ij} - (y_n)_{ij}\big)^2
$$

**本仓库口径**（`GuidedDiffusionModule/sample_gdm.py` 中的 `consistency()`）：
- A 用 `F.avg_pool2d(x̂, 4)`，与上面的定义相同。
- 像素在 **0–255** 上计算。换算到 GDP 的 [0,1] 口径时除以 255² = 65025。
- 报告时同时给出两种尺度，并注明单位。

**注意**：
- 论文 Table 2 的 6.84 没有说明尺度，也没有说明是否只算海洋像素，**不能直接比较**。
- 这个指标只衡量与输入的一致性：任何"每个 4×4 块内均值等于输入"的输出（如最近邻放大）都得 0 分，所以只能当合格线，不能衡量质量。
- 建议额外报告仅海洋像素（`ocean_mask.npy` 4× 最近邻放大后）上的 Consistency，避免陆地常数区稀释误差。

## 2. FID（Fréchet Inception Distance，越低越好）

**出处**：Heusel et al., *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium*, NeurIPS 2017。

**公式**：用 Inception-v3 pool3 层（2048 维）分别提取真实图集合 r 和生成图集合 g 的特征，求均值 μ 和协方差 Σ：

$$
\mathrm{FID} = \lVert \mu_r - \mu_g \rVert_2^2 + \mathrm{Tr}\!\left(\Sigma_r + \Sigma_g - 2\left(\Sigma_r\Sigma_g\right)^{1/2}\right)
$$

**论文未说明、本仓库规定的计算协议**：
- 实现：`pytorch-fid`（标准 Inception 权重），图像缩放到 299×299 由库内部完成。
- 输入：单通道 SIC 0–255 复制成 3 通道，按 256×256 patch 切块后计算（与 GDM 训练分辨率一致）。
- 真实分布 r：6.25 km 没有真值，因此：
  - **有真值评测**（推荐）：输入 y = 25 km 观测做 4×4 平均池化得到的 100 km 图，输出降尺度回 448×304，r = 真实 25 km 测试图的 patch。
  - **论文设定**（25 km → 6.25 km）：r 只能用 25 km 测试图的 patch（分辨率不同），结果只能用于方法间相对比较。
- 同时报告双三次、双线性插值基线的 FID 作为对照；样本数要足够（patch 数 ≥ 2048，否则协方差估计不稳定）。

## 3. 下一步推理的评测清单

| 指标 | 空间 | 对照 |
|---|---|---|
| Consistency（0–255 与 [0,1] 两种尺度；全图与仅海洋） | 低分辨率输入空间 | 双三次、双线性 |
| FID | 256×256 patch | 双三次、双线性 |
| RMSE / SSIM / 冰缘误差（仅有真值评测） | 输出空间，与真实 25 km 比较 | 双三次、双线性 |
