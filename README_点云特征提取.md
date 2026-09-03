# GeoPT 逐点特征提取

用 GeoPT 预训练模型，把**三维点云坐标 + 0/1 掩码**转换成**掩码为 1 的每个点的 256 维特征**，
导出到输出目录，供后续下游预测头使用。支持可变点数（实测 N 从 37 到 100000）。

---

## 1. 输入格式

每个样本一个 `.npz`，统一放在一个目录下（也可以直接传单个 `.npz` 文件路径）。

| 数组 | 形状 | 类型 | 必需 | 说明 |
| --- | --- | --- | --- | --- |
| `points` | `(N, 3)` | float | **是** | 点云坐标 |
| `mask` | `(N,)` | 0/1 | 否 | 要导出特征的点，缺省视为全 1 |
| `normals` | `(N, 3)` | float | 否 | 表面法线，优先于 `--normal-source` |
| `sdf` | `(N,)` | float | 否 | 有向距离，优先于 `--sdf-value` |

不同样本的 `N` 可以不同。

```python
import numpy as np
np.savez('my_clouds/sample_0.npz', points=points, mask=mask)
```

> 预训练权重固定消费 14 个通道（坐标 3 + 坐标 3 + SDF 1 + 法线 3 + 动力学方向 3 + 幅值 1），
> **只给坐标和掩码是喂不进去的**。缺失的法线由 `--normal-source` 估计，动力学 prompt 由
> `--dynamics-direction` / `--dynamics-magnitude` 合成，这些默认值**会影响特征数值**，
> 若你的任务有明确流向 / 撞击方向请显式指定。

### 坐标系约定

GeoPT 在归一化域中预训练，约定 **轴 0 = 长度方向（跨度 5）、轴 1 = 竖直方向（最小值 0）**，
默认会自动做这个归一化。

- 数据是 **Z 轴朝上** → 加 `--axis-order xzy`
- 坐标已在 GeoPT 归一化域 → 加 `--no-normalize-geometry`

---

## 2. 掩码语义

默认 **`mask == 0` 的点被视为无效点**，在归一化统计、法线估计和编码阶段整体丢弃。
批处理时不同长度的样本会自动补齐并屏蔽，实测「补齐+掩码」与「物理删除无效点」数值等价。

若这些点其实是有效几何（例如完整场景中只取某个物体的特征），加 `--encode-all-points`：
所有点都参与几何编码，但仍只导出 `mask == 1` 的点的特征。

---

## 3. 运行命令和参数

```bash
cd GeoPT
python extract_features.py --input ./my_clouds --output ./results/geopt_features
```

Z 轴朝上、指定动力学方向：

```bash
python extract_features.py \
  --input ./my_clouds \
  --output ./results/geopt_features \
  --axis-order xzy \
  --dynamics-direction 0,0,1 \
  --batch-size 4 \
  --gpu 0
```

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--input` | 必填 | `.npz` 所在目录，或单个 `.npz` 文件 |
| `--output` | 必填 | 特征输出目录 |
| `--checkpoint` | `checkpoints/GeoPT_8layers.pt` | 预训练权重路径（相对 GeoPT 根目录） |
| `--gpu` | `0` | `CUDA_VISIBLE_DEVICES` |
| `--device` | `cuda` | `cuda` / `cpu` |
| `--dtype` | `float32` | `float32` / `float16` |
| `--batch-size` | `1` | 每次前向的样本数，变长自动补齐+掩码 |
| `--no-normalize-geometry` | 关 | 不做几何归一化，直接用原始坐标 |
| `--axis-order` | `xyz` | 归一化前的轴置换，Z 朝上用 `xzy` |
| `--target-length` | `5.0` | 归一化后轴 0 的跨度 |
| `--normal-source` | `estimate` | 无法线时的填充方式：`estimate` / `zeros` |
| `--normal-neighbors` | `16` | 法线估计的 kNN 邻居数 |
| `--sdf-value` | `0.0` | SDF 通道取值，表面点用 0 |
| `--dynamics-direction` | `1,0,0` | 动力学 prompt 方向，逗号分隔 |
| `--dynamics-magnitude` | `0.3` | 动力学 prompt 幅值 |
| `--encode-all-points` | 关 | 让 `mask == 0` 的点参与几何编码（仍不导出） |
| `--overwrite` | 关 | 重算已有输出的样本，否则跳过 |

`--normal-source estimate` 用局部 kNN 的 PCA 最小特征向量作法线，并统一朝远离质心方向定向
（PCA 无法定符号）。有真实法线时直接放进 `.npz` 的 `normals`，质量更好。

也可以直接用 Python API：

```python
from geopt_feature import ExtractConfig, GeoPTFeatureExtractor

extractor = GeoPTFeatureExtractor(ExtractConfig(axis_order='xzy'))
features, prepared = extractor.extract(points, mask)   # (mask.sum(), 256)
```

---

## 4. 输出格式

输出目录下每个样本一个 `<样本名>_features.npz`，外加一个 `manifest.json`
（记录本次导出的完整配置和每样本点数，便于追溯）。

| 数组 | 形状 | 类型 | 说明 |
| --- | --- | --- | --- |
| `features` | `(M, 256)` | float32 | **逐点特征**，`M = mask.sum()` |
| `point_index` | `(M,)` | int64 | 每行特征对应**输入** `points` 里的行号 |
| `points` | `(M, 3)` | float32 | 这些点的**原始**坐标（未归一化） |
| `geometry_scale` | 标量 | float64 | 几何归一化缩放系数 |
| `geometry_offset` | `(3,)` | float64 | 几何归一化平移量 |

`features` 取自最后一个 Transolver block 在输出头 `mlp2` **之前**的逐点隐层表示
（`n_hidden = 256`），行序与 `point_index` 升序一致。
归一化关系为 `归一化坐标 = 原始坐标 * geometry_scale + geometry_offset`（轴置换之后）。

```python
import numpy as np

with np.load('results/geopt_features/sample_0_features.npz') as d:
    feat, idx = d['features'], d['point_index']

# 还原成与输入等长的稠密数组（未选中的点填 0）
dense = np.zeros((N, 256), dtype=np.float32)
dense[idx] = feat
```

---

## 5. 环境配置

```bash
pip install torch numpy scipy einops
```

`timm` **不需要**：`models/Transolver.py` 的 `trunc_normal_` 导入会自动回退到
`torch.nn.init.trunc_normal_`。`scipy` 用于法线估计（`scipy.spatial.cKDTree`）。

权重 `checkpoints/GeoPT_8layers.pt`（14.8 MiB）需存在，加载时**严格校验** 169 个张量，
不匹配会直接报错而不是静默用随机初始化的层产出特征。若仓库中缺失，可从官方渠道获取：

- 项目主页 <https://physics-scaling.github.io/GeoPT/>
- 代码仓库 <https://github.com/Physics-Scaling/GeoPT>
- HuggingFace <https://huggingface.co/GeoPT>
