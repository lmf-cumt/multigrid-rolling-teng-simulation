# 多栅格交替电极滚球式 TENG 仿真与专利支撑材料

本仓库保存多栅格交替电极滚球式摩擦纳米发电机（TENG）的 COMSOL 几何模型、等效仿真脚本、实验校准后处理结果和专利说明图。

## 当前推荐使用的结果

用于专利说明书的当前冻结版本为四电极完整周期电流图：

- 图像：`results_multigrid_calibrated/n4_full_cycle_current/n4_full_cycle_current_1G.svg`
- 数据：`results_multigrid_calibrated/n4_full_cycle_current/n4_full_cycle_current_1G.csv`
- 说明：`results_multigrid_calibrated/n4_full_cycle_current/四电极完整周期电流图生成说明.md`
- 复现脚本：`scripts/produce_n4_full_cycle_current.py`

该版本不依赖仍在迭代的主仿真脚本，已冻结关键参数：

```text
四电极结构：A-B-A-B
有效相邻交替边界数：3
PTFE 球电荷密度：固定
四电极实测转移电荷：650 nC
运动距离：100 mm
加速度/减速度：5 m/s²
起点停留：30 ms
终点停留：30 ms
负载电阻：1 GΩ
等效内电容：112.2 pF
```

## 核心物理口径

仿真和专利表述采用如下解释：

在相同 PTFE 球材料、滚球数量和摩擦起电条件下，PTFE 球表面电荷密度视为基本一致。双电极、四电极和六电极输出差异主要来自电极结构改变导致的静电感应分布、有效电荷转移次数和电荷利用率变化，而不是 PTFE 球带电量增加。

已知实测校准点：

```text
双电极转移电荷：380 nC
四电极转移电荷：650 nC
```

## 复现方式

推荐使用 bundled Python 或本机安装了 `numpy` 的 Python：

```powershell
python .\scripts\produce_n4_full_cycle_current.py
```

输出将写入：

```text
results_multigrid_calibrated/n4_full_cycle_current/
```

## COMSOL 模型

`comsol_models/` 中保存了双电极、四电极、六电极的 2D 几何/静电模型入口，可用于进一步静电场分析。当前电流波形不是 COMSOL 直接瞬态电路求解结果，而是结合实验转移电荷校准后的等效电荷转移/负载响应模型。

## 目录说明

```text
scripts/                      仿真、绘图、COMSOL 建模脚本
comsol_models/                COMSOL 几何与静电模型
results_multigrid_calibrated/ 多栅格校准仿真结果
results_p2_calibrated/        双电极阵列负载扫描校准结果
docs/                         复现说明与方法文档
```

## 注意

仓库中保留了若干历史中间结果目录，用于追溯建模过程。用于专利插图和说明时，请优先采用 `results_multigrid_calibrated/n4_full_cycle_current/` 中的冻结版本。
