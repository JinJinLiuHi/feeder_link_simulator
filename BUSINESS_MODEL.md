# 业务建模算法（热点簇协同）

## 模块边界

本模块实现需求文档 5.4 节中从全球业务生成到接入卫星负载的部分：

```text
H3 网格和 WorldPop
  -> 地方时双峰业务
  -> 网格业务需求
  -> 用户侧可见卫星
  -> 相邻高业务网格聚类
  -> 热点簇内多星协同、每小区单星接入
  -> 每颗卫星的入口业务量（Gb/s）
```

本模块止于用户业务到接入卫星的映射。当前阶段不使用 ISL 拓扑、馈电星、信关站或端到端路径；这些因素留给后续站星规划与组网调度模块处理。

## 1. 离线空间预处理

`traffic_preprocess.aggregate_worldpop_to_h3()` 分块读取 WorldPop GeoTIFF。每个人口像元依据中心经纬度映射到 H3 单元，并在单元内求和：

```math
w_g=\sum_{p\in g}p_{pixel}
```

海洋、无效像元和低于人口门限的单元不输出。处理结果保存为 CSV，后续场景直接加载，不重复读取大型 GeoTIFF。

## 2. 静态权重

运行时先计算：

```math
\bar w_g = \frac{w_g^\alpha}{\sum_j w_j^\alpha}
```

`population_exponent` 对应指数 α。默认 α=1，即保持文档的人口求和口径；可设置小于 1 的值减弱超大城市集中效应。

## 3. 地方时双峰业务

地方时使用文档公式：

```math
T_{local}(g,t)=UTC(t)+\lambda_g/15
```

双峰曲线为：

```math
C(T)=c_0+A_1e^{-d(T,\mu_1)^2/(2\sigma_1^2)}+A_2e^{-d(T,\mu_2)^2/(2\sigma_2^2)}
```

其中 `d` 是 24 小时时钟上的循环距离，避免午夜附近的高斯曲线被截断。默认办公峰 10.5 h、σ=2.0 h，生活峰 20.5 h、σ=2.5 h。

网格业务量为：

```math
D_g(t)=D_{ref}\bar w_g C(T_{local})R_g(t)\quad[Gb/s]
```

`R_g(t)` 是可选、可复现的对数正态随机因子，默认关闭。随机因子的均值为 1，不改变长期平均量。

## 4. 高密业务识别

每个时间片根据当前业务量计算百分位门限。默认先取业务量前 5% 的网格作为热点候选，再按“热点相邻判定距离”连接候选网格并求连通分量。

至少包含两个相邻候选网格的连通分量才是热点簇。孤立的大业务网格仍按普通网格处理。每个小区只接入一颗主卫星；热点簇共享最多 K 颗协同卫星，使相邻小区分散承载。

## 5. 接入卫星硬约束

覆盖计算模块向业务模型提供每个网格的 `VisibilityCandidate`。候选卫星必须满足：

1. 用户侧几何可见；
2. 仰角不低于最低接入仰角；
3. 卫星属于当前业务场景允许使用的接入卫星集合。

## 6. 候选评分

当前实现只使用业务阶段已经具备的信息：

```math
Score_{g,s}=w_e Q_{elevation}+w_l Q_{load-balance}
```

其中仰角项优先选择用户侧链路几何条件较好的卫星，负载均衡项避免大量业务小区集中到同一颗卫星。负载单位为 Gb/s。模型不设置卫星容量与波束数量。

## 7. 普通小区与热点簇分配

普通网格选择综合评分最高的一颗卫星。热点簇先从簇内全部可见星中形成最多 K 颗卫星的协同池，再按业务量从高到低处理小区；每个小区在协同池中选择一颗可见卫星，并对已承载较多相邻小区的卫星施加负载惩罚。

业务模型不设置卫星容量和波束数量。如果网格没有任何可见卫星，其业务量计为未分配；物理容量是否能够承载这些需求由后续规划和链路模块判断。

## 8. 输出

每个时间片输出两级结果：

- 网格级：业务量（Gb/s）、当地时间、热点簇编号、可见卫星、接入卫星、未分配业务量；
- 卫星级：聚合入口业务量（Gb/s）和服务网格数；
- 全局级：总业务量、已分配量和未分配量（Gb/s）。

站星连接规划模块读取卫星级 `assigned_weight`，再结合信关站可见性、雨衰、馈电容量和ISL拓扑完成物理容量标定并选择馈电出口。

## 9. 调用接口

```python
from datetime import datetime, timezone
from traffic_model import TrafficModel, TrafficModelConfig, VisibilityCandidate
from traffic_preprocess import load_cells_csv

cells = load_cells_csv("traffic_grid.csv")
model = TrafficModel(cells, TrafficModelConfig())

visibility = {
    "grid-a": [
        VisibilityCandidate(100, 62.0),
        VisibilityCandidate(315, 44.0),
    ]
}

result = model.run_frame(
    datetime.now(timezone.utc), visibility
)
