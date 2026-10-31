# 桌面仿真软件目录边界

```text
desktop_simulator/
├─ link_budget/
│  ├─ algorithm/   链路预算、传播损耗、可用性、雨衰序列、调制编码
│  └─ ui/          链路计算页面
├─ link_sim/
│  ├─ algorithm/   功放、ADC/DAC、滤波器、混频器和器件模型库
│  └─ ui/          链路仿真页面
├─ constellation_sim/
│  ├─ algorithm/   星座、业务、规划调度、缓存和结果存储
│  └─ ui/          主窗口、Cesium视图和星座交互
└─ main.py         统一启动入口
```

三个业务域通过公开接口交互。原 `feeder_link_sim/ui` 只作为界面功能参考，不是统一软件的运行时依赖。
