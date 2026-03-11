# AGENTS.md - 项目上下文说明

## 项目概述

这是一个基于 Python 的**机器人导航与规划实验程序**，面向机器人足球/导航仿真平台（ZSS - 浙江大学智能系统与仿真平台）。核心目标是实现机器人在动态环境中的实时导航、避障与脱困规划。

### 核心技术栈

- **导航方法**: 人工势场法 (APF) + PRM 概率路网混合导航
- **控制方法**: PID 控制器
- **通信协议**: Protocol Buffers (protobuf) over UDP
- **主要依赖**: `numpy`, `scipy` (KDTree), `protobuf`

---

## 项目架构

```
动态规划_self/
├── main.py              # 主程序入口，整体闭环控制
├── func.py              # 人工势场法导航核心
├── prm.py               # PRM 概率路网规划与 Dijkstra 搜索
├── pid.py               # PID 控制器
├── vision.py            # 视觉信息接收与解析（UDP + Protobuf）
├── action.py            # 运动命令发送（UDP + Protobuf）
├── debug.py             # 调试绘图发送
├── proto/               # Protobuf 协议定义
│   ├── vision_detection.proto
│   ├── zss_cmd.proto
│   └── zss_debug.proto
├── vision_detection_pb2.py  # 自动生成
├── zss_cmd_pb2.py           # 自动生成
└── zss_debug_pb2.py         # 自动生成
```

### 系统分层

| 层级 | 文件 | 职责 |
|------|------|------|
| 感知层 | `vision.py` | 接收并解析视觉数据 |
| 局部导航层 | `func.py` | 计算势场合力方向 |
| 脱困规划层 | `prm.py` | PRM 图搜索生成脱困路径 |
| 控制执行层 | `pid.py` + `action.py` | 方向误差控制与命令发送 |
| 可视化层 | `debug.py` | 绘制机器人/目标/路径 |

---

## 通信端口配置

| 功能 | 地址 | 端口 |
|------|------|------|
| 视觉输入 | `127.0.0.1` | `23333` |
| 动作输出 | `localhost` | `50001` |
| 调试输出 | `localhost` | `20001` |

---

## 运行方式

### 环境要求

```bash
# Python 3.12+
pip install numpy scipy protobuf
```

### 启动程序

```bash
python main.py
```

**注意**: 程序依赖外部仿真平台提供视觉数据源，需要先启动 ZSS 仿真平台。

---

## 核心算法说明

### 1. 人工势场法 (APF) - `func.py`

**吸引力计算**:
```python
if d_goal <= d_a:
    Fatt = -2 * k_a * (robot_position - goal)
else:
    Fatt = -2 * k_a * d_a * (robot_position - goal) / d_goal
```

**斥力计算** (针对黄方机器人):
```python
if d_frep[i] <= d_o:
    Frep = k_b * (1/d - 1/d_o) * 1/d^2 * direction
```

**关键参数**:
| 参数 | 值 | 含义 |
|------|-----|------|
| `k_a` | 0.00055 | 吸引力增益 |
| `k_b` | 7e7 | 斥力增益 |
| `d_o` | 800 | 障碍影响半径 |
| `d_a` | 1000 | 吸引力饱和距离 |

### 2. PRM 概率路网规划 - `prm.py`

**工作流程**:
1. 提取障碍物位置（蓝方+黄方机器人）
2. 自由空间采样 (`N_SAMPLE=50`)
3. 构建路网图 (`KNN=10`)
4. Dijkstra 最短路径搜索
5. 返回路径中继点作为脱困目标

**关键参数**:
| 参数 | 值 | 含义 |
|------|-----|------|
| `N_SAMPLE` | 50 | 采样点数量 |
| `KNN` | 10 | 每节点最大邻居数 |
| `robot_size` | 200 | 机器人尺寸半径 |
| `avoid_dist` | 300 | 额外安全距离 |
| `MAX_EDGE_LEN` | 5000 | 边长上限 |

### 3. PID 控制器 - `pid.py`

**默认参数**: `Kp=0.3, Ki=0, Kd=0.001`

**控制策略**:
```python
vw = -w_pid.output * 10  # 角速度
vx = ||totalv|| * 2 / (1 + 2 * abs(w_pid.output))  # 线速度
```

"转得越急，走得越慢" —— 方向误差大时优先转向。

---

## 主循环逻辑 (`main.py`)

```
while True:
    1. 读取机器人当前位置
    2. 根据 flag 选择目标点
    3. 计算势场合力 F = Force(...)
    4. 估计实际朝向 w_real 与期望朝向 w_expect
    5. PID 计算角速度修正量
    6. 判断是否到达目标 / 是否卡住
    7. 卡住时触发 PRM 脱困规划
    8. 发送控制命令
    9. 发送调试绘图
```

### 状态机

| flag 值 | 状态 |
|---------|------|
| `1` | 导航至目标 `[-2400, -1500]` |
| `-1` | 导航至目标 `[2400, 1500]` |
| `0` | PRM 脱困状态 |

### 卡住检测条件

```python
d_move < 10 and repeat > 200
```

---

## 模块接口说明

### Vision 类 (`vision.py`)

```python
vision = Vision()
robot = vision.my_robot  # 蓝方 0 号机器人
# 属性: robot.x, robot.y, robot.orientation, robot.visible
#       vision.blue_robot[i], vision.yellow_robot[i]
```

### Action 类 (`action.py`)

```python
action = Action()
action.sendCommand(vx=100, vy=0, vw=10)  # 前向速度、侧向速度、角速度
```

### Debugger 类 (`debug.py`)

```python
debugger = Debugger()
package = Debug_Msgs()
debugger.draw_circle(package, x, y)
debugger.draw_line(package, x1, y1, x2, y2)
debugger.draw_finalpath(package, path_x, path_y)
debugger.send(package)
```

### PRM 类 (`prm.py`)

```python
planner = PRM()
path_x, path_y, road_map, sample_x, sample_y = planner.plan(
    vision=vision,
    start_x=robot.x, start_y=robot.y,
    goal_x=goal[0], goal_y=goal[1]
)
```

---

## 代码风格与约定

- 使用 **numpy** 进行向量运算
- 类名采用 **大驼峰** 命名
- 函数名采用 **小写下划线** 命名
- 主循环中不使用复杂类继承，保持过程式清晰
- Protobuf 文件 (`*_pb2.py`) 为自动生成，**不要手动修改**

---

## 已知局限

1. APF 容易陷入局部极小值（由 PRM 补救）
2. APF 仅使用黄方机器人作为障碍，PRM 使用蓝方+黄方（不一致）
3. 朝向估计使用位移方向而非 `orientation`
4. `angle()` 函数未使用 `atan2`，边界条件处理不够稳健
5. `search()` 备用函数鲁棒性不足

---

## 推荐阅读顺序

1. `main.py` → 理解主循环流程
2. `vision.py` + `action.py` → 理解输入输出接口
3. `func.py` → 理解人工势场导航
4. `pid.py` → 理解方向控制
5. `prm.py` → 理解脱困规划
6. `debug.py` + `proto/*.proto` → 理解可视化与通信格式

---

## 扩展建议

1. 统一 APF 与 PRM 的障碍建模
2. 直接利用 `orientation` 做姿态控制
3. 引入路径跟踪器而非只跟踪中间点
4. 增加速度限幅、加速度约束与轨迹平滑
5. 改进卡死检测与动态障碍预测机制
