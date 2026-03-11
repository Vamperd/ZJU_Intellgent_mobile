# AGENTS.md - 项目上下文文件

## 项目概述

本项目是一个基于 Python 的二维机器人导航与路径规划实验程序，面向 RoboCup/小型组足球仿真平台。程序通过 UDP 从视觉模块接收场上机器人状态，基于当前障碍物分布进行路径规划，再将速度指令发送给机器人。

**技术栈**: Python 3.12, NumPy, SciPy, Protocol Buffers, UDP Socket

**当前使用算法**: A* 路径规划算法（保证最优路径）

---

## 目录结构

```
静态规划_self/
├── main.py              # 主程序入口，串联整个导航流程
├── vision.py            # 视觉数据接收模块（UDP客户端）
├── action.py            # 运动指令发送模块（UDP客户端）
├── astar.py             # A* 路径规划算法实现（当前使用）
├── rrt.py               # RRT* 路径规划算法实现（备选）
├── prm.py               # PRM 路径规划算法实现（备选）
├── move.py              # PID 路径跟踪控制器
├── Track.py             # 另一种跟踪控制器实现
├── debug.py             # 规划结果可视化与调试
├── vision_detection_pb2.py   # 视觉数据协议缓冲区（已编译）
├── zss_cmd_pb2.py            # 命令协议缓冲区（已编译）
├── zss_debug_pb2.py          # 调试协议缓冲区（已编译）
└── proto/                    # Protocol Buffers 源文件
    ├── vision_detection.proto
    ├── zss_cmd.proto
    └── zss_debug.proto
```

---

## 核心模块说明

### 1. vision.py - 视觉数据接收
- 监听 UDP 端口 `127.0.0.1:23333` 接收视觉数据
- 解析 Protocol Buffers 格式的机器人位置、速度、朝向
- 维护蓝方/黄方机器人状态列表（最多16个）
- `my_robot` 属性返回己方0号车状态

### 2. action.py - 运动指令发送
- 发送速度指令到 `localhost:50001`
- 支持线速度 (vx, vy) 和角速度 (vw) 控制
- 包含踢球和运球功能（当前未使用）

### 3. astar.py - A* 路径规划（当前使用）
- **A*** 最优路径规划算法实现
- 将场地离散化为网格进行搜索
- 关键参数：
  - `grid_resolution = 100`：网格分辨率（mm）
  - `diagonal_move = True`：允许对角移动
- 核心方法：
  - `plan()`: 主规划函数
  - `_create_grid_map()`: 创建障碍物网格地图
  - `_astar_search()`: A* 搜索核心算法
  - `_smooth_path()`: 路径剪枝优化
  - `_interpolate_path()`: 路径插值平滑
- 优势：
  - 保证找到最短路径
  - 计算确定性，结果可复现
  - 搜索过程可完全可视化

### 4. move.py - PID 路径跟踪
- 线速度 PID 参数：`Kp=1.5, Ki=0.01, Kd=0`
- 角速度 PID 参数：`Kp=5, Ki=0.015, Kd=0.015`
- `get_action()`: 计算当前速度指令（含边界检查）
- `turn_arround()`: 原地旋转约π弧度
- **路径跟踪逻辑**：
  - `path_x[0]` = 起点，`path_x[point_num]` = 终点
  - 正向：从索引 1 逐步增加到 point_num
  - 反向：从 point_num 逐步减少到 0

### 5. debug.py - 可视化调试
- 发送调试图形到 `localhost:20001`
- 支持绘制：圆、线段、点、路径、树结构
- `create_astar_visual_callback()`: A* 规划过程实时可视化
  - `grid_map`: 显示障碍物分布
  - `searching`: 实时搜索过程（已访问-白色，开放集-黄色，当前-青色）
  - `path_found`: 原始路径
  - `path_smoothed`: 平滑前后对比
  - `final_result`: 最终路径
- `create_visual_callback()`: RRT* 规划过程实时可视化

### 6. rrt.py - RRT* 路径规划（备选）
- 渐近最优路径规划算法
- 关键参数：`MAX_ITER=3000`, `SEARCH_RADIUS=1500`
- 适合动态环境，但不保证最优

### 7. prm.py - PRM 规划算法（备选）
- 基于概率路线图的路径规划
- 使用 Dijkstra 算法搜索最短路径
- 参数：`N_SAMPLE=100, KNN=10`

---

## 运行方式

```bash
python main.py
```

**运行依赖**:
- Python 3.x
- `numpy`, `scipy`, `protobuf`
- 外部视觉系统 (UDP 23333)
- 机器人控制接收端 (UDP 50001)
- 调试显示接收端 (UDP 20001)

---

## 系统架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  视觉数据    │────▶│  障碍物提取  │────▶│  A* 规划    │
│  (UDP 23333)│     │  (KDTree)   │     │  (网格搜索)  │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  速度指令   │◀────│  PID 控制   │◀────│  路径平滑   │
│  (UDP 50001)│     │  (Move)     │     │  (剪枝+插值) │
└─────────────┘     └─────────────┘     └─────────────┘
```

---

## 关键参数配置

| 参数 | 值 | 说明 |
|------|-----|------|
| 场地范围 | x: [-4500, 4500], y: [-3000, 3000] | 单位：mm |
| 机器人半径 | 200mm | `robot_size` |
| 安全距离 | 200mm | `avoid_dist` |
| 网格分辨率 | 100mm | A* 算法使用 |
| 最大速度 | 4000mm/s | 线速度上限 |
| 最大角速度 | 5 rad/s | 角速度上限 |
| 航点切换阈值 | 200mm | 到达判定距离 |

---

## 数据结构

### Robot 类（vision.py）
```python
class Robot:
    id: int           # 机器人编号
    visible: bool     # 是否可见
    x, y: float       # 位置（滤波后）
    vel_x, vel_y      # 速度
    orientation: float # 朝向角度
    raw_x, raw_y      # 原始位置
```

### Node 类（rrt.py/prm.py）
```python
class Node:
    x, y: float       # 节点坐标
    cost: float       # 累计代价
    parent: int       # 父节点索引
```

---

## 开发约定

1. **坐标系**: 场地中心为原点，x轴向右为正，y轴向上为正
2. **角度**: 弧度制，范围 [-π, π]
3. **单位**: 长度为 mm，角度为 rad
4. **障碍物建模**: 圆形膨胀，半径 = robot_size + avoid_dist
5. **己方机器人**: 蓝方0号车（`vision.my_robot`）
6. **路径索引**: `path_x[0]` = 起点，`path_x[n]` = 终点

---

## 通信协议

### 视觉数据（接收）
- 端口: 23333
- 格式: Protocol Buffers (`Vision_DetectionFrame`)
- 包含: 蓝方/黄方机器人位置、速度、朝向

### 控制指令（发送）
- 端口: 50001
- 格式: Protocol Buffers (`Robots_Command`)
- 包含: robot_id, velocity_x, velocity_y, velocity_r

### 调试信息（发送）
- 端口: 20001
- 格式: Protocol Buffers (`Debug_Msgs`)
- 支持: 线段、圆、点等图形元素

---

## 常见修改点

1. **修改目标点**: `main.py` 中的 `goal_x, goal_y`
2. **调整 A* 参数**: `main.py` 中的 `AStar(grid_resolution=100, diagonal_move=True)`
3. **调整控制参数**: `move.py` 中的 PID 系数
4. **切换规划算法**: `main.py` 中修改 planner 和 visual_callback：
   ```python
   # 使用 A*（当前）
   planner = AStar(grid_resolution=100, diagonal_move=True)
   visual_callback = debugger.create_astar_visual_callback(...)
   
   # 切换为 RRT*
   planner = RRT()
   visual_callback = debugger.create_visual_callback(...)
   ```

---

## 注意事项

- 视觉系统需要先启动，否则会超时报错
- A* 算法保证找到最优路径（最短路径）
- 路径平滑后会进行剪枝和插值处理
- 到达目标后会自动调头并原路返回（可配置往返次数）
- 调试可视化功能可选，不影响核心功能运行