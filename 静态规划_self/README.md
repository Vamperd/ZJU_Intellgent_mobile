# 静态规划_self 代码解析说明

## 1. 项目概述

本项目是一个基于 Python 的二维机器人导航与路径规划实验程序，面向 RoboCup/小型组足球仿真平台。程序通过 UDP 从视觉模块接收场上机器人状态，基于当前障碍物分布进行路径规划，再将速度指令发送给机器人。

核心模块：
- `vision.py`：接收视觉数据，维护场上机器人状态
- `rrt.py`：RRT* 路径规划算法
- `move.py`：PID 路径跟踪控制
- `action.py`：发送运动指令
- `debug.py`：规划结果可视化
- `main.py`：主程序，串联整个导航流程

---

## 2. RRT* 路径规划算法

### 2.1 算法背景与原理

**RRT（快速扩展随机树）** 是一种基于采样的路径规划算法，通过从起点开始逐步构建一棵搜索树来探索空间。RRT 的优点是能够高效处理高维空间和复杂障碍物环境，但缺点是找到的路径往往不是最优的。

**RRT*** 是 RRT 的优化版本，由 Sertac Karaman 和 Emilio Frazzoli 在 2010 年提出。RRT* 引入了两个关键机制：

1. **最优父节点选择**：新节点加入树时，从多个邻居中选择代价最小的作为父节点
2. **重布线（Rewiring）**：检查已有节点是否通过新节点可以获得更小代价，若是则重新连接

这两个机制使得 RRT* 具有**渐近最优性**——随着采样点数量增加，路径代价会收敛到最优值。

### 2.2 核心数据结构

```python
class Node:
    def __init__(self, x, y, cost, parent):
        self.x = x          # 节点x坐标
        self.y = y          # 节点y坐标
        self.cost = cost    # 从起点到该节点的累计代价
        self.parent = parent  # 父节点索引
```

树的表示方式：
- `sample_x[], sample_y[]`：所有节点的坐标数组
- `parent_index[]`：每个节点的父节点索引（根节点为 -1）
- `cost[]`：每个节点的累计代价

### 2.3 算法整体流程

```
RRT* 主循环：
while iteration < MAX_ITER:
    1. 随机采样目标点 Q_rand
    2. 找到树中最近节点 Q_nearest
    3. 沿方向扩展固定步长得到 Q_new
    4. 碰撞检测
    5. [RRT*特有] 搜索邻居节点
    6. [RRT*特有] 选择最优父节点
    7. 添加新节点到树
    8. [RRT*特有] 重布线优化
    9. 尝试连接目标点
```

### 2.4 详细实现步骤

#### 2.4.1 障碍物初始化

```python
def plan(self, vision, start_x, start_y, goal_x, goal_y, visual_callback=None):
    # 从视觉系统提取障碍物位置
    obstacle_x = [-9999]
    obstacle_y = [-9999]
    for robot_blue in vision.blue_robot:
        if robot_blue.visible and robot_blue.id > 0:  # 排除己方0号车
            obstacle_x.append(robot_blue.x)
            obstacle_y.append(robot_blue.y)
    for robot_yellow in vision.yellow_robot:
        if robot_yellow.visible:
            obstacle_x.append(robot_yellow.x)
            obstacle_y.append(robot_yellow.y)
    
    # 使用 KDTree 构建障碍物空间索引，加速最近邻查询
    obstree = KDTree(np.vstack((obstacle_x, obstacle_y)).T)
```

障碍物建模说明：
- 蓝方机器人：仅将 `id > 0` 的可见车辆视为障碍（0号车是己方）
- 黄方机器人：所有可见车辆都视为障碍
- 安全膨胀半径：`robot_size + avoid_dist = 200 + 200 = 400`

#### 2.4.2 目标偏置采样

```python
def sampling(self, start_x, start_y, goal_x, goal_y, obstree):
    while iteration < self.MAX_ITER:
        # 目标偏置采样：15% 概率直接采样目标点
        random_num = random.random()
        if random_num < 0.15:
            Q_rand_x = goal_x
            Q_rand_y = goal_y
        else:
            # 85% 概率在场地范围内随机采样
            Q_rand_x = (random.random() * (self.maxx - self.minx)) + self.minx
            Q_rand_y = (random.random() * (self.maxy - self.miny)) + self.miny
```

目标偏置的作用：
- **加速收敛**：直接朝目标扩展可以更快找到可行路径
- **保持探索性**：大部分时候随机采样，保证对空间的充分探索
- **平衡策略**：15% 的偏置率是经验值，过大会降低探索能力，过小会减慢收敛

#### 2.4.3 最近邻查找与扩展

```python
# 每轮迭代重建节点 KDTree
Nodetree = KDTree(np.vstack((sample_x, sample_y)).T)

# 找到离采样点最近的树节点
distance, nearest_idx = Nodetree.query(np.array([Q_rand_x, Q_rand_y]))
Q_nearest_x, Q_nearest_y = sample_x[nearest_idx], sample_y[nearest_idx]

# 固定步长扩展
def extend(self, Q_nearest_x, Q_nearest_y, Q_rand_x, Q_rand_y):
    dy = Q_rand_y - Q_nearest_y
    dx = Q_rand_x - Q_nearest_x
    theta = math.atan2(dy, dx)  # 计算方向角
    Q_new_x = Q_nearest_x + 1000 * math.cos(theta)
    Q_new_y = Q_nearest_y + 1000 * math.sin(theta)
    return Q_new_x, Q_new_y
```

扩展策略说明：
- 使用 `atan2` 计算方向角，能正确处理所有象限
- 固定步长 1000 可以平衡树的稠密程度和计算效率
- 步长过大会增加碰撞概率，过小会降低探索效率

#### 2.4.4 RRT* 核心：邻居搜索

```python
def find_near_nodes(self, tree, x, y, n):
    """查找新节点周围的邻居节点"""
    # RRT* 标准动态半径公式
    # γ 应大于场地对角线/√π，这里取 5000
    gamma = 5000
    # d=2 为空间维度
    radius = gamma * math.sqrt(math.log(n + 1) / (n + 1))
    # 不超过最大搜索半径
    radius = min(radius, self.SEARCH_RADIUS)  # SEARCH_RADIUS = 1500
    # 设置最小半径避免搜索范围过小
    if radius < 300:
        radius = 300
    
    # 使用 KDTree 的球查询
    indices = tree.query_ball_point([x, y], radius)
    return list(indices)
```

动态搜索半径的数学原理：

RRT* 的渐近最优性要求搜索半径满足：
$$r_n = \gamma \cdot \left(\frac{\log n}{n}\right)^{1/d}$$

其中：
- $n$ 是当前节点数
- $d$ 是空间维度（二维空间 $d=2$）
- $\gamma$ 是常数，需满足 $\gamma > \gamma_{RRT^*} = 2\left(1+\frac{1}{d}\right)^{1/d}\left(\frac{\mu(X_{free})}{\zeta_d}\right)^{1/d}$

这个公式的意义：
- 随着节点数增加，搜索半径逐渐减小
- 保证每个新节点能够找到足够多的邻居进行优化
- 同时控制计算复杂度在合理范围内

#### 2.4.5 RRT* 核心：最优父节点选择

```python
def choose_parent(self, new_x, new_y, neighbor_indices, sample_x, sample_y, cost, obstree):
    """在邻居中选择代价最小的作为父节点"""
    best_parent = None
    best_cost = float('inf')
    
    for idx in neighbor_indices:
        px, py = sample_x[idx], sample_y[idx]
        
        # 检查连线是否无碰撞
        if not self.check_obs(px, py, new_x, new_y, obstree):
            edge_cost = math.hypot(new_x - px, new_y - py)
            total_cost = cost[idx] + edge_cost  # 父节点代价 + 边代价
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_parent = idx
    
    return best_parent, best_cost
```

选择逻辑：
1. 遍历搜索半径内的所有邻居节点
2. 对每个邻居，计算通过它到达新节点的总代价
3. 只有无碰撞的连线才被考虑
4. 选择总代价最小的邻居作为父节点

与普通 RRT 的区别：
- **普通 RRT**：直接选择最近的节点作为父节点
- **RRT***：从所有可达邻居中选择代价最小的，可能不是最近节点

#### 2.4.6 RRT* 核心：重布线（Rewiring）

```python
def rewire(self, new_x, new_y, new_idx, neighbor_indices, 
           sample_x, sample_y, cost, parent_index, obstree):
    """重布线：检查邻居是否通过新节点可以获得更小代价"""
    for idx in neighbor_indices:
        if idx == new_idx:
            continue
        
        px, py = sample_x[idx], sample_y[idx]
        edge_cost = math.hypot(new_x - px, new_y - py)
        potential_cost = cost[new_idx] + edge_cost  # 通过新节点的代价
        
        # 如果通过新节点代价更小，且连线无碰撞
        if potential_cost < cost[idx]:
            if not self.check_obs(new_x, new_y, px, py, obstree):
                cost[idx] = potential_cost      # 更新代价
                parent_index[idx] = new_idx     # 更新父节点
```

重布线的意义：

重布线是 RRT* 实现渐近最优性的关键。当一个新节点加入树后，它不仅作为后续节点的潜在父节点，还会回过头优化已有节点的连接方式。

示例说明：
```
假设树结构：
    A -----> B (cost=10)
    
新节点 C 加入后，发现 A -> C -> B 的代价是 8
    A --C--> B (cost=8)
    
重布线会将 B 的父节点从 A 改为 C
```

重布线的效果：
- 随着迭代进行，树的拓扑结构会不断优化
- 路径会变得越来越短、越来越平滑
- 最终收敛到近似最优路径

#### 2.4.7 碰撞检测

```python
def check_obs(self, ix, iy, nx, ny, obstree):
    """检测两点之间的连线是否与障碍物碰撞"""
    x, y = ix, iy
    dx = nx - ix
    dy = ny - iy
    angle = math.atan2(dy, dx)
    dis = math.hypot(dx, dy)

    # 边长超过阈值直接判定碰撞
    if dis > self.MAX_EDGE_LEN:
        return True

    # 离散步进检测
    step_size = self.robot_size + self.avoid_dist  # 400
    steps = round(dis / step_size)
    
    for i in range(steps):
        distance, index = obstree.query(np.array([x, y]))
        if distance <= self.robot_size + self.avoid_dist:
            return True  # 碰撞
        x += step_size * math.cos(angle)
        y += step_size * math.sin(angle)

    # 检查终点
    distance, index = obstree.query(np.array([nx, ny]))
    if distance <= 100:
        return True

    return False  # 无碰撞
```

碰撞检测策略：
- 采用离散采样方式近似连续碰撞检测
- 步长等于安全距离，确保不会遗漏碰撞
- 使用 KDTree 快速查询最近障碍物距离

#### 2.4.8 目标点连接与持续优化

```python
# 在主循环中，尝试从新节点连接到目标
goal_dist = math.hypot(Q_new_x - goal_x, Q_new_y - goal_y)
if not self.check_obs(Q_new_x, Q_new_y, goal_x, goal_y, obstree):
    new_goal_cost = cost[new_node_idx] + goal_dist
    
    if goal_node_idx == -1:
        # 第一次到达目标
        goal_node_idx = len(sample_x)
        sample_x.append(goal_x)
        sample_y.append(goal_y)
        parent_index.append(new_node_idx)
        cost.append(new_goal_cost)
        best_goal_cost = new_goal_cost
        print(f"Goal reached at iteration {iteration}, cost: {best_goal_cost:.0f}")
    elif new_goal_cost < best_goal_cost:
        # 找到更好的路径
        best_goal_cost = new_goal_cost
        parent_index[goal_node_idx] = new_node_idx
        cost[goal_node_idx] = new_goal_cost

# 到达目标后继续优化
if goal_node_idx != -1:
    post_goal_iterations -= 1
    if post_goal_iterations <= 0:
        print(f"Optimization finished. Final cost: {cost[goal_node_idx]:.0f}")
        break
```

持续优化策略：
- 第一次到达目标后不立即停止
- 继续迭代 300 次，寻找更优路径
- 通过重布线机制，后续迭代可能找到代价更低的路径

### 2.5 路径后处理

#### 2.5.1 路径剪枝（Path Pruning）

```python
def smooth_path(self, path_x, path_y, obstree, visual_callback=None):
    """路径剪枝：跳过可直接连通的中间节点，减少转角"""
    if len(path_x) <= 2:
        return path_x, path_y
    
    smoothed_x, smoothed_y = [path_x[0]], [path_y[0]]
    i = 0
    
    while i < len(path_x) - 1:
        # 从远到近尝试跳过中间节点
        for j in range(len(path_x) - 1, i, -1):
            if not self.check_obs(path_x[i], path_y[i], path_x[j], path_y[j], obstree):
                # 找到可以跳过的节点
                smoothed_x.append(path_x[j])
                smoothed_y.append(path_y[j])
                i = j
                break
        else:
            i += 1
            if i < len(path_x):
                smoothed_x.append(path_x[i])
                smoothed_y.append(path_y[i])
    
    return smoothed_x, smoothed_y
```

剪枝算法说明：
- **贪心策略**：从当前位置尝试跳到最远的可达节点
- **减少转角**：跳过不必要的中间节点，使路径更直接
- **保持安全**：每次跳跃都要进行碰撞检测

#### 2.5.2 路径插值平滑

```python
def interpolate_path(self, path_x, path_y, obstree, interval=400):
    """路径插值平滑：在路径节点间插入中间点"""
    if len(path_x) <= 1:
        return path_x, path_y
    
    new_x, new_y = [path_x[0]], [path_y[0]]
    
    for i in range(len(path_x) - 1):
        x1, y1 = path_x[i], path_y[i]
        x2, y2 = path_x[i + 1], path_y[i + 1]
        
        dist = math.hypot(x2 - x1, y2 - y1)
        if dist > interval:
            # 插入中间点
            num_points = int(dist / interval)
            for j in range(1, num_points + 1):
                ratio = j / (num_points + 1)
                interp_x = x1 + ratio * (x2 - x1)
                interp_y = y1 + ratio * (y2 - y1)
                # 检查插值点安全性
                col_dist, _ = obstree.query([interp_x, interp_y])
                if col_dist >= self.robot_size + self.avoid_dist:
                    new_x.append(interp_x)
                    new_y.append(interp_y)
        
        new_x.append(x2)
        new_y.append(y2)
    
    return new_x, new_y
```

插值的作用：
- **增加路径点密度**：便于控制器更精细地跟踪
- **平滑轨迹**：减少相邻路径点之间的角度变化
- **安全验证**：每个插值点都经过碰撞检测

### 2.6 关键参数说明

| 参数 | 值 | 说明 |
|------|-----|------|
| `MAX_ITER` | 3000 | 最大迭代次数，控制规划时间上限 |
| `SEARCH_RADIUS` | 1500 | 最大邻居搜索半径，限制重布线范围 |
| `N_SAMPLE` | 100 | 采样点数量（PRM使用） |
| `KNN` | 10 | 每节点最大邻居数（PRM使用） |
| `MAX_EDGE_LEN` | 5000 | 最大边长，超过则判定不可达 |
| `robot_size` | 200 | 机器人本体半径（mm） |
| `avoid_dist` | 200 | 额外安全距离（mm） |
| 扩展步长 | 1000 | 每次扩展的固定距离（mm） |
| 目标偏置率 | 15% | 直接采样目标点的概率 |
| 场地范围 | x: [-4500, 4500], y: [-3000, 3000] | 单位：mm |

### 2.7 RRT* vs 普通 RRT 对比

| 特性 | 普通 RRT | RRT* |
|------|----------|------|
| 父节点选择 | 最近节点 | 代价最小的邻居 |
| 路径优化 | 无 | 重布线机制 |
| 路径质量 | 可行解 | 渐近最优 |
| 计算复杂度 | O(n log n) | O(n log n) 但常数更大 |
| 适用场景 | 快速响应 | 需要高质量路径 |

---

## 3. Move 路径跟踪控制

### 3.1 控制架构

Move 模块实现了**航点跟踪（Waypoint Tracking）**控制策略，将规划生成的离散路径点序列转化为机器人的速度指令。

控制流程：
```
当前位置 + 目标航点 → 误差计算 → PID控制器 → 速度指令
```

### 3.2 辅助函数详解

#### 3.2.1 距离计算

```python
def cal_dist(x1, y1, x2, y2):
    """计算两点之间的欧氏距离"""
    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist
```

#### 3.2.2 目标角度计算

```python
def cal_theta(x1, y1, x2, y2):
    """计算从点(x1,y1)指向点(x2,y2)的方向角"""
    if x1 >= x2 and y1 >= y2:
        return math.atan((y1 - y2) / (x1 - x2)) - math.pi
    if x1 >= x2 and y1 < y2:
        return math.atan((y1 - y2) / (x1 - x2)) + math.pi
    if x1 < x2:
        return math.atan((y1 - y2) / (x1 - x2))
```

角度计算说明：
- 返回的角度是机器人需要朝向的方向
- 根据目标点的相对位置分情况处理
- 确保角度在正确的象限

#### 3.2.3 角度误差归一化

```python
def cal_theta_error(theta1, vision):
    """计算目标角度与当前朝向的误差，归一化到 [-π, π]"""
    theta_error = theta1 - vision.my_robot.orientation
    while theta_error > math.pi:
        theta_error = theta_error - 2 * math.pi
    while theta_error < -math.pi:
        theta_error = theta_error + 2 * math.pi
    return theta_error
```

归一化的必要性：
- 角度是周期性的，π 和 -π 表示同一方向
- 直接相减可能导致误差被高估（如 3.14 - (-3.14) = 6.28）
- 归一化后误差表示"最短旋转角度"

#### 3.2.4 速度映射函数

```python
def function(x):
    """速度映射函数：抛物线形式"""
    if x > 3000:
        x = 3000
    return -1/3000 * x * (x - 6000)
```

函数特性：
```
f(x) = -1/3000 * x * (x - 6000)
     = -1/3000 * x² + 2x
```

这是一个开口向下的抛物线：
- 当 x = 0 时，f(0) = 0
- 当 x = 3000 时，f(3000) = 3000（最大值）
- 当 x = 6000 时，f(6000) = 0

作用：
- 将 PID 输出映射到合理的速度范围
- 限制最大速度
- 提供平滑的速度变化曲线

### 3.3 PID 控制器详解

#### 3.3.1 线速度 PID

```python
def PID_vx(vx_error, vx_error_sum, vx_error_last, flag):
    Kp = 1.5
    Ki = 0.01
    Kd = 0
    
    # 积分限幅，防止积分饱和
    if vx_error_sum > 10000:
        vx_error_sum = 10000
    
    output = Kp * vx_error + Ki * vx_error_sum + Kd * vx_error_last
    
    # 非线性映射
    if flag:
        output = function(output)
        output = max(output, 1200)  # 中间点最小速度
    else:
        output = function(output)
        output = max(output, 450)   # 终点最小速度
    
    # 速度上限
    if output > 4000:
        output = 4000
    
    return output
```

参数说明：
- **Kp = 1.5**：比例系数，决定对位置误差的响应强度
- **Ki = 0.01**：积分系数，消除稳态误差
- **Kd = 0**：微分系数，本实现中未使用

最小速度策略：
- 中间航点：最小 1200，确保快速通过
- 终点航点：最小 450，确保精确定位

#### 3.3.2 角速度 PID

```python
def PID_vw(vw_error, vw_error_sum, vw_error_last):
    Kp = 5
    Ki = 0.015
    Kd = 0.015
    
    # 积分限幅
    if vw_error_sum > 40:
        vw_error_sum = 40
    
    output = Kp * vw_error + Ki * vw_error_sum + Kd * vw_error_last
    
    # 角速度上限
    if output > 5:
        output = 5
    
    return output
```

参数说明：
- **Kp = 5**：较高的比例系数，快速修正朝向
- **Ki = 0.015**：消除朝向稳态误差
- **Kd = 0.015**：抑制角速度震荡

### 3.4 Move 类核心方法

#### 3.4.1 状态变量

```python
class Move:
    def __init__(self):
        # 位置误差相关
        self.position_error_last = 0      # 上一次位置误差
        self.position_error = 0           # 当前位置误差
        self.position_error_sum = 0       # 位置误差积分
        
        # 角度误差相关
        self.theta_error_last = 0         # 上一次角度误差
        self.theta_error = 0              # 当前角度误差
        self.theta_error_sum = 0          # 角度误差积分
```

#### 3.4.2 主控制函数

```python
def get_action(self, vision, path_x, path_y, Next_point_index, direction=1):
    current_x, current_y = vision.my_robot.x, vision.my_robot.y
    
    # 计算到目标航点的距离
    dist_error1 = cal_dist(current_x, current_y, 
                           path_x[Next_point_index], path_y[Next_point_index])
    
    # 计算到相邻航点的距离（平滑误差）
    if direction == 1:
        dist_error2 = cal_dist(current_x, current_y,
                               path_x[Next_point_index+1], path_y[Next_point_index+1])
    else:
        dist_error2 = cal_dist(current_x, current_y,
                               path_x[Next_point_index-1], path_y[Next_point_index-1])
    
    # 取较小值作为误差（处理航点切换瞬间的误差跳变）
    dist_error = min(dist_error1, dist_error2)
    
    # 更新位置误差状态
    self.position_error_last = self.position_error
    self.position_error = dist_error
    self.position_error_sum = self.position_error_sum + self.position_error
    
    # 计算目标角度和角度误差
    theta_error = cal_theta(current_x, current_y,
                            path_x[Next_point_index], path_y[Next_point_index])
    
    # 更新角度误差状态
    self.theta_error_last = self.theta_error
    self.theta_error = cal_theta_error(theta_error, vision)
    self.theta_error_sum = self.theta_error_sum + self.theta_error
    
    # 判断是否为终点
    if Next_point_index == 0 or Next_point_index == len(path_x)-1:
        flag = False  # 终点，使用较低最小速度
    else:
        flag = True   # 中间点，使用较高最小速度
    
    # PID 计算输出
    vx = PID_vx(self.position_error, self.position_error_sum, 
                self.position_error_last, flag)
    vw = PID_vw(self.theta_error, self.theta_error_sum, self.theta_error_last)
    
    return vx, vw
```

控制逻辑说明：

1. **位置误差计算**：取到当前目标点和相邻点的最小距离，避免航点切换时误差突变

2. **角度误差计算**：计算机器人需要朝向的方向与当前朝向的差值

3. **状态更新**：维护误差的历史值用于 PID 计算

4. **速度输出**：根据位置和角度误差分别计算线速度和角速度

#### 3.4.3 原地旋转函数

```python
def turn_arround(self, action, vision):
    """原地旋转约 π 弧度"""
    start_theta = vision.my_robot.orientation
    theta_change = 0
    
    while theta_change < math.pi - 0.2:
        if theta_change > 2*math.pi/3:
            # 后期减速，提高停止精度
            action.sendCommand(vx=0, vw=2.5)
        else:
            # 前期快速旋转
            action.sendCommand(vx=0, vw=5)
        time.sleep(0.1)
        current_theta = vision.my_robot.orientation
        theta_change = abs(current_theta - start_theta)
```

旋转策略：
- **分阶段控制**：前期高速旋转，后期减速
- **开环控制**：不使用 PID，直接发送固定角速度
- **停止条件**：累计转角接近 π 弧度

### 3.5 速度耦合策略

在 `main.py` 中，实际发送的线速度进行了耦合调整：

```python
vx_actual = 1.2 * vx / (1 + 2 * abs(vw))
```

耦合效果分析：

| 角速度 | 分母 | 线速度比例 |
|--------|------|-----------|
| 0 | 1 | 100% |
| 1 | 3 | 40% |
| 2 | 5 | 24% |
| 3 | 7 | 17% |
| 4 | 9 | 13% |
| 5 | 11 | 11% |

设计意义：
- **直行加速**：角速度小时（接近直行），保持较高线速度
- **转向减速**：角速度大时（大转弯），自动降低线速度
- **平滑过渡**：使用连续函数，避免速度突变
- **轨迹稳定**：防止高速转向导致的轨迹震荡

---

## 4. 整体执行流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  视觉数据    │────▶│  障碍物提取  │────▶│  RRT* 规划  │
└─────────────┘     └─────────────┘     └─────────────┘
                                              │
                                              ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  速度指令   │◀────│  PID 控制   │◀────│  路径平滑   │
└─────────────┘     └─────────────┘     └─────────────┘
```

详细步骤：

1. **感知阶段**：接收视觉系统给出的场上机器人位姿
2. **建模阶段**：将其他机器人作为圆形障碍物（安全膨胀半径 400）
3. **规划阶段**：使用 RRT* 生成无碰撞最优路径
4. **后处理阶段**：剪枝优化 + 插值平滑
5. **控制阶段**：使用 PID 控制器沿路径点行驶
6. **往返阶段**：到达目标后调头并原路返回

---

## 5. 运行方式

```bash
python main.py
```

运行依赖：
- Python 3.x
- `numpy`, `scipy`, `protobuf`
- 外部视觉系统 (UDP 23333)
- 机器人控制接收端 (UDP 50001)
- 调试显示接收端 (UDP 20001)
