"""
导航避障模块

包含:
- APFNavigator: 人工势场法导航器
- PRMPlanner: PRM 概率路网规划器
- Navigator: 统一导航接口
"""

import math
import random
import numpy as np
from scipy.spatial import KDTree


# =============================================================================
# APF 人工势场法导航器
# =============================================================================

class APFConfig:
    """APF 参数配置"""
    K_ATTRACT = 0.00055      # 吸引力增益
    K_REPEL = 7e7            # 斥力增益
    D_OBSTACLE = 800         # 障碍影响半径
    D_ATTRACT = 1000         # 吸引力饱和距离


class APFNavigator:
    """
    人工势场法导航器
    
    通过目标点吸引力和障碍物斥力的叠加，生成导航方向向量。
    """
    
    def __init__(self, config: APFConfig = None):
        self.config = config or APFConfig()
    
    def compute_force(self, robot_pos: np.ndarray, vision, goal: np.ndarray) -> np.ndarray:
        """
        计算势场合力
        
        Args:
            robot_pos: 机器人当前位置 [x, y]
            vision: 视觉信息对象
            goal: 目标位置 [x, y]
            
        Returns:
            合力向量 [fx, fy]
        """
        robot_pos = np.array(robot_pos)
        goal = np.array(goal)
        
        # 计算吸引力
        f_attract = self._compute_attractive_force(robot_pos, goal)
        
        # 计算斥力
        f_repel = self._compute_repulsive_force(robot_pos, vision)
        
        return f_attract + f_repel
    
    def _compute_attractive_force(self, robot_pos: np.ndarray, goal: np.ndarray) -> np.ndarray:
        """计算目标吸引力"""
        d_goal = np.linalg.norm(robot_pos - goal)
        
        if d_goal <= self.config.D_ATTRACT:
            # 近距离：线性吸引力
            return -2 * self.config.K_ATTRACT * (robot_pos - goal)
        else:
            # 远距离：饱和吸引力
            return -2 * self.config.K_ATTRACT * self.config.D_ATTRACT * (robot_pos - goal) / d_goal
    
    def _compute_repulsive_force(self, robot_pos: np.ndarray, vision) -> np.ndarray:
        """计算障碍物斥力"""
        f_repel_sum = np.array([0.0, 0.0])
        
        # 获取障碍物位置（黄方机器人）
        obstacles = self._get_obstacles(vision)
        
        for obs_pos in obstacles:
            d_obs = np.linalg.norm(robot_pos - obs_pos)
            
            if d_obs <= self.config.D_OBSTACLE and d_obs > 0:
                # 斥力方向：从障碍物指向机器人
                direction = (robot_pos - obs_pos) / d_obs
                # 斥力大小
                magnitude = self.config.K_REPEL * (1.0/d_obs - 1.0/self.config.D_OBSTACLE) / (d_obs ** 2)
                f_repel_sum += magnitude * direction
        
        return f_repel_sum
    
    def _get_obstacles(self, vision) -> list:
        """获取障碍物位置列表"""
        obstacles = []
        for i in range(16):
            if vision.yellow_robot[i].visible:
                obstacles.append(np.array([vision.yellow_robot[i].x, vision.yellow_robot[i].y]))
        return obstacles


# =============================================================================
# PRM 概率路网规划器
# =============================================================================

class PRMConfig:
    """PRM 参数配置"""
    N_SAMPLE = 50           # 采样点数量
    KNN = 10                # 每节点最大邻居数
    MAX_EDGE_LEN = 5000     # 边长上限
    ROBOT_SIZE = 200        # 机器人尺寸半径
    AVOID_DIST = 300        # 额外安全距离
    MIN_X = -4500
    MAX_X = 4500
    MIN_Y = -3000
    MAX_Y = 3000


class PRMNode:
    """PRM 路网节点"""
    def __init__(self, x: float, y: float, cost: float = 0.0, parent: int = -1):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent


class PRMPlanner:
    """
    PRM 概率路网规划器
    
    在自由空间中采样构建路网，使用 Dijkstra 算法搜索最短路径。
    """
    
    def __init__(self, config: PRMConfig = None):
        self.config = config or PRMConfig()
    
    def plan(self, vision, start: tuple, goal: tuple):
        """
        规划路径
        
        Args:
            vision: 视觉信息对象
            start: 起点坐标 (x, y)
            goal: 终点坐标 (x, y)
            
        Returns:
            path_x: 路径 x 坐标列表
            path_y: 路径 y 坐标列表
            road_map: 路网连接关系
            sample_x: 采样点 x 坐标列表
            sample_y: 采样点 y 坐标列表
        """
        start_x, start_y = start
        goal_x, goal_y = goal
        
        # 1. 提取障碍物
        obstacle_x, obstacle_y = self._extract_obstacles(vision)
        obs_tree = KDTree(np.vstack((obstacle_x, obstacle_y)).T)
        
        # 2. 采样自由空间
        sample_x, sample_y = self._sample_free_space(
            start_x, start_y, goal_x, goal_y,
            obstacle_x, obstacle_y, obs_tree
        )
        
        # 3. 构建路网
        road_map = self._build_roadmap(sample_x, sample_y, obs_tree)
        
        # 4. Dijkstra 搜索
        path_x, path_y = self._dijkstra_search(
            start_x, start_y, goal_x, goal_y,
            road_map, sample_x, sample_y
        )
        
        # 5. 备用搜索（主搜索失败时）
        if len(path_x) <= 2:
            path_x, path_y = self._fallback_search(
                start_x, start_y, goal_x, goal_y,
                sample_x, sample_y
            )
        
        return path_x, path_y, road_map, sample_x, sample_y
    
    def _extract_obstacles(self, vision) -> tuple:
        """提取障碍物位置"""
        obstacle_x = [-999999]
        obstacle_y = [-999999]
        
        # 蓝方机器人（排除自己）
        for robot in vision.blue_robot:
            if robot.visible and robot.id > 0:
                obstacle_x.append(robot.x)
                obstacle_y.append(robot.y)
        
        # 黄方机器人
        for robot in vision.yellow_robot:
            if robot.visible:
                obstacle_x.append(robot.x)
                obstacle_y.append(robot.y)
        
        return obstacle_x, obstacle_y
    
    def _sample_free_space(self, start_x, start_y, goal_x, goal_y,
                           obstacle_x, obstacle_y, obs_tree) -> tuple:
        """在自由空间中采样"""
        sample_x, sample_y = [], []
        k = 0.21  # 采样范围系数
        
        safe_dist = self.config.ROBOT_SIZE + self.config.AVOID_DIST
        
        while len(sample_x) < self.config.N_SAMPLE:
            # 随机选择采样中心
            i = random.randint(0, len(obstacle_x))
            
            if i == len(obstacle_x):
                # 以目标点为中心采样
                tx = ((2 * random.random() - 1) * k * (self.config.MAX_X - self.config.MIN_X)) + goal_x
                ty = ((2 * random.random() - 1) * k * (self.config.MAX_Y - self.config.MIN_Y)) + goal_y
            else:
                # 以障碍物为中心采样
                tx = ((2 * random.random() - 1) * k * (self.config.MAX_X - self.config.MIN_X)) + obstacle_x[i]
                ty = ((2 * random.random() - 1) * k * (self.config.MAX_Y - self.config.MIN_Y)) + obstacle_y[i]
            
            # 检查是否在自由空间
            distance, _ = obs_tree.query(np.array([tx, ty]))
            if distance >= safe_dist:
                sample_x.append(tx)
                sample_y.append(ty)
        
        # 添加起点和终点
        sample_x.extend([start_x, goal_x])
        sample_y.extend([start_y, goal_y])
        
        return sample_x, sample_y
    
    def _build_roadmap(self, sample_x, sample_y, obs_tree) -> list:
        """构建路网"""
        road_map = []
        n_sample = len(sample_x)
        sample_tree = KDTree(np.vstack((sample_x, sample_y)).T)
        
        for i, (ix, iy) in enumerate(zip(sample_x, sample_y)):
            distances, indices = sample_tree.query(np.array([ix, iy]), k=n_sample)
            edges = []
            
            for j in range(1, len(indices)):
                nx, ny = sample_x[indices[j]], sample_y[indices[j]]
                
                # 碰撞检测
                if not self._check_collision(ix, iy, nx, ny, obs_tree):
                    edges.append(indices[j])
                
                if len(edges) >= self.config.KNN:
                    break
            
            road_map.append(edges)
        
        return road_map
    
    def _check_collision(self, x1, y1, x2, y2, obs_tree) -> bool:
        """检查两点连线是否碰撞障碍物"""
        dx, dy = x2 - x1, y2 - y1
        distance = math.hypot(dx, dy)
        
        if distance > self.config.MAX_EDGE_LEN:
            return True
        
        angle = math.atan2(dy, dx)
        step_size = self.config.ROBOT_SIZE + self.config.AVOID_DIST
        steps = round(distance / step_size)
        
        x, y = x1, y1
        for _ in range(steps):
            dist, _ = obs_tree.query(np.array([x, y]))
            if dist <= step_size:
                return True
            x += step_size * math.cos(angle)
            y += step_size * math.sin(angle)
        
        # 检查终点
        dist, _ = obs_tree.query(np.array([x2, y2]))
        if dist <= step_size:
            return True
        
        return False
    
    def _dijkstra_search(self, start_x, start_y, goal_x, goal_y,
                         road_map, sample_x, sample_y) -> tuple:
        """Dijkstra 最短路径搜索"""
        path_x, path_y = [], []
        
        start = PRMNode(start_x, start_y, 0.0, -1)
        goal = PRMNode(goal_x, goal_y, 0.0, -1)
        
        open_set = {len(road_map) - 2: start}
        closed_set = {}
        
        goal_id = len(road_map) - 1
        path_found = False
        
        while open_set:
            # 取代价最小节点
            current_id = min(open_set, key=lambda k: open_set[k].cost)
            current = open_set[current_id]
            
            if current_id == goal_id:
                goal.cost = current.cost
                goal.parent = current.parent
                path_found = True
                break
            
            closed_set[current_id] = open_set.pop(current_id)
            
            # 扩展邻居
            for neighbor_id in road_map[current_id]:
                if neighbor_id in closed_set:
                    continue
                
                dx = sample_x[neighbor_id] - current.x
                dy = sample_y[neighbor_id] - current.y
                new_cost = current.cost + math.hypot(dx, dy)
                
                node = PRMNode(sample_x[neighbor_id], sample_y[neighbor_id],
                              new_cost, current_id)
                
                if neighbor_id in open_set:
                    if open_set[neighbor_id].cost > new_cost:
                        open_set[neighbor_id] = node
                else:
                    open_set[neighbor_id] = node
        
        # 回溯路径
        if path_found:
            path_x.append(goal.x)
            path_y.append(goal.y)
            parent = goal.parent
            
            while parent != -1:
                path_x.append(closed_set[parent].x)
                path_y.append(closed_set[parent].y)
                parent = closed_set[parent].parent
        
        return path_x, path_y
    
    def _fallback_search(self, start_x, start_y, goal_x, goal_y,
                         sample_x, sample_y) -> tuple:
        """备用搜索：找一个中间采样点"""
        path_x, path_y = [], []
        
        robot = np.array([start_x, start_y])
        goal = np.array([goal_x, goal_y])
        
        samples = np.array([[sample_x[i], sample_y[i]] for i in range(len(sample_x))])
        
        d_robot = np.array([np.linalg.norm(robot - s) for s in samples])
        d_goal = np.array([np.linalg.norm(goal - s) for s in samples])
        d_total = d_robot + d_goal
        
        # 找距离和最小的点
        min_idx = np.argmin(d_total)
        
        if (sample_x[min_idx] != goal_x and sample_y[min_idx] != goal_y and
            -4000 < sample_x[min_idx] < 4000 and -2500 < sample_y[min_idx] < 2500):
            path_x = [sample_x[min_idx], start_x]
            path_y = [sample_y[min_idx], start_y]
        
        return path_x, path_y


# =============================================================================
# 统一导航接口
# =============================================================================

class Navigator:
    """
    统一导航器
    
    整合 APF 局部导航和 PRM 脱困规划，提供统一接口。
    """
    
    # 目标点定义
    GOAL_A = np.array([-2400, -1500])
    GOAL_B = np.array([2400, 1500])
    
    # 状态常量
    STATE_NORMAL = 1       # 正常导航状态
    STATE_REVERSE = -1     # 反向导航状态
    STATE_ESCAPE = 0       # PRM 脱困状态
    
    def __init__(self, apf_config: APFConfig = None, prm_config: PRMConfig = None):
        self.apf = APFNavigator(apf_config)
        self.prm = PRMPlanner(prm_config)
        
        # 状态
        self.state = self.STATE_NORMAL
        self._saved_state = None
        self.escape_goal = None
        self.escape_threshold = 0
        
        # PRM 路径数据
        self.path_x = []
        self.path_y = []
    
    def get_current_goal(self) -> np.ndarray:
        """获取当前目标点"""
        if self.state == self.STATE_NORMAL:
            return self.GOAL_A
        elif self.state == self.STATE_REVERSE:
            return self.GOAL_B
        else:
            return self.escape_goal
    
    def compute_navigation_force(self, robot_pos: np.ndarray, vision) -> np.ndarray:
        """计算导航方向力"""
        goal = self.get_current_goal()
        return self.apf.compute_force(robot_pos, vision, goal)
    
    def switch_goal(self):
        """切换目标点"""
        self.state = -self.state
    
    def check_arrival(self, robot_pos: np.ndarray, threshold: float = 100) -> bool:
        """检查是否到达目标"""
        goal = self.get_current_goal()
        return np.linalg.norm(np.array(robot_pos) - goal) < threshold
    
    def check_stuck(self, robot_pos: np.ndarray, last_pos: np.ndarray,
                    stuck_count: int, threshold: float = 10,
                    min_count: int = 200) -> bool:
        """检查是否卡住"""
        distance = np.linalg.norm(np.array(robot_pos) - np.array(last_pos))
        return distance < threshold and stuck_count > min_count
    
    def start_escape(self, vision, robot_pos: np.ndarray):
        """
        启动脱困规划
        
        Returns:
            bool: 是否成功规划脱困路径
        """
        goal = self.get_current_goal()
        
        self.path_x, self.path_y, _, _, _ = self.prm.plan(
            vision=vision,
            start=(robot_pos[0], robot_pos[1]),
            goal=(goal[0], goal[1])
        )
        
        if len(self.path_x) >= 2:
            # 取倒数第二个点作为脱困中继点
            idx = len(self.path_x) - 2
            self.escape_goal = np.array([self.path_x[idx], self.path_y[idx]])
            
            # 计算脱困阈值
            self.escape_threshold = np.linalg.norm(robot_pos - self.escape_goal) / 2
            if self.escape_threshold < 300:
                self.escape_threshold = 999999
            elif self.escape_threshold < 1000:
                self.escape_threshold = 500
            
            # 保存当前状态
            self._saved_state = self.state
            self.state = self.STATE_ESCAPE
            
            return True
        
        return False
    
    def check_escape_complete(self, robot_pos: np.ndarray) -> bool:
        """检查脱困是否完成"""
        if self.state != self.STATE_ESCAPE:
            return True
        
        distance = np.linalg.norm(np.array(robot_pos) - self.escape_goal)
        return distance < self.escape_threshold
    
    def end_escape(self):
        """结束脱困状态"""
        if self._saved_state is not None:
            self.state = self._saved_state
            self._saved_state = None
        self.escape_goal = None
        self.path_x = []
        self.path_y = []
    
    def is_escaping(self) -> bool:
        """是否处于脱困状态"""
        return self.state == self.STATE_ESCAPE


# =============================================================================
# 工具函数
# =============================================================================

def vector_to_angle(vec: np.ndarray) -> float:
    """
    将二维向量转换为角度
    
    Args:
        vec: 二维向量 [x, y]
        
    Returns:
        角度（弧度，范围 [-pi, pi]）
    """
    return math.atan2(vec[1], vec[0])


def normalize_angle(angle: float) -> float:
    """
    将角度归一化到 [-pi, pi] 范围
    
    Args:
        angle: 输入角度（弧度）
        
    Returns:
        归一化后的角度
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle
