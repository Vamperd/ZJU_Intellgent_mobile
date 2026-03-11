"""
全局路径规划模块

包含:
- AStarPlanner: A* 确定性最优路径规划器（主要）
- RRTStarPlanner: RRT* 渐进最优路径规划器（备选）
- PathSmoother: 路径平滑器
- PathManager: 路径管理器（带路径一致性保持）
"""

import math
import random
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
from scipy.spatial import KDTree

from config import Config, config as global_config
from utils import Point, Circle, Path, euclidean_distance, circle_segment_collision


# =============================================================================
# RRT* 节点
# =============================================================================

@dataclass
class RRTNode:
    """RRT* 树节点"""
    x: float
    y: float
    cost: float = 0.0              # 从起点到该节点的代价
    parent: Optional['RRTNode'] = None
    children: List['RRTNode'] = field(default_factory=list)
    
    @property
    def point(self) -> Point:
        return Point(self.x, self.y)
    
    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    def distance_to(self, other: 'RRTNode') -> float:
        return math.hypot(self.x - other.x, self.y - other.y)
    
    def distance_to_point(self, point: Point) -> float:
        return math.hypot(self.x - point.x, self.y - point.y)


# =============================================================================
# A* 规划器
# =============================================================================

class AStarPlanner:
    """
    A* 确定性最优路径规划器
    
    特点:
    - 确定性结果：相同输入产生相同输出
    - 最优性保证：保证找到最短路径
    - 高效：启发式搜索效率高
    - 路径一致性：重规划时路径方向稳定
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        
        # 网格分辨率
        self.grid_resolution = 100
        self.diagonal_move = True
        
        # 场地范围
        self.minx = self.cfg.field.MIN_X
        self.maxx = self.cfg.field.MAX_X
        self.miny = self.cfg.field.MIN_Y
        self.maxy = self.cfg.field.MAX_Y
        
        # 网格尺寸
        self.grid_width = int((self.maxx - self.minx) / self.grid_resolution) + 1
        self.grid_height = int((self.maxy - self.miny) / self.grid_resolution) + 1
        
        self.last_path: Optional[Path] = None
        self.last_grid_map: Optional[np.ndarray] = None
    
    def plan(self, start: np.ndarray, goal: np.ndarray, 
             obstacles: List[Circle]) -> Optional[Path]:
        """
        规划从起点到终点的路径
        
        Args:
            start: 起点坐标 [x, y]
            goal: 终点坐标 [x, y]
            obstacles: 障碍物列表
            
        Returns:
            Path 对象，规划失败返回 None
        """
        # 提取障碍物坐标
        obstacle_x, obstacle_y = self._extract_obstacle_coords(obstacles)
        obstree = KDTree(np.vstack((obstacle_x, obstacle_y)).T)
        
        # 创建网格地图
        grid_map = self._create_grid_map(obstree)
        self.last_grid_map = grid_map
        
        # 转换坐标到网格索引
        start_grid = self._world_to_grid(start[0], start[1])
        goal_grid = self._world_to_grid(goal[0], goal[1])
        
        # A* 搜索
        path_grid = self._astar_search(grid_map, start_grid, goal_grid)
        
        if not path_grid:
            return None
        
        # 转换到世界坐标
        path_x, path_y = self._grid_path_to_world(path_grid)
        
        # 路径平滑
        path_x, path_y = self._smooth_path(path_x, path_y, obstree)
        
        # 路径插值
        path_x, path_y = self._interpolate_path(path_x, path_y, obstree, interval=400)
        
        # 创建 Path 对象
        points = [Point(x, y) for x, y in zip(path_x, path_y)]
        self.last_path = Path(points=points)
        
        return self.last_path
    
    def _extract_obstacle_coords(self, obstacles: List[Circle]) -> Tuple[List[float], List[float]]:
        """提取障碍物坐标"""
        obstacle_x = [-9999]
        obstacle_y = [-9999]
        
        for obs in obstacles:
            obstacle_x.append(obs.center.x)
            obstacle_y.append(obs.center.y)
        
        return obstacle_x, obstacle_y
    
    def _create_grid_map(self, obstree: KDTree) -> np.ndarray:
        """创建网格地图"""
        grid_map = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        inflate_radius = self.cfg.robot.RADIUS + self.cfg.robot.SAFE_DISTANCE
        
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                wx, wy = self._grid_to_world(j, i)
                dist, _ = obstree.query([wx, wy])
                if dist <= inflate_radius:
                    grid_map[i, j] = 1
        
        return grid_map
    
    def _world_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """世界坐标转网格索引"""
        gx = int((x - self.minx) / self.grid_resolution)
        gy = int((y - self.miny) / self.grid_resolution)
        gx = max(0, min(gx, self.grid_width - 1))
        gy = max(0, min(gy, self.grid_height - 1))
        return (gx, gy)
    
    def _grid_to_world(self, gx: int, gy: int) -> Tuple[float, float]:
        """网格索引转世界坐标"""
        x = self.minx + gx * self.grid_resolution + self.grid_resolution / 2
        y = self.miny + gy * self.grid_resolution + self.grid_resolution / 2
        return (x, y)
    
    def _grid_path_to_world(self, path_grid: List[Tuple[int, int]]) -> Tuple[List[float], List[float]]:
        """网格路径转世界坐标"""
        path_x, path_y = [], []
        for (gx, gy) in path_grid:
            x, y = self._grid_to_world(gx, gy)
            path_x.append(x)
            path_y.append(y)
        return path_x, path_y
    
    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """启发式函数：欧几里得距离"""
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def _get_neighbors(self, current: Tuple[int, int], 
                       grid_map: np.ndarray) -> List[Tuple[int, int]]:
        """获取邻居节点"""
        neighbors = []
        x, y = current
        
        if self.diagonal_move:
            directions = [
                (1, 0), (-1, 0), (0, 1), (0, -1),
                (1, 1), (1, -1), (-1, 1), (-1, -1)
            ]
        else:
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                if grid_map[ny, nx] == 0:
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def _get_move_cost(self, current: Tuple[int, int], 
                       neighbor: Tuple[int, int]) -> float:
        """计算移动代价"""
        dx = abs(neighbor[0] - current[0])
        dy = abs(neighbor[1] - current[1])
        return math.sqrt(2) if dx + dy == 2 else 1.0
    
    def _astar_search(self, grid_map: np.ndarray, 
                      start: Tuple[int, int], 
                      goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* 搜索核心"""
        import heapq
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        visited = set()
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current == goal:
                # 重建路径
                path = [current]
                while current in came_from:
                    current = came_from[current]
                    path.append(current)
                path.reverse()
                return path
            
            if current in visited:
                continue
            visited.add(current)
            
            for neighbor in self._get_neighbors(current, grid_map):
                if neighbor in visited:
                    continue
                
                tentative_g = g_score[current] + self._get_move_cost(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []  # 未找到路径
    
    def _smooth_path(self, path_x: List[float], path_y: List[float],
                     obstree: KDTree) -> Tuple[List[float], List[float]]:
        """路径平滑：剪枝优化"""
        if len(path_x) <= 2:
            return path_x, path_y
        
        smoothed_x, smoothed_y = [path_x[0]], [path_y[0]]
        i = 0
        
        while i < len(path_x) - 1:
            for j in range(len(path_x) - 1, i, -1):
                if not self._check_line_collision(path_x[i], path_y[i],
                                                   path_x[j], path_y[j], obstree):
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
    
    def _check_line_collision(self, x1: float, y1: float, 
                               x2: float, y2: float,
                               obstree: KDTree) -> bool:
        """检查线段是否与障碍物碰撞"""
        dist = math.hypot(x2 - x1, y2 - y1)
        angle = math.atan2(y2 - y1, x2 - x1)
        
        step_size = self.cfg.robot.RADIUS + self.cfg.robot.SAFE_DISTANCE
        steps = int(dist / step_size) + 1
        
        x, y = x1, y1
        safe_dist = self.cfg.robot.RADIUS + self.cfg.robot.SAFE_DISTANCE
        
        for _ in range(steps):
            distance, _ = obstree.query([x, y])
            if distance <= safe_dist:
                return True
            x += step_size * math.cos(angle)
            y += step_size * math.sin(angle)
        
        distance, _ = obstree.query([x2, y2])
        return distance <= safe_dist
    
    def _interpolate_path(self, path_x: List[float], path_y: List[float],
                          obstree: KDTree, interval: float = 400) -> Tuple[List[float], List[float]]:
        """路径插值：增加路径点密度"""
        if len(path_x) <= 1:
            return path_x, path_y
        
        new_x, new_y = [path_x[0]], [path_y[0]]
        safe_dist = self.cfg.robot.RADIUS + self.cfg.robot.SAFE_DISTANCE
        
        for i in range(len(path_x) - 1):
            x1, y1 = path_x[i], path_y[i]
            x2, y2 = path_x[i + 1], path_y[i + 1]
            
            dist = math.hypot(x2 - x1, y2 - y1)
            if dist > interval:
                num_points = int(dist / interval)
                for j in range(1, num_points + 1):
                    ratio = j / (num_points + 1)
                    interp_x = x1 + ratio * (x2 - x1)
                    interp_y = y1 + ratio * (y2 - y1)
                    
                    col_dist, _ = obstree.query([interp_x, interp_y])
                    if col_dist >= safe_dist:
                        new_x.append(interp_x)
                        new_y.append(interp_y)
            
            new_x.append(x2)
            new_y.append(y2)
        
        return new_x, new_y


# =============================================================================
# RRT* 规划器 (备选)
# =============================================================================

class RRTStarPlanner:
    """
    RRT* 渐进最优路径规划器
    
    特点:
    - 概率完备性: 能找到解的概率随迭代次数增加
    - 渐进最优性: 路径质量随迭代次数提升
    - 支持动态重规划
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        self.nodes: List[RRTNode] = []
        self.last_path: Optional[Path] = None
    
    def plan(self, start: np.ndarray, goal: np.ndarray, 
             obstacles: List[Circle]) -> Optional[Path]:
        """
        规划从起点到终点的路径
        
        Args:
            start: 起点坐标 [x, y]
            goal: 终点坐标 [x, y]
            obstacles: 障碍物列表
            
        Returns:
            Path 对象，规划失败返回 None
        """
        # 初始化树
        start_node = RRTNode(x=start[0], y=start[1], cost=0.0)
        goal_node = RRTNode(x=goal[0], y=goal[1])
        self.nodes = [start_node]
        
        # RRT* 主循环
        for _ in range(self.cfg.rrt.MAX_ITERATIONS):
            # 1. 采样
            sample = self._sample(goal)
            
            # 2. 找最近节点
            nearest = self._find_nearest(sample)
            
            # 3. 向采样点扩展
            new_node = self._steer(nearest, sample)
            
            if new_node is None:
                continue
            
            # 4. 碰撞检测
            if self._check_collision(nearest.position, new_node.position, obstacles):
                continue
            
            # 5. 找近邻节点
            near_nodes = self._find_near_nodes(new_node)
            
            # 6. 选择最优父节点
            best_parent = self._choose_best_parent(new_node, near_nodes, obstacles)
            if best_parent is None:
                best_parent = nearest
            
            # 添加新节点到树
            new_node.parent = best_parent
            new_node.cost = best_parent.cost + best_parent.distance_to(new_node)
            best_parent.children.append(new_node)
            self.nodes.append(new_node)
            
            # 7. 重连优化
            if self.cfg.rrt.REWIRE_ENABLED:
                self._rewire(new_node, near_nodes, obstacles)
            
            # 8. 检查是否到达目标
            if new_node.distance_to(goal_node) <= self.cfg.rrt.GOAL_THRESHOLD:
                # 尝试直接连接目标
                if not self._check_collision(new_node.position, goal, obstacles):
                    goal_node.parent = new_node
                    goal_node.cost = new_node.cost + new_node.distance_to(goal_node)
                    self.nodes.append(goal_node)
                    path = self._extract_path(goal_node)
                    self.last_path = self._smooth_path(path, obstacles)
                    return self.last_path
        
        # 未找到路径，返回最接近目标的路径
        return self._get_best_partial_path(goal)
    
    def replan(self, current_pos: np.ndarray, goal: np.ndarray,
               obstacles: List[Circle]) -> Optional[Path]:
        """
        动态重规划
        
        Args:
            current_pos: 当前位置
            goal: 目标位置
            obstacles: 障碍物列表
            
        Returns:
            新路径
        """
        return self.plan(current_pos, goal, obstacles)
    
    # -------------------------------------------------------------------------
    # 内部方法
    # -------------------------------------------------------------------------
    
    def _sample(self, goal: np.ndarray) -> Point:
        """采样一个点"""
        # 目标偏置采样
        if random.random() < self.cfg.rrt.GOAL_BIAS:
            return Point(goal[0], goal[1])
        
        # 随机采样
        x = random.uniform(self.cfg.field.MIN_X, self.cfg.field.MAX_X)
        y = random.uniform(self.cfg.field.MIN_Y, self.cfg.field.MAX_Y)
        return Point(x, y)
    
    def _find_nearest(self, point: Point) -> RRTNode:
        """找到距离采样点最近的节点"""
        min_dist = float('inf')
        nearest = None
        
        for node in self.nodes:
            dist = node.distance_to_point(point)
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    def _steer(self, from_node: RRTNode, to_point: Point) -> Optional[RRTNode]:
        """从 from_node 向 to_point 扩展一步"""
        dx = to_point.x - from_node.x
        dy = to_point.y - from_node.y
        dist = math.hypot(dx, dy)
        
        if dist < 1e-6:
            return None
        
        # 限制步长
        step = min(dist, self.cfg.rrt.STEP_SIZE)
        
        new_x = from_node.x + step * dx / dist
        new_y = from_node.y + step * dy / dist
        
        # 边界检查
        if not self._is_in_field(new_x, new_y):
            return None
        
        return RRTNode(x=new_x, y=new_y)
    
    def _is_in_field(self, x: float, y: float) -> bool:
        """检查点是否在场地内"""
        return (self.cfg.field.MIN_X < x < self.cfg.field.MAX_X and
                self.cfg.field.MIN_Y < y < self.cfg.field.MAX_Y)
    
    def _find_near_nodes(self, node: RRTNode) -> List[RRTNode]:
        """找到节点附近的邻居节点"""
        # 动态搜索半径
        n = len(self.nodes) + 1
        radius = min(self.cfg.rrt.SEARCH_RADIUS,
                     self.cfg.rrt.STEP_SIZE * math.sqrt(math.log(n) / n))
        
        near_nodes = []
        for other in self.nodes:
            if other.distance_to(node) <= radius:
                near_nodes.append(other)
        
        return near_nodes
    
    def _choose_best_parent(self, new_node: RRTNode, 
                            near_nodes: List[RRTNode],
                            obstacles: List[Circle]) -> Optional[RRTNode]:
        """选择最优父节点"""
        best_parent = None
        min_cost = float('inf')
        
        for near in near_nodes:
            if not self._check_collision(near.position, new_node.position, obstacles):
                cost = near.cost + near.distance_to(new_node)
                if cost < min_cost:
                    min_cost = cost
                    best_parent = near
        
        return best_parent
    
    def _rewire(self, new_node: RRTNode, 
                near_nodes: List[RRTNode],
                obstacles: List[Circle]):
        """重连优化：检查是否可以通过新节点降低邻居节点的代价"""
        for near in near_nodes:
            # 跳过父节点
            if near == new_node.parent:
                continue
            
            # 检查是否可以改善
            potential_cost = new_node.cost + new_node.distance_to(near)
            if potential_cost < near.cost:
                # 碰撞检测
                if not self._check_collision(new_node.position, near.position, obstacles):
                    # 重连
                    if near.parent:
                        near.parent.children.remove(near)
                    near.parent = new_node
                    near.cost = potential_cost
                    new_node.children.append(near)
    
    def _check_collision(self, p1: np.ndarray, p2: np.ndarray,
                         obstacles: List[Circle]) -> bool:
        """检查两点连线和障碍物是否碰撞"""
        for obs in obstacles:
            if circle_segment_collision(
                np.array([obs.center.x, obs.center.y]),
                obs.radius,
                p1, p2
            ):
                return True
        return False
    
    def _extract_path(self, goal_node: RRTNode) -> Path:
        """从目标节点回溯提取路径"""
        points = []
        node = goal_node
        
        while node is not None:
            points.append(Point(node.x, node.y))
            node = node.parent
        
        points.reverse()
        return Path(points=points)
    
    def _get_best_partial_path(self, goal: np.ndarray) -> Optional[Path]:
        """获取最接近目标的部分路径"""
        if not self.nodes:
            return None
        
        # 找距离目标最近的节点
        best_node = min(self.nodes, 
                        key=lambda n: math.hypot(n.x - goal[0], n.y - goal[1]))
        
        # 提取到该节点的路径
        path = self._extract_path(best_node)
        # 添加目标点
        path.points.append(Point(goal[0], goal[1]))
        
        return path
    
    def _smooth_path(self, path: Path, obstacles: List[Circle]) -> Path:
        """路径平滑"""
        if len(path.points) <= 2:
            return path
        
        smoothed_points = list(path.points)
        
        for _ in range(self.cfg.rrt.SMOOTH_ITERATIONS):
            if len(smoothed_points) <= 2:
                break
            
            # 随机选择两个点
            i = random.randint(0, len(smoothed_points) - 2)
            j = random.randint(i + 1, len(smoothed_points) - 1)
            
            # 尝试直接连接
            p1 = smoothed_points[i].to_array()
            p2 = smoothed_points[j].to_array()
            
            if not self._check_collision(p1, p2, obstacles):
                # 移除中间点
                smoothed_points = smoothed_points[:i+1] + smoothed_points[j:]
        
        return Path(points=smoothed_points)


# =============================================================================
# 路径管理器（改进版：路径一致性保持）
# =============================================================================

class PathManager:
    """
    路径管理器
    
    负责路径的存储、更新和查询
    特点：
    - 路径一致性保持：重规划时尽量保持原路径方向
    - 路径验证：检测当前路径是否被障碍物阻挡
    - 渐进更新：只在必要时更新路径
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        # 使用 A* 作为主要规划器（确定性，路径一致）
        self.planner = AStarPlanner(self.cfg)
        self.rrt_planner = RRTStarPlanner(self.cfg)  # 备选
        
        self.current_path: Optional[Path] = None
        self.current_waypoint_index: int = 0
        
        # 路径一致性相关
        self.last_goal: Optional[np.ndarray] = None
        self.path_blocked: bool = False
    
    def plan_new_path(self, start: np.ndarray, goal: np.ndarray,
                      obstacles: List[Circle]) -> bool:
        """规划新路径"""
        self.current_path = self.planner.plan(start, goal, obstacles)
        self.current_waypoint_index = 0
        self.last_goal = goal.copy()
        self.path_blocked = False
        return self.current_path is not None
    
    def replan_path(self, current_pos: np.ndarray, goal: np.ndarray,
                    obstacles: List[Circle]) -> bool:
        """
        重规划路径
        
        策略：保持路径一致性，只有在必要时才完全重规划
        """
        # 检查目标是否改变
        if self.last_goal is not None:
            goal_dist = np.linalg.norm(goal - self.last_goal)
            if goal_dist < 10:  # 目标没变
                # 检查当前路径是否仍然有效
                if self._validate_path(obstacles):
                    # 路径有效，检查是否需要从当前位置重新规划
                    if self._should_replan_from_current(current_pos, obstacles):
                        # 从当前位置规划到最近的路径点
                        return self._partial_replan(current_pos, obstacles)
                    return True
        
        # 目标改变或路径无效，完全重规划
        return self.plan_new_path(current_pos, goal, obstacles)
    
    def _validate_path(self, obstacles: List[Circle]) -> bool:
        """验证当前路径是否与障碍物碰撞"""
        if self.current_path is None or self.current_path.is_empty:
            return False
        
        self.path_blocked = False
        
        # 只检查剩余路径
        remaining = self.get_remaining_path()
        if remaining is None or remaining.is_empty:
            return True
        
        # 检查每个路径段
        for i in range(len(remaining) - 1):
            p1 = remaining[i].to_array()
            p2 = remaining[i + 1].to_array()
            
            for obs in obstacles:
                if circle_segment_collision(
                    np.array([obs.center.x, obs.center.y]),
                    obs.radius,
                    p1, p2
                ):
                    self.path_blocked = True
                    return False
        
        return True
    
    def _should_replan_from_current(self, current_pos: np.ndarray,
                                     obstacles: List[Circle]) -> bool:
        """判断是否需要从当前位置重新规划"""
        if self.current_path is None:
            return True
        
        # 检查当前位置是否偏离路径太远
        remaining = self.get_remaining_path()
        if remaining is None or remaining.is_empty:
            return False
        
        # 找最近的路径点
        min_dist = float('inf')
        for point in remaining:
            dist = euclidean_distance(current_pos, point.to_array())
            min_dist = min(min_dist, dist)
        
        # 偏离超过阈值，需要重规划
        return min_dist > 500
    
    def _partial_replan(self, current_pos: np.ndarray,
                        obstacles: List[Circle]) -> bool:
        """部分重规划：从当前位置连接到原路径"""
        remaining = self.get_remaining_path()
        if remaining is None or remaining.is_empty:
            return False
        
        # 找到原路径上最远可达的点
        best_point = remaining.end
        best_idx = len(remaining) - 1
        
        for i, point in enumerate(remaining):
            # 检查是否能直线连接
            can_connect = True
            for obs in obstacles:
                if circle_segment_collision(
                    np.array([obs.center.x, obs.center.y]),
                    obs.radius,
                    current_pos,
                    point.to_array()
                ):
                    can_connect = False
                    break
            
            if can_connect:
                best_point = point
                best_idx = i
                break
        
        # 如果能连接到较远的点，更新路径索引
        if best_idx > 0:
            self.current_waypoint_index += best_idx
            return True
        
        # 否则需要完全重规划
        return self.plan_new_path(current_pos, self.last_goal, obstacles)
    
    def check_path_blocked(self, robot_pos: np.ndarray, 
                           obstacles: List[Circle]) -> bool:
        """检查前方路径是否被阻挡"""
        if self.current_path is None:
            return False
        
        # 检查前方一定范围内的路径
        check_distance = 1000  # 检查前方 1m
        
        remaining = self.get_remaining_path()
        if remaining is None or remaining.is_empty:
            return False
        
        checked_dist = 0
        for i in range(len(remaining) - 1):
            p1 = remaining[i].to_array()
            p2 = remaining[i + 1].to_array()
            segment_dist = euclidean_distance(p1, p2)
            
            for obs in obstacles:
                if circle_segment_collision(
                    np.array([obs.center.x, obs.center.y]),
                    obs.radius,
                    p1, p2
                ):
                    return True
            
            checked_dist += segment_dist
            if checked_dist > check_distance:
                break
        
        return False
    
    def get_current_waypoint(self) -> Optional[Point]:
        """获取当前目标路径点"""
        if self.current_path is None:
            return None
        if self.current_waypoint_index >= len(self.current_path):
            return None
        return self.current_path[self.current_waypoint_index]
    
    def advance_waypoint(self) -> bool:
        """前进到下一个路径点"""
        if self.current_path is None:
            return False
        if self.current_waypoint_index < len(self.current_path) - 1:
            self.current_waypoint_index += 1
            return True
        return False
    
    def is_path_complete(self) -> bool:
        """是否已完成路径"""
        if self.current_path is None:
            return True
        return self.current_waypoint_index >= len(self.current_path)
    
    def get_remaining_path(self) -> Optional[Path]:
        """获取剩余路径"""
        if self.current_path is None:
            return None
        remaining_points = self.current_path.points[self.current_waypoint_index:]
        return Path(points=remaining_points)
    
    def get_path_arrays(self) -> Tuple[List[float], List[float]]:
        """获取路径坐标数组（用于可视化）"""
        if self.current_path is None:
            return [], []
        return self.current_path.to_arrays()
    
    def get_remaining_path_arrays(self) -> Tuple[List[float], List[float]]:
        """获取剩余路径坐标数组"""
        remaining = self.get_remaining_path()
        if remaining is None:
            return [], []
        return remaining.to_arrays()