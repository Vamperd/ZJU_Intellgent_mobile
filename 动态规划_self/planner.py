"""
全局路径规划模块

包含:
- AStarPlanner: A* 全局路径规划器
- PathSmoother: 路径平滑器
"""

import math
import random
import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import heapq

from config import Config, config as global_config
from utils import Point, Circle, Path, euclidean_distance, circle_segment_collision


# =============================================================================
# A* 节点
# =============================================================================

@dataclass
class AStarNode:
    """A* 树节点"""
    x: float
    y: float
    g_cost: float = 0.0              # 从起点到该节点的代价
    h_cost: float = 0.0              # 从该节点到目标的启发式代价
    parent: Optional['AStarNode'] = None
    
    @property
    def f_cost(self) -> float:
        return self.g_cost + self.h_cost
    
    @property
    def point(self) -> Point:
        return Point(self.x, self.y)
    
    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    def __lt__(self, other: 'AStarNode') -> bool:
        return self.f_cost < other.f_cost
    
    def __eq__(self, other: 'AStarNode') -> bool:
        if isinstance(other, AStarNode):
            return abs(self.x - other.x) < 1e-3 and abs(self.y - other.y) < 1e-3
        return False
    
    def __hash__(self) -> int:
        return hash((round(self.x, 1), round(self.y, 1)))
    
    def distance_to(self, other: 'AStarNode') -> float:
        return math.hypot(self.x - other.x, self.y - other.y)
    
    def distance_to_point(self, point: Point) -> float:
        return math.hypot(self.x - point.x, self.y - point.y)


# =============================================================================
# A* 规划器
# =============================================================================

class AStarPlanner:
    """
    A* 全局路径规划器
    
    特点:
    - 启发式搜索，效率高
    - 可保证最优路径
    - 适合静态环境
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        self.last_path: Optional[Path] = None
        self.grid_resolution = 150.0
        self.visited_cells = set()
    
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
        start_node = AStarNode(x=start[0], y=start[1])
        goal_node = AStarNode(x=goal[0], y=goal[1])
        
        open_set = []
        closed_set = set()
        heapq.heappush(open_set, start_node)
        
        came_from = {}
        g_score = {start_node: 0.0}
        
        iterations = 0
        max_iterations = 5000
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current = heapq.heappop(open_set)
            
            if current.distance_to(goal_node) <= self.cfg.astar.GOAL_THRESHOLD:
                path = self._reconstruct_path(current, came_from)
                self.last_path = self._smooth_path(path, obstacles)
                return self.last_path
            
            closed_set.add(current)
            
            neighbors = self._get_neighbors(current)
            
            for neighbor in neighbors:
                if neighbor in closed_set:
                    continue
                
                if self._check_collision_node(neighbor, obstacles):
                    continue
                
                tentative_g = current.g_cost + current.distance_to(neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    neighbor.g_cost = tentative_g
                    neighbor.h_cost = neighbor.distance_to(goal_node)
                    
                    if neighbor not in open_set:
                        heapq.heappush(open_set, neighbor)
        
        return None
    
    def _get_neighbors(self, node: AStarNode) -> List[AStarNode]:
        """获取邻居节点"""
        neighbors = []
        resolution = self.grid_resolution
        
        directions = [
            (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]
        
        for dx, dy in directions:
            new_x = node.x + dx * resolution
            new_y = node.y + dy * resolution
            
            if self._is_in_field(new_x, new_y):
                neighbor = AStarNode(x=new_x, y=new_y)
                neighbors.append(neighbor)
        
        return neighbors
    
    def _is_in_field(self, x: float, y: float) -> bool:
        """检查点是否在场地内"""
        margin = 100
        return (self.cfg.field.MIN_X + margin < x < self.cfg.field.MAX_X - margin and
                self.cfg.field.MIN_Y + margin < y < self.cfg.field.MAX_Y - margin)
    
    def _check_collision_node(self, node: AStarNode, obstacles: List[Circle]) -> bool:
        """检查节点是否与障碍物碰撞"""
        node_pos = node.position
        for obs in obstacles:
            dist = euclidean_distance(node_pos, [obs.center.x, obs.center.y])
            if dist < obs.radius + self.grid_resolution * 0.5:
                return True
        return False
    
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
    
    def _reconstruct_path(self, current: AStarNode, 
                         came_from: dict) -> Path:
        """重建路径"""
        points = []
        node = current
        
        while node in came_from:
            points.append(Point(node.x, node.y))
            node = came_from[node]
        
        points.append(Point(node.x, node.y))
        points.reverse()
        return Path(points=points)
    
    def _smooth_path(self, path: Path, obstacles: List[Circle]) -> Path:
        """路径平滑 - 使用随机优化"""
        if len(path.points) <= 2:
            return path
        
        smoothed_points = list(path.points)
        
        for _ in range(self.cfg.astar.SMOOTH_ITERATIONS):
            if len(smoothed_points) <= 2:
                break
            
            i = random.randint(0, len(smoothed_points) - 2)
            j = random.randint(i + 1, len(smoothed_points) - 1)
            
            p1 = smoothed_points[i].to_array()
            p2 = smoothed_points[j].to_array()
            
            if not self._check_collision(p1, p2, obstacles):
                smoothed_points = smoothed_points[:i+1] + smoothed_points[j:]
        
        return Path(points=smoothed_points)


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
# RRT* 规划器
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
# 路径管理器
# =============================================================================

class PathManager:
    """
    路径管理器
    
    负责路径的存储、更新和查询
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        self.planner = AStarPlanner(self.cfg)
        self.current_path: Optional[Path] = None
        self.current_waypoint_index: int = 0
    
    def plan_new_path(self, start: np.ndarray, goal: np.ndarray,
                      obstacles: List[Circle]) -> bool:
        """规划新路径"""
        self.current_path = self.planner.plan(start, goal, obstacles)
        self.current_waypoint_index = 0
        return self.current_path is not None
    
    def replan_path(self, current_pos: np.ndarray, goal: np.ndarray,
                    obstacles: List[Circle]) -> bool:
        """重规划路径"""
        return self.plan_new_path(current_pos, goal, obstacles)
    
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
