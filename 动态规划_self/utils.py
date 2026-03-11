"""
通用工具函数模块

包含几何计算、角度处理、数据结构等通用工具。
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# 数据结构
# =============================================================================

@dataclass
class Point:
    """二维点"""
    x: float
    y: float
    
    def to_array(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'Point':
        return cls(x=arr[0], y=arr[1])
    
    def distance_to(self, other: 'Point') -> float:
        return math.hypot(self.x - other.x, self.y - other.y)
    
    def __eq__(self, other):
        if isinstance(other, Point):
            return abs(self.x - other.x) < 1e-6 and abs(self.y - other.y) < 1e-6
        return False


@dataclass
class Circle:
    """圆形障碍物"""
    center: Point
    radius: float
    
    def contains(self, point: Point) -> bool:
        return self.center.distance_to(point) <= self.radius


@dataclass
class Path:
    """路径数据结构"""
    points: List[Point]
    
    def __len__(self) -> int:
        return len(self.points)
    
    def __getitem__(self, index: int) -> Point:
        return self.points[index]
    
    def __iter__(self):
        return iter(self.points)
    
    @property
    def is_empty(self) -> bool:
        return len(self.points) == 0
    
    @property
    def start(self) -> Optional[Point]:
        return self.points[0] if self.points else None
    
    @property
    def end(self) -> Optional[Point]:
        return self.points[-1] if self.points else None
    
    @classmethod
    def from_arrays(cls, x_list: List[float], y_list: List[float]) -> 'Path':
        points = [Point(x, y) for x, y in zip(x_list, y_list)]
        return cls(points=points)
    
    def to_arrays(self) -> Tuple[List[float], List[float]]:
        x_list = [p.x for p in self.points]
        y_list = [p.y for p in self.points]
        return x_list, y_list
    
    def total_length(self) -> float:
        """计算路径总长度"""
        if len(self.points) < 2:
            return 0.0
        length = 0.0
        for i in range(len(self.points) - 1):
            length += self.points[i].distance_to(self.points[i + 1])
        return length


# =============================================================================
# 角度工具函数
# =============================================================================

def normalize_angle(angle: float) -> float:
    """
    将角度归一化到 [-pi, pi] 范围
    
    Args:
        angle: 输入角度 (弧度)
        
    Returns:
        归一化后的角度
    """
    while angle > math.pi:
        angle -= 2 * math.pi
    while angle < -math.pi:
        angle += 2 * math.pi
    return angle


def angle_difference(angle1: float, angle2: float) -> float:
    """
    计算两个角度的最短差值
    
    Args:
        angle1: 角度1 (弧度)
        angle2: 角度2 (弧度)
        
    Returns:
        归一化的角度差 [-pi, pi]
    """
    return normalize_angle(angle1 - angle2)


def vector_to_angle(vec: np.ndarray) -> float:
    """
    将二维向量转换为角度
    
    Args:
        vec: 二维向量 [x, y]
        
    Returns:
        角度 (弧度, 范围 [-pi, pi])
    """
    return math.atan2(vec[1], vec[0])


def angle_to_vector(angle: float, magnitude: float = 1.0) -> np.ndarray:
    """
    将角度转换为单位向量
    
    Args:
        angle: 角度 (弧度)
        magnitude: 向量模长
        
    Returns:
        二维向量 [x, y]
    """
    return np.array([magnitude * math.cos(angle), magnitude * math.sin(angle)])


# =============================================================================
# 几何计算
# =============================================================================

def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    计算两点间的欧氏距离
    
    Args:
        p1: 点1 [x, y]
        p2: 点2 [x, y]
        
    Returns:
        欧氏距离
    """
    return np.linalg.norm(np.array(p1) - np.array(p2))


def point_to_segment_distance(point: np.ndarray, 
                              seg_start: np.ndarray, 
                              seg_end: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    计算点到线段的最短距离
    
    Args:
        point: 点坐标
        seg_start: 线段起点
        seg_end: 线段终点
        
    Returns:
        (距离, 最近点)
    """
    point = np.array(point)
    seg_start = np.array(seg_start)
    seg_end = np.array(seg_end)
    
    segment = seg_end - seg_start
    segment_length_sq = np.dot(segment, segment)
    
    if segment_length_sq < 1e-10:
        return euclidean_distance(point, seg_start), seg_start
    
    t = max(0, min(1, np.dot(point - seg_start, segment) / segment_length_sq))
    closest_point = seg_start + t * segment
    distance = euclidean_distance(point, closest_point)
    
    return distance, closest_point


def circle_segment_collision(circle_center: np.ndarray, 
                             radius: float,
                             seg_start: np.ndarray, 
                             seg_end: np.ndarray) -> bool:
    """
    检查圆与线段是否碰撞
    
    Args:
        circle_center: 圆心
        radius: 圆半径
        seg_start: 线段起点
        seg_end: 线段终点
        
    Returns:
        是否碰撞
    """
    distance, _ = point_to_segment_distance(circle_center, seg_start, seg_end)
    return distance <= radius


def circles_collision(center1: np.ndarray, radius1: float,
                      center2: np.ndarray, radius2: float) -> bool:
    """
    检查两个圆是否碰撞
    
    Args:
        center1: 圆心1
        radius1: 半径1
        center2: 圆心2
        radius2: 半径2
        
    Returns:
        是否碰撞
    """
    return euclidean_distance(center1, center2) <= radius1 + radius2


# =============================================================================
# 障碍物提取
# =============================================================================

def extract_obstacles(vision, include_teammates: bool = True) -> List[Circle]:
    """
    从视觉数据中提取障碍物
    
    Args:
        vision: 视觉对象
        include_teammates: 是否包含队友
        
    Returns:
        障碍物列表 (圆形)
    """
    from config import config
    
    obstacles = []
    safe_radius = config.robot.RADIUS + config.robot.SAFE_DISTANCE
    
    # 黄方机器人 (对手)
    for robot in vision.yellow_robot:
        if robot.visible:
            obstacles.append(Circle(
                center=Point(robot.x, robot.y),
                radius=safe_radius
            ))
    
    # 蓝方机器人 (队友，排除自己)
    if include_teammates:
        for robot in vision.blue_robot:
            if robot.visible and robot.id > 0:
                obstacles.append(Circle(
                    center=Point(robot.x, robot.y),
                    radius=safe_radius
                ))
    
    return obstacles


def extract_dynamic_obstacles(vision) -> List[Tuple[Circle, np.ndarray]]:
    """
    从视觉数据中提取动态障碍物及其速度
    
    Args:
        vision: 视觉对象
        
    Returns:
        障碍物列表 [(障碍物, 速度向量), ...]
    """
    from config import config
    
    obstacles_with_velocity = []
    safe_radius = config.robot.RADIUS + config.robot.SAFE_DISTANCE
    
    # 黄方机器人
    for robot in vision.yellow_robot:
        if robot.visible:
            circle = Circle(
                center=Point(robot.x, robot.y),
                radius=safe_radius
            )
            velocity = np.array([robot.vel_x, robot.vel_y])
            obstacles_with_velocity.append((circle, velocity))
    
    # 蓝方队友
    for robot in vision.blue_robot:
        if robot.visible and robot.id > 0:
            circle = Circle(
                center=Point(robot.x, robot.y),
                radius=safe_radius
            )
            velocity = np.array([robot.vel_x, robot.vel_y])
            obstacles_with_velocity.append((circle, velocity))
    
    return obstacles_with_velocity


# =============================================================================
# 轨迹预测
# =============================================================================

def predict_position(pos: np.ndarray, vel: np.ndarray, dt: float) -> np.ndarray:
    """
    预测未来位置 (匀速模型)
    
    Args:
        pos: 当前位置
        vel: 当前速度
        dt: 时间间隔
        
    Returns:
        预测位置
    """
    return pos + vel * dt


def predict_trajectory(pos: np.ndarray, heading: float,
                       vx: float, vw: float,
                       dt: float, steps: int) -> List[np.ndarray]:
    """
    预测机器人的运动轨迹
    
    Args:
        pos: 当前位置 [x, y]
        heading: 当前朝向 (弧度)
        vx: 前向速度
        vw: 角速度
        dt: 时间步长
        steps: 预测步数
        
    Returns:
        轨迹点列表 [[x, y], ...]
    """
    trajectory = []
    x, y, theta = pos[0], pos[1], heading
    
    for _ in range(steps):
        # 更新朝向
        theta += vw * dt
        # 更新位置 (前向运动)
        x += vx * math.cos(theta) * dt
        y += vx * math.sin(theta) * dt
        trajectory.append(np.array([x, y]))
    
    return trajectory


# =============================================================================
# 速度限制
# =============================================================================

def clamp_velocity(vx: float, vy: float, vw: float,
                   max_vx: float, max_vy: float, max_vw: float) -> Tuple[float, float, float]:
    """
    限制速度在允许范围内
    
    Args:
        vx, vy, vw: 输入速度
        max_vx, max_vy, max_vw: 最大速度
        
    Returns:
        限制后的速度 (vx, vy, vw)
    """
    vx = np.clip(vx, -max_vx, max_vx)
    vy = np.clip(vy, -max_vy, max_vy)
    vw = np.clip(vw, -max_vw, max_vw)
    return vx, vy, vw


def apply_acceleration_limit(v_current: float, v_target: float,
                             max_accel: float, dt: float) -> float:
    """
    应用加速度限制
    
    Args:
        v_current: 当前速度
        v_target: 目标速度
        max_accel: 最大加速度
        dt: 时间间隔
        
    Returns:
        限制后的速度
    """
    delta = v_target - v_current
    max_delta = max_accel * dt
    
    if abs(delta) > max_delta:
        return v_current + np.sign(delta) * max_delta
    return v_target
