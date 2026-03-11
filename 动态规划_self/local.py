"""
局部路径规划模块

包含:
- DWAPlanner: DWA 动态窗口法局部规划器
- VelocityWindow: 速度窗口
"""

import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass

from config import Config, config as global_config
from utils import (Point, Circle, Path, euclidean_distance, normalize_angle,
                   angle_difference, vector_to_angle)


# =============================================================================
# 速度命令
# =============================================================================

@dataclass
class VelocityCommand:
    """速度命令"""
    vx: float = 0.0      # 前向速度 (mm/s)
    vy: float = 0.0      # 侧向速度 (mm/s)
    vw: float = 0.0      # 角速度 (rad/s)
    
    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.vx, self.vy, self.vw)
    
    @property
    def linear_speed(self) -> float:
        return math.hypot(self.vx, self.vy)


# =============================================================================
# 机器人状态
# =============================================================================

@dataclass
class RobotState:
    """机器人状态"""
    x: float                # x 坐标
    y: float                # y 坐标
    heading: float          # 朝向角 (弧度)
    vx: float = 0.0         # 当前前向速度
    vw: float = 0.0         # 当前角速度
    
    @property
    def position(self) -> np.ndarray:
        return np.array([self.x, self.y])
    
    @property
    def point(self) -> Point:
        return Point(self.x, self.y)


# =============================================================================
# 动态窗口
# =============================================================================

class VelocityWindow:
    """
    动态速度窗口
    
    考虑机器人速度限制和加速度限制
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
    
    def compute(self, current_vx: float, current_vw: float,
                dt: float = 0.5) -> Tuple[float, float, float, float]:
        """
        计算动态窗口
        
        Args:
            current_vx: 当前线速度
            current_vw: 当前角速度
            dt: 时间步长
            
        Returns:
            (v_min, v_max, w_min, w_max)
        """
        # 运动学限制
        v_min = 0  # 不允许后退
        v_max = self.cfg.robot.MAX_VX
        w_min = -self.cfg.robot.MAX_VW
        w_max = self.cfg.robot.MAX_VW
        
        # 加速度限制 - 使用更大的时间窗口
        accel_time = max(dt, 0.5)  # 至少0.5秒的加速时间
        v_min = max(v_min, current_vx - self.cfg.robot.MAX_AX * accel_time)
        v_max = min(v_max, current_vx + self.cfg.robot.MAX_AX * accel_time)
        w_min = max(w_min, current_vw - self.cfg.robot.MAX_AW * accel_time)
        w_max = min(w_max, current_vw + self.cfg.robot.MAX_AW * accel_time)
        
        return v_min, v_max, w_min, w_max


# =============================================================================
# DWA 规划器
# =============================================================================

class DWAPlanner:
    """
    DWA 动态窗口法局部规划器
    
    在速度空间中搜索最优速度命令，实现实时避障
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        self.velocity_window = VelocityWindow(cfg)
        
        # 缓存上一时刻速度（用于加速度限制）
        self.last_vx = 0.0
        self.last_vw = 0.0
    
    def compute_velocity(self,
                         robot_state: RobotState,
                         target_point: Point,
                         path: Path,
                         obstacles: List[Circle]) -> VelocityCommand:
        """
        计算最优速度命令
        
        Args:
            robot_state: 机器人当前状态
            target_point: 目标点
            path: 全局路径
            obstacles: 障碍物列表
            
        Returns:
            最优速度命令
        """
        # 计算动态窗口
        v_min, v_max, w_min, w_max = self.velocity_window.compute(
            self.last_vx, self.last_vw
        )
        
        # 计算目标朝向
        dx = target_point.x - robot_state.x
        dy = target_point.y - robot_state.y
        dist_to_target = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx)
        
        # 计算朝向误差
        heading_error = angle_difference(target_heading, robot_state.heading)
        
        # 基础控制：先转向，再前进
        # 角速度：根据朝向误差直接计算
        # heading_error = target_heading - current_heading (归一化后)
        # 当 heading_error > 0 时，目标在左边，需要向左转（正角速度）
        # 当 heading_error < 0 时，目标在右边，需要向右转（负角速度）
        base_vw = 4.0 * heading_error
        base_vw = np.clip(base_vw, w_min, w_max)
        
        # 线速度：根据朝向误差决定
        abs_error = abs(heading_error)
        
        if abs_error > math.pi * 2 / 3:  # > 120度，停止前进，原地转
            base_vx = 0
        elif abs_error > math.pi / 2:  # > 90度，几乎停止
            base_vx = 50
        elif abs_error > math.pi / 3:  # > 60度，慢速
            base_vx = min(500, v_max * 0.3)
        elif abs_error > math.pi / 6:  # > 30度，中速
            base_vx = min(1000, v_max * 0.5)
        else:  # < 30度，全速
            base_vx = v_max
        
        # 障碍物避让调整
        vx, vw = self._adjust_for_obstacles(
            robot_state, base_vx, base_vw, obstacles, target_heading
        )
        
        # 更新缓存
        self.last_vx = vx
        self.last_vw = vw
        
        return VelocityCommand(vx=vx, vw=vw)
    
    def _adjust_for_obstacles(self,
                               robot_state: RobotState,
                               base_vx: float,
                               base_vw: float,
                               obstacles: List[Circle],
                               target_heading: float) -> Tuple[float, float]:
        """根据障碍物调整速度"""
        vx, vw = base_vx, base_vw
        
        # 检查前方障碍物
        for obs in obstacles:
            dx = obs.center.x - robot_state.x
            dy = obs.center.y - robot_state.y
            dist = math.hypot(dx, dy)
            
            # 障碍物相对于机器人的角度
            angle_to_obs = math.atan2(dy, dx)
            
            # 障碍物相对于机器人朝向的角度差
            angle_diff = angle_difference(angle_to_obs, robot_state.heading)
            
            # 前方障碍物检测 (前方60度扇形区域)
            if abs(angle_diff) < math.pi / 3:
                safe_dist = obs.radius + 200  # 额外安全距离
                
                if dist < safe_dist * 2:
                    # 接近障碍物，减速
                    slowdown = max(0.2, (dist - safe_dist) / safe_dist)
                    vx = base_vx * slowdown
                    
                    if dist < safe_dist * 1.2:
                        # 太近了，需要转向避让
                        # 选择远离障碍物的方向
                        # 如果障碍物在左边，向右转；如果在右边，向左转
                        avoid_direction = -np.sign(angle_diff)
                        vw += avoid_direction * 4.0
        
        return vx, vw
    
    def compute_velocity_simple(self,
                                robot_state: RobotState,
                                target_point: Point,
                                obstacles: List[Circle]) -> VelocityCommand:
        """
        简化版本：不考虑全局路径，直接跟踪目标点
        
        Args:
            robot_state: 机器人状态
            target_point: 目标点
            obstacles: 障碍物列表
            
        Returns:
            速度命令
        """
        empty_path = Path(points=[target_point])
        return self.compute_velocity(robot_state, target_point, empty_path, obstacles)
    
    # -------------------------------------------------------------------------
    # 内部方法
    # -------------------------------------------------------------------------
    
    def _predict_trajectory(self,
                            state: RobotState,
                            v: float,
                            w: float,
                            predict_time: float,
                            step: float) -> List[np.ndarray]:
        """
        预测机器人轨迹
        
        Args:
            state: 当前状态
            v: 线速度
            w: 角速度
            predict_time: 预测时间
            step: 时间步长
            
        Returns:
            轨迹点列表
        """
        trajectory = []
        x, y, theta = state.x, state.y, state.heading
        
        t = 0.0
        while t < predict_time:
            # 运动学模型
            theta += w * step
            x += v * math.cos(theta) * step
            y += v * math.sin(theta) * step
            
            trajectory.append(np.array([x, y]))
            t += step
        
        return trajectory
    
    def _is_trajectory_safe(self,
                            trajectory: List[np.ndarray],
                            obstacles: List[Circle]) -> bool:
        """
        检查轨迹是否安全
        
        Args:
            trajectory: 预测轨迹
            obstacles: 障碍物列表
            
        Returns:
            是否安全
        """
        # 障碍物半径已包含安全距离，所以不需要再加机器人半径
        for point in trajectory:
            for obs in obstacles:
                dist = euclidean_distance(point, [obs.center.x, obs.center.y])
                # 障碍物半径 = 机器人半径 + 安全距离
                # 所以这里只需要保证不进入障碍物半径即可
                if dist < obs.radius:
                    return False
        
        return True
    
    def _evaluate_trajectory(self,
                             trajectory: List[np.ndarray],
                             state: RobotState,
                             target_point: Point,
                             path: Path,
                             obstacles: List[Circle]) -> float:
        """
        评估轨迹得分
        
        G(v,w) = α·heading + β·dist + γ·velocity + δ·path
        
        Args:
            trajectory: 预测轨迹
            state: 当前状态
            target_point: 目标点
            path: 全局路径
            obstacles: 障碍物列表
            
        Returns:
            轨迹得分
        """
        if not trajectory:
            return float('-inf')
        
        # 轨迹终点
        end_point = trajectory[-1]
        
        # 1. 朝向得分：从当前位置到目标点的朝向
        heading_score = self._heading_score(state.position, target_point)
        
        # 2. 障碍距离得分
        dist_score = self._obstacle_distance_score(trajectory, obstacles)
        
        # 3. 速度得分
        velocity_score = self._velocity_score(trajectory)
        
        # 4. 路径跟随得分
        path_score = self._path_following_score(trajectory, path)
        
        # 加权求和
        total_score = (
            self.cfg.dwa.HEADING_WEIGHT * heading_score +
            self.cfg.dwa.DIST_WEIGHT * dist_score +
            self.cfg.dwa.VELOCITY_WEIGHT * velocity_score +
            self.cfg.dwa.PATH_WEIGHT * path_score
        )
        
        return total_score
    
    def _heading_score(self,
                       current_pos: np.ndarray,
                       target_point: Point) -> float:
        """计算朝向得分 - 终点越接近目标得分越高"""
        # 计算当前位置到目标的方向
        dx = target_point.x - current_pos[0]
        dy = target_point.y - current_pos[1]
        dist = math.hypot(dx, dy)
        
        if dist < 1.0:
            return 1.0  # 已到达目标
        
        # 归一化：距离目标越近得分越高
        # 使用距离的倒数作为得分
        max_dist = 8000.0  # 最大距离
        return 1.0 - min(dist / max_dist, 1.0)
    
    def _obstacle_distance_score(self,
                                 trajectory: List[np.ndarray],
                                 obstacles: List[Circle]) -> float:
        """计算障碍距离得分"""
        min_dist = float('inf')
        
        for point in trajectory:
            for obs in obstacles:
                dist = euclidean_distance(point, [obs.center.x, obs.center.y])
                # 减去障碍物半径（已包含安全距离）
                clearance = dist - obs.radius
                min_dist = min(min_dist, clearance)
        
        # 如果最小间隙为负，说明有碰撞风险
        if min_dist < 0:
            return 0.0
        
        # 归一化
        normalized = min(min_dist / self.cfg.dwa.OBSTACLE_THRESHOLD, 1.0)
        return normalized
    
    def _velocity_score(self, trajectory: List[np.ndarray]) -> float:
        """计算速度得分（鼓励前进）"""
        if len(trajectory) < 2:
            return 0.0
        
        # 计算移动距离
        start = trajectory[0]
        end = trajectory[-1]
        distance = euclidean_distance(start, end)
        
        # 归一化
        max_distance = self.cfg.robot.MAX_VX * self.cfg.dwa.PREDICT_TIME
        return min(distance / max_distance, 1.0)
    
    def _path_following_score(self,
                              trajectory: List[np.ndarray],
                              path: Path) -> float:
        """计算路径跟随得分"""
        if path.is_empty or len(trajectory) == 0:
            return 1.0
        
        # 计算轨迹到路径的最大偏差
        max_deviation = 0.0
        
        for point in trajectory:
            # 找到路径上最近的点
            min_dist = float('inf')
            for path_point in path:
                dist = euclidean_distance(point, [path_point.x, path_point.y])
                min_dist = min(min_dist, dist)
            max_deviation = max(max_deviation, min_dist)
        
        # 归一化（偏差越小得分越高）
        threshold = 500.0  # 最大允许偏差
        if max_deviation > threshold:
            return 0.0
        
        return 1.0 - max_deviation / threshold
    
    def _check_path_turn(self, path: Path, current_pos: np.ndarray) -> float:
        """
        检测路径拐点，返回速度因子
        
        当接近路径拐点时返回较小的因子，使机器人减速
        """
        if path.is_empty or len(path) < 3:
            return 1.0
        
        # 找到当前位置之后的路径点
        min_dist = float('inf')
        closest_idx = 0
        for i, point in enumerate(path):
            dist = euclidean_distance(current_pos, point.to_array())
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # 检查接下来的路径方向变化
        if closest_idx + 2 < len(path):
            p1 = path[closest_idx].to_array()
            p2 = path[closest_idx + 1].to_array()
            p3 = path[closest_idx + 2].to_array()
            
            # 计算方向变化
            dir1 = p2 - p1
            dir2 = p3 - p2
            
            if np.linalg.norm(dir1) > 1 and np.linalg.norm(dir2) > 1:
                dir1 = dir1 / np.linalg.norm(dir1)
                dir2 = dir2 / np.linalg.norm(dir2)
                
                # 计算方向变化角度
                dot = np.clip(np.dot(dir1, dir2), -1, 1)
                angle_change = math.acos(dot)
                
                # 如果方向变化大（拐点），减速
                if angle_change > math.pi / 6:  # > 30度拐弯
                    # 计算到拐点的距离
                    dist_to_turn = euclidean_distance(current_pos, p2)
                    if dist_to_turn < 800:  # 接近拐点
                        return max(0.3, dist_to_turn / 800)
        
        return 1.0

    def reset(self):
        """重置状态"""
        self.last_vx = 0.0
        self.last_vw = 0.0


# =============================================================================
# 简化的避障控制器 (用于紧急避障)
# =============================================================================

class EmergencyAvoidance:
    """
    紧急避障控制器
    
    当检测到危险时进行紧急避让
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
    
    def check_emergency(self,
                        robot_state: RobotState,
                        obstacles: List[Circle]) -> Tuple[bool, Optional[VelocityCommand]]:
        """
        检查是否需要紧急避障
        
        Args:
            robot_state: 机器人状态
            obstacles: 障碍物列表
            
        Returns:
            (是否紧急, 避障命令)
        """
        robot_pos = robot_state.position
        robot_radius = self.cfg.robot.RADIUS
        emergency_dist = robot_radius + self.cfg.robot.SAFE_DISTANCE * 0.5
        
        for obs in obstacles:
            dist = euclidean_distance(robot_pos, [obs.center.x, obs.center.y])
            
            if dist < emergency_dist:
                # 计算避让方向
                dx = robot_pos[0] - obs.center.x
                dy = robot_pos[1] - obs.center.y
                direction = math.atan2(dy, dx)
                
                # 计算角度差
                angle_diff = angle_difference(robot_state.heading, direction)
                
                # 急转避让
                if abs(angle_diff) < math.pi / 2:
                    vw = np.sign(angle_diff) * self.cfg.robot.MAX_VW * 0.5
                    return True, VelocityCommand(vx=0, vw=vw)
                else:
                    # 后退
                    return True, VelocityCommand(vx=-500, vw=0)
        
        return False, None
