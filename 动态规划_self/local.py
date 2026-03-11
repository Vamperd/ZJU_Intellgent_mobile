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
    改进的局部路径跟踪器
    
    使用Pure Pursuit + PID控制实现稳定的路径跟踪
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        self.velocity_window = VelocityWindow(cfg)
        
        self.last_vx = 0.0
        self.last_vw = 0.0
        self.obstacle_history = []
        self.max_history = 10
        
        self.kp_angle = 4.0
        self.kd_angle = 0.3
        self.last_heading_error = 0.0
    
    def compute_velocity(self,
                         robot_state: RobotState,
                         target_point: Point,
                         path: Path,
                         obstacles: List[Circle]) -> VelocityCommand:
        """
        计算速度命令 - 简化和稳定的Pure Pursuit版本
        
        Args:
            robot_state: 机器人当前状态
            target_point: 目标点
            path: 全局路径
            obstacles: 障碍物列表
            
        Returns:
            最优速度命令
        """
        self._update_obstacle_history(obstacles)
        
        dx = target_point.x - robot_state.x
        dy = target_point.y - robot_state.y
        dist_to_target = math.hypot(dx, dy)
        target_heading = math.atan2(dy, dx)
        
        heading_error = angle_difference(target_heading, robot_state.heading)
        
        dt = 0.01
        d_error = (heading_error - self.last_heading_error) / dt
        self.last_heading_error = heading_error
        
        vw = self.kp_angle * heading_error + self.kd_angle * d_error
        vw = np.clip(vw, -self.cfg.robot.MAX_VW, self.cfg.robot.MAX_VW)
        
        abs_error = abs(heading_error)
        
        if dist_to_target < 200:
            vx = 200 * (dist_to_target / 200)
        elif abs_error > math.pi / 2:
            vx = 50
        elif abs_error > math.pi / 3:
            vx = 300
        elif abs_error > math.pi / 6:
            vx = 800
        else:
            vx = 1500
        
        avoid_vx, avoid_vw = self._check_obstacle_avoidance(
            robot_state, obstacles, vx, vw
        )
        
        vx = avoid_vx
        vw = avoid_vw
        
        self.last_vx = vx
        self.last_vw = vw
        
        return VelocityCommand(vx=vx, vw=vw)
    
    def _check_obstacle_avoidance(self,
                                  robot_state: RobotState,
                                  obstacles: List[Circle],
                                  base_vx: float,
                                  base_vw: float) -> Tuple[float, float]:
        """检查障碍物并调整速度"""
        vx, vw = base_vx, base_vw
        
        closest_obs = None
        closest_dist = float('inf')
        
        for obs in obstacles:
            dx = obs.center.x - robot_state.x
            dy = obs.center.y - robot_state.y
            dist = math.hypot(dx, dy)
            
            if dist < closest_dist:
                closest_dist = dist
                closest_obs = obs
        
        if closest_obs and closest_dist < closest_obs.radius + 300:
            angle_to_obs = math.atan2(
                closest_obs.center.y - robot_state.y,
                closest_obs.center.x - robot_state.x
            )
            angle_diff = angle_difference(angle_to_obs, robot_state.heading)
            
            if abs(angle_diff) < math.pi / 2:
                safe_dist = closest_obs.radius + 150
                
                if closest_dist < safe_dist:
                    vx = 150
                    avoid_dir = -np.sign(angle_diff)
                    vw = avoid_dir * 4.0
                else:
                    slowdown = (closest_dist - safe_dist) / 300
                    vx = base_vx * max(0.5, slowdown)
        
        return vx, vw
    
    def _update_obstacle_history(self, obstacles: List[Circle]):
        """更新障碍物历史"""
        self.obstacle_history.append([(obs.center.x, obs.center.y) for obs in obstacles])
        if len(self.obstacle_history) > self.max_history:
            self.obstacle_history.pop(0)
    
    def _predict_obstacle_position(self, obs_center: np.ndarray, dt: float) -> np.ndarray:
        """预测障碍物未来位置"""
        if len(self.obstacle_history) < 2:
            return obs_center
        
        prev_centers = [h for h in self.obstacle_history[-2] 
                       if np.linalg.norm(np.array(h) - obs_center) < 500]
        
        if not prev_centers:
            return obs_center
        
        velocity = obs_center - np.array(prev_centers[0])
        return obs_center + velocity * dt
    
    def _search_best_velocity(self,
                              robot_state: RobotState,
                              target_point: Point,
                              path: Path,
                              obstacles: List[Circle],
                              v_min: float, v_max: float,
                              w_min: float, w_max: float,
                              base_vx: float, base_vw: float) -> VelocityCommand:
        """在速度空间中搜索最优速度"""
        
        v_samples = self.cfg.dwa.V_SAMPLES
        w_samples = self.cfg.dwa.W_SAMPLES
        
        best_score = float('-inf')
        best_vx, best_vw = base_vx, base_vw
        
        v_range = v_max - v_min
        w_range = w_max - w_min
        
        for i in range(v_samples):
            for j in range(w_samples):
                if v_range > 0:
                    v = v_min + (i + 1) * v_range / (v_samples + 1)
                else:
                    v = v_max
                
                if w_range > 0:
                    w = w_min + (j + 1) * w_range / (w_samples + 1)
                else:
                    w = 0
                
                trajectory = self._predict_trajectory(
                    robot_state, v, w,
                    self.cfg.dwa.PREDICT_TIME,
                    self.cfg.dwa.PREDICT_STEP
                )
                
                if not self._is_trajectory_safe(trajectory, obstacles):
                    continue
                
                score = self._evaluate_trajectory(
                    trajectory, robot_state, target_point, path, obstacles
                )
                
                if score > best_score:
                    best_score = score
                    best_vx, best_vw = v, w
        
        if best_score == float('-inf'):
            return self._emergency_stop(robot_state, obstacles)
        
        return VelocityCommand(vx=best_vx, vw=best_vw)
    
    def _emergency_stop(self, robot_state: RobotState, 
                        obstacles: List[Circle]) -> VelocityCommand:
        """紧急停止"""
        for obs in obstacles:
            dx = obs.center.x - robot_state.x
            dy = obs.center.y - robot_state.y
            dist = math.hypot(dx, dy)
            
            if dist < obs.radius + 300:
                angle_to_obs = math.atan2(dy, dx)
                angle_diff = angle_difference(angle_to_obs, robot_state.heading)
                
                if abs(angle_diff) < math.pi / 2:
                    return VelocityCommand(vx=0, vw=np.sign(angle_diff) * 8.0)
        
        return VelocityCommand(vx=100, vw=0)
    
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
