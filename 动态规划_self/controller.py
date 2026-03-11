"""
运动控制模块

包含:
- PIDController: 改进的 PID 控制器
- PurePursuit: Pure Pursuit 路径跟踪器
- AttitudeController: 姿态控制器
- MotionController: 统一运动控制器
"""

import math
import time
import numpy as np
from typing import Optional, Tuple

from config import Config, config as global_config
from utils import Point, Path, normalize_angle, angle_difference, vector_to_angle


# =============================================================================
# PID 控制器
# =============================================================================

class PIDController:
    """
    改进的 PID 控制器
    
    特点:
    - 积分分离：防止积分饱和
    - 输出限幅：限制最大输出
    - 抗积分饱和：输出饱和时停止积分
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        self.reset()
    
    def reset(self):
        """重置控制器状态"""
        self.integral = 0.0
        self.last_error = None
        self.last_time = time.time()
        self.output = 0.0
    
    def update(self, error: float) -> float:
        """
        更新 PID 输出
        
        Args:
            error: 误差值
            
        Returns:
            控制输出
        """
        current_time = time.time()
        dt = current_time - self.last_time
        
        if dt <= 0:
            dt = 0.01  # 防止除零
        
        # 比例项
        p_term = self.cfg.pid.KP * error
        
        # 积分项 (带分离)
        if abs(error) < self.cfg.pid.INTEGRAL_LIMIT:
            self.integral += error * dt
        else:
            self.integral = 0.0  # 积分分离
        
        # 抗积分饱和
        i_term = self.cfg.pid.KI * self.integral
        
        # 微分项
        d_term = 0.0
        if self.last_error is not None:
            d_term = self.cfg.pid.KD * (error - self.last_error) / dt
        
        # 计算输出
        self.output = p_term + i_term + d_term
        
        # 输出限幅
        self.output = np.clip(self.output, -self.cfg.pid.OUTPUT_LIMIT, 
                              self.cfg.pid.OUTPUT_LIMIT)
        
        # 更新状态
        self.last_error = error
        self.last_time = current_time
        
        return self.output
    
    def set_gains(self, kp: float = None, ki: float = None, kd: float = None):
        """设置 PID 增益"""
        if kp is not None:
            self.cfg.pid.KP = kp
        if ki is not None:
            self.cfg.pid.KI = ki
        if kd is not None:
            self.cfg.pid.KD = kd


# =============================================================================
# Pure Pursuit 路径跟踪器
# =============================================================================

class PurePursuit:
    """
    Pure Pursuit 路径跟踪器
    
    通过计算前视点的曲率来控制转向
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        self.current_waypoint_index = 0
    
    def compute_steering(self,
                         robot_pos: np.ndarray,
                         robot_heading: float,
                         path: Path,
                         current_velocity: float = 0.0) -> Tuple[float, Optional[Point]]:
        """
        计算转向角速度
        
        Args:
            robot_pos: 机器人位置 [x, y]
            robot_heading: 机器人朝向 (弧度)
            path: 路径
            current_velocity: 当前速度
            
        Returns:
            (角速度, 前视点)
        """
        if path.is_empty:
            return 0.0, None
        
        # 计算自适应前视距离
        lookahead = self._compute_lookahead_distance(current_velocity)
        
        # 找前视点
        lookahead_point = self._find_lookahead_point(robot_pos, path, lookahead)
        
        if lookahead_point is None:
            return 0.0, None
        
        # 计算曲率
        curvature = self._compute_curvature(robot_pos, robot_heading, lookahead_point)
        
        # 计算角速度
        # ω = v * γ
        angular_velocity = current_velocity * curvature
        
        return angular_velocity, lookahead_point
    
    def _compute_lookahead_distance(self, velocity: float) -> float:
        """计算自适应前视距离"""
        # Ld = k * v + L_min
        lookahead = (self.cfg.pure_pursuit.LOOKAHEAD_GAIN * velocity +
                     self.cfg.pure_pursuit.MIN_LOOKAHEAD)
        return min(lookahead, self.cfg.pure_pursuit.MAX_LOOKAHEAD)
    
    def _find_lookahead_point(self,
                              robot_pos: np.ndarray,
                              path: Path,
                              lookahead: float) -> Optional[Point]:
        """
        找前视点
        
        从当前路径点开始，找到第一个距离大于前视距离的点
        """
        # 从上次的位置开始搜索
        start_index = max(0, self.current_waypoint_index - 1)
        
        for i in range(start_index, len(path)):
            point = path[i]
            dist = np.linalg.norm(robot_pos - point.to_array())
            
            if dist >= lookahead:
                self.current_waypoint_index = i
                return point
        
        # 如果没有找到，返回最后一个点
        return path.end
    
    def _compute_curvature(self,
                           robot_pos: np.ndarray,
                           robot_heading: float,
                           target_point: Point) -> float:
        """
        计算曲率
        
        γ = 2 * sin(α) / Ld
        
        其中:
        - α: 机器人朝向与目标点方向的夹角
        - Ld: 前视距离
        """
        dx = target_point.x - robot_pos[0]
        dy = target_point.y - robot_pos[1]
        Ld = math.hypot(dx, dy)
        
        if Ld < 1e-6:
            return 0.0
        
        # 目标点相对于机器人的角度
        target_angle = math.atan2(dy, dx)
        
        # 计算夹角 α
        alpha = angle_difference(target_angle, robot_heading)
        
        # 计算曲率
        curvature = 2.0 * math.sin(alpha) / Ld
        
        return curvature
    
    def reset(self):
        """重置状态"""
        self.current_waypoint_index = 0


# =============================================================================
# 姿态控制器
# =============================================================================

class AttitudeController:
    """
    姿态控制器
    
    使用 PID 控制机器人朝向
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        self.pid = PIDController(cfg)
    
    def compute_angular_velocity(self,
                                 target_heading: float,
                                 current_heading: float) -> float:
        """
        计算角速度
        
        Args:
            target_heading: 目标朝向 (弧度)
            current_heading: 当前朝向 (弧度)
            
        Returns:
            角速度 (rad/s)
        """
        error = angle_difference(target_heading, current_heading)
        return -self.pid.update(error)
    
    def compute_angular_velocity_to_point(self,
                                          robot_pos: np.ndarray,
                                          target_point: Point,
                                          current_heading: float) -> float:
        """
        计算转向目标点的角速度
        
        Args:
            robot_pos: 机器人位置
            target_point: 目标点
            current_heading: 当前朝向
            
        Returns:
            角速度
        """
        dx = target_point.x - robot_pos[0]
        dy = target_point.y - robot_pos[1]
        target_heading = math.atan2(dy, dx)
        
        return self.compute_angular_velocity(target_heading, current_heading)
    
    def reset(self):
        """重置控制器"""
        self.pid.reset()


# =============================================================================
# 速度规划器
# =============================================================================

class VelocityPlanner:
    """
    速度规划器
    
    根据距离目标点的距离和转弯角度规划速度
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
    
    def compute_velocity(self,
                         distance_to_goal: float,
                         heading_error: float,
                         near_obstacle: bool = False) -> float:
        """
        计算期望速度
        
        Args:
            distance_to_goal: 到目标点的距离
            heading_error: 朝向误差 (弧度)
            near_obstacle: 是否接近障碍物
            
        Returns:
            期望速度 (mm/s)
        """
        max_v = self.cfg.robot.MAX_VX
        
        # 接近目标时减速
        slowdown_distance = 1000.0
        if distance_to_goal < slowdown_distance:
            speed_factor = distance_to_goal / slowdown_distance
            max_v *= max(0.3, speed_factor)
        
        # 转弯时减速
        turn_factor = 1.0 - abs(heading_error) / math.pi
        max_v *= max(0.2, turn_factor)
        
        # 接近障碍物时减速
        if near_obstacle:
            max_v *= 0.5
        
        return max_v
    
    def compute_velocity_simple(self,
                                distance_to_goal: float,
                                heading_error: float) -> float:
        """简化版本"""
        return self.compute_velocity(distance_to_goal, heading_error, False)


# =============================================================================
# 统一运动控制器
# =============================================================================

class MotionController:
    """
    统一运动控制器
    
    整合 Pure Pursuit 和姿态控制
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        self.pure_pursuit = PurePursuit(cfg)
        self.attitude_ctrl = AttitudeController(cfg)
        self.velocity_planner = VelocityPlanner(cfg)
        
        # 当前速度状态（用于加速度限制）
        self.current_vx = 0.0
        self.current_vw = 0.0
    
    def compute_command(self,
                        robot_pos: np.ndarray,
                        robot_heading: float,
                        path: Path,
                        goal: np.ndarray,
                        near_obstacle: bool = False) -> Tuple[float, float]:
        """
        计算速度命令
        
        Args:
            robot_pos: 机器人位置 [x, y]
            robot_heading: 机器人朝向
            path: 路径
            goal: 最终目标
            near_obstacle: 是否接近障碍物
            
        Returns:
            (vx, vw) 速度命令
        """
        # 计算到目标的距离
        distance_to_goal = np.linalg.norm(robot_pos - goal)
        
        # Pure Pursuit 计算转向
        vw_pp, lookahead_point = self.pure_pursuit.compute_steering(
            robot_pos, robot_heading, path, self.current_vx
        )
        
        # 计算期望朝向
        if lookahead_point:
            dx = lookahead_point.x - robot_pos[0]
            dy = lookahead_point.y - robot_pos[1]
            target_heading = math.atan2(dy, dx)
        else:
            target_heading = robot_heading
        
        # 姿态控制
        heading_error = angle_difference(target_heading, robot_heading)
        vw_attitude = self.attitude_ctrl.compute_angular_velocity(
            target_heading, robot_heading
        )
        
        # 融合两种转向控制
        # 当速度较低时，更依赖姿态控制
        # 当速度较高时，更依赖 Pure Pursuit
        if self.current_vx < 500:
            vw = vw_attitude
        else:
            vw = 0.5 * vw_pp + 0.5 * vw_attitude
        
        # 速度规划
        vx = self.velocity_planner.compute_velocity(
            distance_to_goal, heading_error, near_obstacle
        )
        
        # 应用加速度限制
        vx = self._apply_acceleration_limit(self.current_vx, vx)
        vw = self._apply_angular_acceleration_limit(self.current_vw, vw)
        
        # 更新状态
        self.current_vx = vx
        self.current_vw = vw
        
        return vx, vw
    
    def compute_command_to_point(self,
                                 robot_pos: np.ndarray,
                                 robot_heading: float,
                                 target_point: Point) -> Tuple[float, float]:
        """
        计算到单点的速度命令
        
        Args:
            robot_pos: 机器人位置
            robot_heading: 机器人朝向
            target_point: 目标点
            
        Returns:
            (vx, vw) 速度命令
        """
        # 计算距离
        distance = np.linalg.norm(robot_pos - target_point.to_array())
        
        # 计算期望朝向
        dx = target_point.x - robot_pos[0]
        dy = target_point.y - robot_pos[1]
        target_heading = math.atan2(dy, dx)
        
        # 姿态控制
        heading_error = angle_difference(target_heading, robot_heading)
        vw = self.attitude_ctrl.compute_angular_velocity(target_heading, robot_heading)
        
        # 速度规划
        vx = self.velocity_planner.compute_velocity(distance, heading_error)
        
        # 加速度限制
        vx = self._apply_acceleration_limit(self.current_vx, vx)
        vw = self._apply_angular_acceleration_limit(self.current_vw, vw)
        
        self.current_vx = vx
        self.current_vw = vw
        
        return vx, vw
    
    def rotation_command(self, direction: int = 1) -> Tuple[float, float]:
        """
        生成原地旋转命令
        
        Args:
            direction: 旋转方向，1 为正向，-1 为反向
            
        Returns:
            (vx, vw)
        """
        return 0.0, direction * self.cfg.robot.MAX_VW * 0.8
    
    def stop_command(self) -> Tuple[float, float]:
        """生成停止命令"""
        self.current_vx = 0.0
        self.current_vw = 0.0
        return 0.0, 0.0
    
    def _apply_acceleration_limit(self, current: float, target: float) -> float:
        """应用线加速度限制"""
        dt = 0.01  # 控制周期
        max_delta = self.cfg.robot.MAX_AX * dt
        
        delta = target - current
        if abs(delta) > max_delta:
            return current + np.sign(delta) * max_delta
        return target
    
    def _apply_angular_acceleration_limit(self, current: float, target: float) -> float:
        """应用角加速度限制"""
        dt = 0.01
        max_delta = self.cfg.robot.MAX_AW * dt
        
        delta = target - current
        if abs(delta) > max_delta:
            return current + np.sign(delta) * max_delta
        return target
    
    def reset(self):
        """重置所有状态"""
        self.pure_pursuit.reset()
        self.attitude_ctrl.reset()
        self.current_vx = 0.0
        self.current_vw = 0.0
