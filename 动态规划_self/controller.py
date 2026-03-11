"""
运动控制模块

包含:
- PIDController: 改进的 PID 控制器
- PathTracker: 路径跟踪控制器（优化版）
- MotionController: 统一运动控制器
"""

import math
import time
import numpy as np
from typing import Optional, Tuple, List

from config import Config, config as global_config
from utils import Point, Path, normalize_angle, angle_difference, euclidean_distance


# =============================================================================
# PID 控制器
# =============================================================================

class PIDController:
    """PID 控制器"""
    
    def __init__(self, kp: float = 1.0, ki: float = 0.0, kd: float = 0.0,
                 output_limit: float = 12.0, integral_limit: float = 10.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral_limit = integral_limit
        self.reset()
    
    def reset(self):
        """重置控制器状态"""
        self.integral = 0.0
        self.last_error = None
        self.last_time = time.time()
        self.output = 0.0
    
    def update(self, error: float, dt: float = None) -> float:
        """更新 PID 输出"""
        current_time = time.time()
        if dt is None:
            dt = current_time - self.last_time
        
        if dt <= 0:
            dt = 0.01
        
        # 比例项
        p_term = self.kp * error
        
        # 积分项 (带分离和限幅)
        if abs(error) < self.integral_limit:
            self.integral += error * dt
        else:
            self.integral = 0.0
        
        i_term = self.ki * self.integral
        
        # 微分项
        d_term = 0.0
        if self.last_error is not None:
            d_term = self.kd * (error - self.last_error) / dt
        
        # 计算输出
        self.output = p_term + i_term + d_term
        
        # 输出限幅
        self.output = np.clip(self.output, -self.output_limit, self.output_limit)
        
        # 更新状态
        self.last_error = error
        self.last_time = current_time
        
        return self.output


# =============================================================================
# 路径跟踪控制器（优化版）
# =============================================================================

class PathTracker:
    """
    路径跟踪控制器
    
    优化点:
    1. 直接参考静态规划的 Move 类实现
    2. 修复角度计算函数
    3. 改进速度规划策略
    4. 平滑路径点切换
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        
        # PID 状态变量（与静态规划 move.py 一致）
        self.position_error_last = 0.0
        self.position_error = 0.0
        self.position_error_sum = 0.0
        
        self.theta_error_last = 0.0
        self.theta_error = 0.0
        self.theta_error_sum = 0.0
        
        # 当前速度
        self.current_vx = 0.0
        self.current_vw = 0.0
    
    def get_action(self, vision, path_x: List[float], path_y: List[float],
                   waypoint_index: int, direction: int = 1) -> Tuple[float, float]:
        """
        计算速度命令（与静态规划 move.py 一致的实现）
        
        Args:
            vision: 视觉对象
            path_x, path_y: 路径坐标
            waypoint_index: 当前目标路径点索引
            direction: 移动方向，1=正向，-1=反向
            
        Returns:
            (vx, vw) 速度命令
        """
        if waypoint_index >= len(path_x):
            return 0.0, 0.0
        
        current_x = vision.my_robot.x
        current_y = vision.my_robot.y
        current_theta = vision.my_robot.orientation
        
        # 计算当前点与下一个目标点的距离
        dist_error1 = self._cal_dist(
            current_x, current_y,
            path_x[waypoint_index], path_y[waypoint_index]
        )
        
        # 计算当前位置与相邻目标点的距离（用于平滑误差跳变）
        if direction == 1:
            # 正向移动：检查是否有下一个点
            if waypoint_index + 1 < len(path_x):
                dist_error2 = self._cal_dist(
                    current_x, current_y,
                    path_x[waypoint_index + 1], path_y[waypoint_index + 1]
                )
            else:
                dist_error2 = dist_error1
        else:
            # 反向移动：检查是否有上一个点
            if waypoint_index - 1 >= 0:
                dist_error2 = self._cal_dist(
                    current_x, current_y,
                    path_x[waypoint_index - 1], path_y[waypoint_index - 1]
                )
            else:
                dist_error2 = dist_error1
        
        # 取较小的距离作为误差
        dist_error = min(dist_error1, dist_error2)
        
        # 计算当前点与下一个目标点的角度
        target_theta = self._cal_theta(
            current_x, current_y,
            path_x[waypoint_index], path_y[waypoint_index]
        )
        
        # 计算朝向误差
        theta_error = self._cal_theta_error(target_theta, current_theta)
        
        # 更新位置误差状态
        self.position_error_last = self.position_error
        self.position_error = dist_error
        self.position_error_sum += self.position_error
        
        # 更新角度误差状态
        self.theta_error_last = self.theta_error
        self.theta_error = theta_error
        self.theta_error_sum += self.theta_error
        
        # 判断是否为中间点（用于速度规划）
        is_middle_point = (waypoint_index != 0 and 
                          waypoint_index != len(path_x) - 1)
        
        # 计算 vx 和 vw
        vx = self._pid_vx(is_middle_point)
        vw = self._pid_vw()
        
        # 更新当前速度
        self.current_vx = vx
        self.current_vw = vw
        
        return vx, vw
    
    def _cal_dist(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """计算两点距离"""
        return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    
    def _cal_theta(self, x1: float, y1: float, x2: float, y2: float) -> float:
        """
        计算从点1到点2的方向角
        
        使用 atan2 避免除零问题，返回 [-π, π]
        """
        return math.atan2(y2 - y1, x2 - x1)
    
    def _cal_theta_error(self, target_theta: float, current_theta: float) -> float:
        """计算角度误差（归一化到 [-π, π]）"""
        theta_error = target_theta - current_theta
        while theta_error > math.pi:
            theta_error -= 2 * math.pi
        while theta_error < -math.pi:
            theta_error += 2 * math.pi
        return theta_error
    
    def _speed_function(self, x: float) -> float:
        """
        非线性速度规划函数（与静态规划一致）
        
        在目标点附近平滑减速，避免超调
        """
        if x > 3000:
            x = 3000
        return -1.0 / 3000 * x * (x - 6000)
    
    def _pid_vx(self, is_middle_point: bool) -> float:
        """
        计算 vx（与静态规划 move.py 一致）
        """
        # PID 参数
        Kp = 1.5
        Ki = 0.01
        Kd = 0
        
        # 积分限幅
        if self.position_error_sum > 10000:
            self.position_error_sum = 10000
        
        # PID 输出
        output = (Kp * self.position_error + 
                  Ki * self.position_error_sum + 
                  Kd * (self.position_error - self.position_error_last))
        
        # 应用非线性速度规划
        output = self._speed_function(output)
        
        # 根据是否为中间点设置最小速度
        if is_middle_point:
            output = max(output, 1200)
        else:
            output = max(output, 450)
        
        # 最大速度限制
        if output > 4000:
            output = 4000
        
        return output
    
    def _pid_vw(self) -> float:
        """计算 vw（与静态规划 move.py 一致）"""
        # PID 参数
        Kp = 5
        Ki = 0.015
        Kd = 0.015
        
        # 积分限幅
        if self.theta_error_sum > 40:
            self.theta_error_sum = 40
        
        # PID 输出
        output = (Kp * self.theta_error + 
                  Ki * self.theta_error_sum + 
                  Kd * (self.theta_error - self.theta_error_last))
        
        # 限幅
        if output > 5:
            output = 5
        if output < -5:
            output = -5
        
        return output
    
    def reset(self):
        """重置状态"""
        self.position_error_last = 0.0
        self.position_error = 0.0
        self.position_error_sum = 0.0
        self.theta_error_last = 0.0
        self.theta_error = 0.0
        self.theta_error_sum = 0.0
        self.current_vx = 0.0
        self.current_vw = 0.0


# =============================================================================
# 统一运动控制器
# =============================================================================

class MotionController:
    """
    统一运动控制器
    
    整合路径跟踪和点对点控制
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        
        # 路径跟踪器
        self.path_tracker = PathTracker(cfg)
        
        # 点对点控制的朝向 PID
        self.heading_pid = PIDController(
            kp=3.0, ki=0.1, kd=0.3,
            output_limit=5.0
        )
        
        # 当前速度状态
        self.current_vx = 0.0
        self.current_vw = 0.0
    
    def compute_command(self, vision, path_x: List[float], path_y: List[float],
                        waypoint_index: int, goal: np.ndarray,
                        near_obstacle: bool = False) -> Tuple[float, float]:
        """
        计算路径跟踪速度命令
        
        Args:
            vision: 视觉对象
            path_x, path_y: 路径坐标
            waypoint_index: 当前路径点索引
            goal: 最终目标
            near_obstacle: 是否接近障碍物
            
        Returns:
            (vx, vw) 速度命令
        """
        if not path_x or waypoint_index >= len(path_x):
            return self.compute_command_to_point(
                vision.my_robot.x, vision.my_robot.y,
                vision.my_robot.orientation,
                Point(goal[0], goal[1])
            )
        
        # 使用路径跟踪控制器
        vx, vw = self.path_tracker.get_action(
            vision, path_x, path_y, waypoint_index
        )
        
        # 检查是否接近终点
        is_near_endpoint = waypoint_index >= len(path_x) - 2
        
        if is_near_endpoint:
            robot_pos = np.array([vision.my_robot.x, vision.my_robot.y])
            dist_to_goal = euclidean_distance(robot_pos, goal)
            if dist_to_goal < 500:
                vx *= max(0.3, dist_to_goal / 500)
        
        # 接近障碍物时减速
        if near_obstacle:
            vx *= 0.6
        
        # 更新当前速度
        self.current_vx = vx
        self.current_vw = vw
        
        return vx, vw
    
    def compute_command_to_point(self, robot_x: float, robot_y: float,
                                 robot_heading: float,
                                 target_point: Point) -> Tuple[float, float]:
        """
        计算到单点的速度命令
        
        Args:
            robot_x, robot_y: 机器人位置
            robot_heading: 机器人朝向
            target_point: 目标点
            
        Returns:
            (vx, vw) 速度命令
        """
        robot_pos = np.array([robot_x, robot_y])
        
        # 计算距离
        distance = euclidean_distance(robot_pos, target_point.to_array())
        
        # 计算期望朝向
        dx = target_point.x - robot_x
        dy = target_point.y - robot_y
        target_heading = math.atan2(dy, dx)
        
        # 朝向误差
        heading_error = angle_difference(target_heading, robot_heading)
        
        # 朝向控制
        vw = -self.heading_pid.update(heading_error)
        
        # 速度规划
        max_v = self.cfg.robot.MAX_VX
        
        # 距离减速
        slowdown_distance = 1000.0
        if distance < slowdown_distance:
            speed_factor = distance / slowdown_distance
            max_v *= max(0.2, speed_factor)
        
        # 方向误差大时，减速或停止
        abs_error = abs(heading_error)
        if abs_error > math.pi * 2 / 3:  # > 120度
            vx = 0
        elif abs_error > math.pi / 2:  # > 90度
            vx = max_v * 0.1
        elif abs_error > math.pi / 3:  # > 60度
            vx = max_v * 0.3
        elif abs_error > math.pi / 6:  # > 30度
            vx = max_v * 0.5
        else:
            vx = max_v
        
        # 非线性速度规划
        vx = self._speed_function(min(vx, 3000))
        
        # 更新当前速度
        self.current_vx = vx
        self.current_vw = vw
        
        return vx, vw
    
    def _speed_function(self, x: float) -> float:
        """非线性速度规划函数"""
        if x > 3000:
            x = 3000
        return -1.0 / 3000 * x * (x - 6000)
    
    def rotation_command(self, direction: int = 1) -> Tuple[float, float]:
        """生成原地旋转命令"""
        vw = direction * self.cfg.robot.MAX_VW * 0.6
        self.current_vx = 0.0
        self.current_vw = vw
        return 0.0, vw
    
    def stop_command(self) -> Tuple[float, float]:
        """生成停止命令"""
        self.current_vx = 0.0
        self.current_vw = 0.0
        return 0.0, 0.0
    
    def reset(self):
        """重置所有状态"""
        self.path_tracker.reset()
        self.heading_pid.reset()
        self.current_vx = 0.0
        self.current_vw = 0.0
