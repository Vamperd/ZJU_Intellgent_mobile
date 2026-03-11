"""
移动控制模块

包含:
- PIDController: PID 控制器
- MotionController: 运动控制器
- VelocityCommand: 速度命令
"""

import math
import time
import numpy as np


# =============================================================================
# PID 控制器
# =============================================================================

class PIDConfig:
    """PID 参数配置"""
    KP = 0.3       # 比例增益
    KI = 0.0       # 积分增益
    KD = 0.001     # 微分增益
    WINDUP_GUARD = 20.0  # 积分限幅


class PIDController:
    """
    PID 控制器
    
    用于角度误差到角速度的转换控制。
    """
    
    def __init__(self, config: PIDConfig = None):
        self.config = config or PIDConfig()
        self.reset()
    
    def reset(self):
        """重置控制器状态"""
        self.setpoint = 0.0
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0
        self.last_error = 0.0
        self.int_error = 0.0
        self.output = 0.0
        self.last_time = time.time()
        self.sample_time = 0.0
    
    def update(self, feedback: float) -> float:
        """
        更新 PID 输出
        
        Args:
            feedback: 反馈值（误差）
            
        Returns:
            控制输出
        """
        error = self.setpoint - feedback
        current_time = time.time()
        delta_time = current_time - self.last_time
        delta_error = error - self.last_error
        
        if delta_time >= self.sample_time:
            # 比例项
            self.p_term = self.config.KP * error
            
            # 积分项
            self.i_term += error * delta_time
            self.i_term = max(-self.config.WINDUP_GUARD, 
                            min(self.config.WINDUP_GUARD, self.i_term))
            
            # 微分项
            self.d_term = 0.0
            if delta_time > 0:
                self.d_term = delta_error / delta_time
            
            self.last_time = current_time
            self.last_error = error
            
            # 计算输出
            self.output = (self.p_term + 
                          self.config.KI * self.i_term + 
                          self.config.KD * self.d_term)
        
        return self.output
    
    def set_gains(self, kp: float = None, ki: float = None, kd: float = None):
        """设置 PID 增益"""
        if kp is not None:
            self.config.KP = kp
        if ki is not None:
            self.config.KI = ki
        if kd is not None:
            self.config.KD = kd


# =============================================================================
# 速度命令
# =============================================================================

class VelocityCommand:
    """速度命令"""
    
    def __init__(self, vx: float = 0, vy: float = 0, vw: float = 0):
        self.vx = vx  # 前向速度
        self.vy = vy  # 侧向速度
        self.vw = vw  # 角速度
    
    def to_tuple(self) -> tuple:
        """转换为元组"""
        return (self.vx, self.vy, self.vw)


# =============================================================================
# 运动控制器
# =============================================================================

class MotionConfig:
    """运动控制参数配置"""
    ANGULAR_GAIN = 10.0       # 角速度增益
    LINEAR_GAIN = 2.0         # 线速度增益
    LINEAR_DAMPING = 2.0      # 线速度阻尼系数


class MotionController:
    """
    运动控制器
    
    将导航方向转换为速度命令，实现"转得越急，走得越慢"策略。
    """
    
    def __init__(self, pid_config: PIDConfig = None, motion_config: MotionConfig = None):
        self.pid = PIDController(pid_config)
        self.config = motion_config or MotionConfig()
    
    def estimate_heading(self, current_pos: np.ndarray, last_pos: np.ndarray) -> float:
        """
        估计当前朝向
        
        通过位移方向估计机器人朝向。
        
        Args:
            current_pos: 当前位置 [x, y]
            last_pos: 上一时刻位置 [x, y]
            
        Returns:
            估计的朝向角（弧度）
        """
        displacement = np.array(current_pos) - np.array(last_pos)
        
        if np.linalg.norm(displacement) < 1e-6:
            return None  # 位移太小，无法估计
        
        return math.atan2(displacement[1], displacement[0])
    
    def compute_velocity(self, 
                         target_force: np.ndarray,
                         current_heading: float,
                         target_heading: float) -> VelocityCommand:
        """
        计算速度命令
        
        Args:
            target_force: 目标力/速度向量
            current_heading: 当前朝向角
            target_heading: 目标朝向角
            
        Returns:
            VelocityCommand 速度命令
        """
        # 计算角度误差
        heading_error = self._normalize_angle(target_heading - current_heading)
        
        # PID 计算角速度修正
        self.pid.setpoint = 0
        pid_output = self.pid.update(heading_error)
        
        # 角速度
        vw = -pid_output * self.config.ANGULAR_GAIN
        
        # 线速度（角度误差大时减小）
        force_magnitude = np.linalg.norm(target_force)
        vx = (force_magnitude * self.config.LINEAR_GAIN / 
              (1 + self.config.LINEAR_DAMPING * abs(pid_output)))
        
        return VelocityCommand(vx=vx, vy=0, vw=vw)
    
    def compute_velocity_from_error(self,
                                    target_force: np.ndarray,
                                    heading_error: float) -> VelocityCommand:
        """
        从角度误差直接计算速度命令
        
        Args:
            target_force: 目标力/速度向量
            heading_error: 已归一化的角度误差
            
        Returns:
            VelocityCommand 速度命令
        """
        # PID 计算角速度修正
        self.pid.setpoint = 0
        pid_output = self.pid.update(heading_error)
        
        # 角速度
        vw = -pid_output * self.config.ANGULAR_GAIN
        
        # 线速度（角度误差大时减小）
        force_magnitude = np.linalg.norm(target_force)
        vx = (force_magnitude * self.config.LINEAR_GAIN / 
              (1 + self.config.LINEAR_DAMPING * abs(pid_output)))
        
        return VelocityCommand(vx=vx, vy=0, vw=vw)
    
    def rotation_command(self, direction: int = 1) -> VelocityCommand:
        """
        生成原地旋转命令
        
        Args:
            direction: 旋转方向，1 为正向，-1 为反向
            
        Returns:
            VelocityCommand 速度命令
        """
        return VelocityCommand(vx=0, vy=0, vw=direction * 10)
    
    def stop_command(self) -> VelocityCommand:
        """生成停止命令"""
        return VelocityCommand(vx=0, vy=0, vw=0)
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """将角度归一化到 [-pi, pi]"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle


# =============================================================================
# 位置跟踪器
# =============================================================================

class PositionTracker:
    """
    位置跟踪器
    
    用于记录和比较位置变化，检测是否卡住。
    """
    
    def __init__(self):
        self.last_position = None
        self.reference_position = None
        self.stuck_count = 0
    
    def update(self, position: np.ndarray):
        """更新当前位置"""
        position = np.array(position)
        
        if self.last_position is None:
            self.last_position = position.copy()
            self.reference_position = position.copy()
        
        self.last_position = position.copy()
    
    def get_displacement(self) -> float:
        """获取与参考位置的距离"""
        if self.reference_position is None:
            return 0
        return np.linalg.norm(self.last_position - self.reference_position)
    
    def check_stuck(self, threshold: float = 10, min_count: int = 200) -> bool:
        """检查是否卡住"""
        if self.get_displacement() < threshold:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
            self.reset_reference()
        
        return self.stuck_count > min_count
    
    def reset_reference(self):
        """重置参考位置"""
        if self.last_position is not None:
            self.reference_position = self.last_position.copy()
    
    def reset_stuck_count(self):
        """重置卡住计数"""
        self.stuck_count = 0
