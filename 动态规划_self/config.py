"""
统一参数配置模块

包含所有算法和系统的参数配置。
"""

from dataclasses import dataclass
import numpy as np


# =============================================================================
# 场地参数
# =============================================================================

@dataclass
class FieldConfig:
    """场地参数配置"""
    MIN_X: float = -4500.0
    MAX_X: float = 4500.0
    MIN_Y: float = -3000.0
    MAX_Y: float = 3000.0
    
    @property
    def width(self) -> float:
        return self.MAX_X - self.MIN_X
    
    @property
    def height(self) -> float:
        return self.MAX_Y - self.MIN_Y


# =============================================================================
# 机器人参数
# =============================================================================

@dataclass
class RobotConfig:
    """机器人参数配置"""
    RADIUS: float = 200.0           # 机器人半径 (mm)
    MAX_VX: float = 3000.0          # 最大线速度 (mm/s)
    MAX_VY: float = 1000.0          # 最大侧向速度 (mm/s)
    MAX_VW: float = 12.0            # 最大角速度 (rad/s)
    MAX_AX: float = 3000.0          # 最大线加速度 (mm/s²)
    MAX_AW: float = 20.0            # 最大角加速度 (rad/s²)
    SAFE_DISTANCE: float = 300.0    # 安全距离余量 (mm)


# =============================================================================
# RRT* 参数
# =============================================================================

@dataclass
class RRTConfig:
    """RRT* 规划器参数配置"""
    MAX_ITERATIONS: int = 800           # 最大迭代次数
    STEP_SIZE: float = 300.0            # 扩展步长 (mm)
    GOAL_BIAS: float = 0.15             # 目标采样概率
    GOAL_THRESHOLD: float = 200.0       # 目标到达阈值 (mm)
    SEARCH_RADIUS: float = 800.0        # 近邻搜索半径 (mm)
    REWIRE_ENABLED: bool = True         # 是否启用重连优化
    SMOOTH_ITERATIONS: int = 50         # 路径平滑迭代次数


# =============================================================================
# DWA 参数
# =============================================================================

@dataclass
class DWAConfig:
    """DWA 局部规划器参数配置"""
    V_SAMPLES: int = 20                 # 线速度采样数
    W_SAMPLES: int = 40                 # 角速度采样数
    PREDICT_TIME: float = 2.0           # 轨迹预测时间 (s)
    PREDICT_STEP: float = 0.1           # 预测步长 (s)
    
    # 评估函数权重
    HEADING_WEIGHT: float = 0.4         # 朝向权重
    DIST_WEIGHT: float = 0.3            # 障碍距离权重
    VELOCITY_WEIGHT: float = 0.1        # 速度权重
    PATH_WEIGHT: float = 0.2            # 路径跟随权重
    
    # 安全参数
    OBSTACLE_THRESHOLD: float = 600.0   # 障碍物影响距离 (mm)
    MIN_OBSTACLE_DIST: float = 100.0    # 最小安全距离 (mm)


# =============================================================================
# Pure Pursuit 参数
# =============================================================================

@dataclass
class PurePursuitConfig:
    """Pure Pursuit 路径跟踪参数配置"""
    LOOKAHEAD_GAIN: float = 0.6         # 前视距离增益
    MIN_LOOKAHEAD: float = 200.0        # 最小前视距离 (mm)
    MAX_LOOKAHEAD: float = 1000.0       # 最大前视距离 (mm)
    WAYPOINT_THRESHOLD: float = 150.0   # 路径点到达阈值 (mm)


# =============================================================================
# PID 控制器参数
# =============================================================================

@dataclass
class PIDConfig:
    """PID 控制器参数配置"""
    KP: float = 1.5                     # 比例增益
    KI: float = 0.1                     # 积分增益
    KD: float = 0.5                    # 微分增益
    INTEGRAL_LIMIT: float = 3.0         # 积分限幅
    OUTPUT_LIMIT: float = 12.0          # 输出限幅 (最大角速度)


# =============================================================================
# 目标点配置
# =============================================================================

@dataclass
class GoalConfig:
    """目标点配置"""
    GOAL_A: np.ndarray = None
    GOAL_B: np.ndarray = None
    ARRIVAL_THRESHOLD: float = 150.0    # 到达阈值 (mm)
    
    def __post_init__(self):
        self.GOAL_A = np.array([-2400.0, -1500.0])
        self.GOAL_B = np.array([2400.0, 1500.0])


# =============================================================================
# 系统参数
# =============================================================================

@dataclass
class SystemConfig:
    """系统运行参数配置"""
    CONTROL_FREQUENCY: float = 100.0    # 控制频率 (Hz)
    REPLAN_INTERVAL: int = 30           # 重规划间隔 (帧)
    STUCK_THRESHOLD: float = 20.0       # 卡住检测阈值 (mm)
    STUCK_FRAMES: int = 100             # 卡住检测帧数
    STARTUP_DELAY: float = 0.5          # 启动延迟 (s)


# =============================================================================
# 统一配置类
# =============================================================================

class Config:
    """统一配置管理类"""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.field = FieldConfig()
        self.robot = RobotConfig()
        self.rrt = RRTConfig()
        self.dwa = DWAConfig()
        self.pure_pursuit = PurePursuitConfig()
        self.pid = PIDConfig()
        self.goal = GoalConfig()
        self.system = SystemConfig()
        
        self._initialized = True
    
    @property
    def total_safe_radius(self) -> float:
        """机器人总安全半径"""
        return self.robot.RADIUS + self.robot.SAFE_DISTANCE


# 全局配置实例
config = Config()
