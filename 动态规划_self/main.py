"""
机器人导航主程序

实现机器人在动态环境中的实时导航、避障与脱困规划。
采用 A* 全局规划 + 改进路径跟踪的混合架构。
"""

import time
import math
import numpy as np
from enum import Enum, auto
from typing import List, Optional, Tuple

from vision import Vision
from action import Action
from debug import Debugger
from zss_debug_pb2 import Debug_Msgs

from config import Config, config as global_config
from utils import (Point, Circle, Path, extract_obstacles, 
                   euclidean_distance, angle_difference)
from planner import PathManager
from local import DWAPlanner, RobotState, EmergencyAvoidance
from controller import MotionController


# =============================================================================
# 状态机定义
# =============================================================================

class NavigationState(Enum):
    """导航状态"""
    IDLE = auto()
    TO_GOAL_A = auto()
    TO_GOAL_B = auto()
    ARRIVED = auto()
    EMERGENCY = auto()


class NavigationStateMachine:
    """导航状态机"""
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        self.state = NavigationState.TO_GOAL_A
        self._goal_switch_count = 0
    
    def get_current_goal(self) -> np.ndarray:
        """获取当前目标"""
        if self.state == NavigationState.TO_GOAL_A:
            return self.cfg.goal.GOAL_A
        elif self.state == NavigationState.TO_GOAL_B:
            return self.cfg.goal.GOAL_B
        else:
            return self.cfg.goal.GOAL_A
    
    def switch_goal(self):
        """切换目标"""
        if self.state == NavigationState.TO_GOAL_A:
            self.state = NavigationState.TO_GOAL_B
        elif self.state == NavigationState.TO_GOAL_B:
            self.state = NavigationState.TO_GOAL_A
        self._goal_switch_count += 1
    
    def set_state(self, state: NavigationState):
        self.state = state
    
    def is_navigating(self) -> bool:
        return self.state in [NavigationState.TO_GOAL_A, NavigationState.TO_GOAL_B]
    
    @property
    def goal_switch_count(self) -> int:
        return self._goal_switch_count


# =============================================================================
# 卡住检测器
# =============================================================================

class StuckDetector:
    """卡住检测器"""
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        self.reference_pos = None
        self.stuck_count = 0
    
    def update(self, current_pos: np.ndarray) -> bool:
        if self.reference_pos is None:
            self.reference_pos = current_pos.copy()
            return False
        
        distance = euclidean_distance(current_pos, self.reference_pos)
        
        if distance < self.cfg.system.STUCK_THRESHOLD:
            self.stuck_count += 1
        else:
            self.stuck_count = 0
            self.reference_pos = current_pos.copy()
        
        return self.stuck_count > self.cfg.system.STUCK_FRAMES
    
    def reset(self):
        self.reference_pos = None
        self.stuck_count = 0


# =============================================================================
# 路径点管理器
# =============================================================================

class WaypointManager:
    """
    路径点管理器
    
    负责管理当前路径点索引和切换逻辑
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        self.current_index = 0
        self.path_x = []
        self.path_y = []
    
    def set_path(self, path_x: List[float], path_y: List[float]):
        """设置新路径"""
        self.path_x = path_x
        self.path_y = path_y
        self.current_index = 0
    
    def clear(self):
        """清除路径"""
        self.path_x = []
        self.path_y = []
        self.current_index = 0
    
    def update(self, robot_pos: np.ndarray) -> bool:
        """
        更新路径点索引
        
        Returns:
            是否前进到下一个路径点
        """
        if not self.path_x or self.current_index >= len(self.path_x):
            return False
        
        # 计算到当前目标点的距离
        current_wp = np.array([self.path_x[self.current_index], 
                               self.path_y[self.current_index]])
        dist = euclidean_distance(robot_pos, current_wp)
        
        # 路径点切换阈值（增大以减少频繁切换）
        threshold = self.cfg.pure_pursuit.WAYPOINT_THRESHOLD
        
        # 如果接近当前路径点且不是最后一个，前进到下一个
        if dist < threshold and self.current_index < len(self.path_x) - 1:
            self.current_index += 1
            return True
        
        # 检查是否可以跳过当前路径点（已经越过）
        if self.current_index < len(self.path_x) - 1:
            # 计算到下一个点的距离
            next_wp = np.array([self.path_x[self.current_index + 1],
                               self.path_y[self.current_index + 1]])
            dist_to_next = euclidean_distance(robot_pos, next_wp)
            
            # 如果下一个点更近，直接跳过
            if dist_to_next < dist:
                self.current_index += 1
                return True
        
        return False
    
    def get_current_waypoint(self) -> Optional[Tuple[float, float]]:
        """获取当前路径点"""
        if self.current_index < len(self.path_x):
            return (self.path_x[self.current_index], 
                    self.path_y[self.current_index])
        return None
    
    def is_complete(self) -> bool:
        """是否已完成路径"""
        return self.current_index >= len(self.path_x) - 1
    
    def get_remaining_path(self) -> Tuple[List[float], List[float]]:
        """获取剩余路径"""
        return (self.path_x[self.current_index:],
                self.path_y[self.current_index:])


# =============================================================================
# 主控制器
# =============================================================================

class NavigationController:
    """导航控制器"""
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        
        # 感知与执行
        self.vision = Vision()
        self.action = Action()
        self.debugger = Debugger()
        
        # 导航模块
        self.state_machine = NavigationStateMachine(cfg)
        self.path_manager = PathManager(cfg)
        self.waypoint_manager = WaypointManager(cfg)
        self.motion_controller = MotionController(cfg)
        self.dwa_planner = DWAPlanner(cfg)
        self.emergency_avoidance = EmergencyAvoidance(cfg)
        
        # 状态检测
        self.stuck_detector = StuckDetector(cfg)
        
        # 计数器
        self.frame_count = 0
        self.replan_counter = 0
    
    def run(self):
        """主循环"""
        print("导航系统启动...")
        print("使用 A* 全局规划 + 优化路径跟踪")
        time.sleep(self.cfg.system.STARTUP_DELAY)
        
        while True:
            try:
                self._update()
            except Exception as e:
                print(f"错误: {e}")
                import traceback
                traceback.print_exc()
                self.action.sendCommand(vx=0, vw=0)
                time.sleep(0.1)
            
            time.sleep(1.0 / self.cfg.system.CONTROL_FREQUENCY)
            self.frame_count += 1
    
    def _update(self):
        """单帧更新"""
        # -----------------------------------------------------------------
        # 1. 获取状态
        # -----------------------------------------------------------------
        robot = self.vision.my_robot
        robot_pos = np.array([robot.x, robot.y])
        robot_heading = robot.orientation
        
        robot_state = RobotState(
            x=robot.x, y=robot.y,
            heading=robot_heading,
            vx=self.motion_controller.current_vx,
            vw=self.motion_controller.current_vw
        )
        
        obstacles = extract_obstacles(self.vision, include_teammates=True)
        goal = self.state_machine.get_current_goal()
        
        # -----------------------------------------------------------------
        # 2. 全局规划
        # -----------------------------------------------------------------
        should_replan = (
            self.path_manager.current_path is None or
            self.path_manager.path_blocked or
            self.replan_counter >= self.cfg.system.REPLAN_INTERVAL or
            self.stuck_detector.stuck_count > 50
        )
        
        if should_replan and self.state_machine.is_navigating():
            success = self.path_manager.replan_path(robot_pos, goal, obstacles)
            if success:
                self.replan_counter = 0
                self.stuck_detector.reset()
                
                # 更新路径点管理器
                path_x, path_y = self.path_manager.get_path_arrays()
                self.waypoint_manager.set_path(path_x, path_y)
                
                print(f"[帧 {self.frame_count}] 规划新路径，长度: {len(path_x)}")
        
        self.replan_counter += 1
        
        # -----------------------------------------------------------------
        # 3. 更新路径点索引
        # -----------------------------------------------------------------
        self.waypoint_manager.update(robot_pos)
        
        # -----------------------------------------------------------------
        # 4. 紧急避障检查
        # -----------------------------------------------------------------
        is_emergency, emergency_cmd = self.emergency_avoidance.check_emergency(
            robot_state, obstacles
        )
        
        if is_emergency and emergency_cmd:
            vx, vw = emergency_cmd.vx, emergency_cmd.vw
            self.state_machine.set_state(NavigationState.EMERGENCY)
        else:
            if self.state_machine.state == NavigationState.EMERGENCY:
                self.state_machine.set_state(
                    NavigationState.TO_GOAL_A 
                    if np.linalg.norm(goal - self.cfg.goal.GOAL_A) < 1
                    else NavigationState.TO_GOAL_B
                )
            
            # -------------------------------------------------------------
            # 5. 路径跟踪
            # -------------------------------------------------------------
            path_x = self.waypoint_manager.path_x
            path_y = self.waypoint_manager.path_y
            waypoint_index = self.waypoint_manager.current_index
            
            if path_x and waypoint_index < len(path_x):
                # 检查前方障碍物
                path_blocked_ahead = self.path_manager.check_path_blocked(
                    robot_pos, obstacles
                )
                
                # 使用优化的路径跟踪控制器
                vx, vw = self.motion_controller.compute_command(
                    self.vision, path_x, path_y,
                    waypoint_index, goal,
                    near_obstacle=path_blocked_ahead
                )
                
                if self.frame_count % 10 == 0:
                    wp = self.waypoint_manager.get_current_waypoint()
                    if wp:
                        print(f"  -> 跟踪路径点[{waypoint_index}/{len(path_x)-1}]: "
                              f"({wp[0]:.0f}, {wp[1]:.0f}), vx={vx:.0f}, vw={vw:.2f}")
            else:
                # 直接控制到目标
                vx, vw = self.motion_controller.compute_command_to_point(
                    robot.x, robot.y, robot_heading,
                    Point(goal[0], goal[1])
                )
                
                if self.frame_count % 10 == 0:
                    print(f"  -> 直接控制到目标: ({goal[0]:.0f}, {goal[1]:.0f})")
        
        # -----------------------------------------------------------------
        # 6. 到达检测
        # -----------------------------------------------------------------
        distance_to_goal = euclidean_distance(robot_pos, goal)
        
        # 到达条件：距离足够近且速度较低
        if distance_to_goal < self.cfg.goal.ARRIVAL_THRESHOLD:
            if self.motion_controller.current_vx < 500 or distance_to_goal < 100:
                print(f"[帧 {self.frame_count}] 到达目标! 切换中...")
                self.state_machine.switch_goal()
                self.path_manager.current_path = None
                self.waypoint_manager.clear()
                self.motion_controller.reset()
                self.stuck_detector.reset()
                
                # 原地旋转
                vx, vw = self.motion_controller.rotation_command()
                self.action.sendCommand(vx=vx, vw=vw)
                time.sleep(0.3)
                return
        
        # -----------------------------------------------------------------
        # 7. 卡住检测
        # -----------------------------------------------------------------
        if self.stuck_detector.update(robot_pos):
            print(f"[帧 {self.frame_count}] 检测到卡住，触发重规划")
            self.path_manager.current_path = None
            self.waypoint_manager.clear()
            self.stuck_detector.reset()
        
        # -----------------------------------------------------------------
        # 8. 执行命令
        # -----------------------------------------------------------------
        self.action.sendCommand(vx=vx, vw=vw)
        
        if self.frame_count % 10 == 0:
            print(f"[帧 {self.frame_count}] 位置: ({robot.x:.0f}, {robot.y:.0f}), "
                  f"朝向: {math.degrees(robot_heading):.1f}°, "
                  f"速度: vx={vx:.0f}, vw={vw:.2f}, "
                  f"到目标距离: {distance_to_goal:.0f}")
        
        # -----------------------------------------------------------------
        # 9. 调试可视化
        # -----------------------------------------------------------------
        self._send_debug_info(robot_pos, goal, obstacles)
    
    def _send_debug_info(self, robot_pos: np.ndarray, goal: np.ndarray, 
                         obstacles: list):
        """发送调试信息"""
        package = Debug_Msgs()
        
        self.debugger.draw_circle(package, robot_pos[0], robot_pos[1])
        self.debugger.draw_circle(package, goal[0], goal[1])
        
        if self.path_manager.current_path:
            path_x, path_y = self.path_manager.get_path_arrays()
            if path_x and path_y:
                self.debugger.draw_finalpath(package, path_x, path_y)
        
        self.debugger.send(package)


# =============================================================================
# 主函数
# =============================================================================

def main():
    controller = NavigationController()
    controller.run()


if __name__ == '__main__':
    main()
