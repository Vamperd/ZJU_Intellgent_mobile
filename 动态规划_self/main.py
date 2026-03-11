"""
机器人导航主程序

实现机器人在动态环境中的实时导航、避障与脱困规划。
采用 A* 全局规划 + 改进的 DWA 局部避障 + Pure Pursuit 路径跟踪的混合架构。

往返目标: 在起点与终点之间自动往返 5 次
"""

import time
import math
import numpy as np
from enum import Enum, auto

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
    IDLE = auto()           # 空闲状态
    TO_GOAL_A = auto()      # 前往目标A
    TO_GOAL_B = auto()      # 前往目标B
    ARRIVED = auto()        # 到达目标
    REPLANNING = auto()     # 重规划中
    EMERGENCY = auto()      # 紧急避障
    COMPLETED = auto()       # 完成所有往返


# =============================================================================
# 导航状态机
# =============================================================================

class NavigationStateMachine:
    """
    导航状态机
    
    管理机器人的导航状态切换
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        self.state = NavigationState.TO_GOAL_A
        self._goal_switch_count = 0
        self._max_trips = 5
        self._total_completed_trips = 0
    
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
            self._goal_switch_count = 1
        elif self.state == NavigationState.TO_GOAL_B:
            self.state = NavigationState.TO_GOAL_A
            self._goal_switch_count = 2
            self._total_completed_trips += 1
            
            if self._total_completed_trips >= self._max_trips:
                self.state = NavigationState.COMPLETED
                print(f"=== 完成所有 {self._max_trips} 次往返任务! ===")
    
    def set_state(self, state: NavigationState):
        """设置状态"""
        self.state = state
    
    def is_navigating(self) -> bool:
        """是否正在导航"""
        return self.state in [NavigationState.TO_GOAL_A, NavigationState.TO_GOAL_B]
    
    @property
    def goal_switch_count(self) -> int:
        """获取目标切换次数"""
        return self._goal_switch_count
    
    @property
    def total_completed_trips(self) -> int:
        """获取已完成的往返次数"""
        return self._total_completed_trips
    
    @property
    def max_trips(self) -> int:
        """获取目标往返次数"""
        return self._max_trips
    
    def get_progress_info(self) -> str:
        """获取进度信息"""
        if self.state == NavigationState.COMPLETED:
            return f"完成! 共 {self._total_completed_trips} 次往返"
        else:
            current_trip = self._total_completed_trips + 1
            direction = "A->B" if self.state == NavigationState.TO_GOAL_B else "B->A"
            return f"第 {current_trip}/{self._max_trips} 次往返 ({direction})"


# =============================================================================
# 卡住检测器
# =============================================================================

class StuckDetector:
    """
    卡住检测器
    
    检测机器人是否卡住
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        self.reference_pos = None
        self.stuck_count = 0
    
    def update(self, current_pos: np.ndarray) -> bool:
        """
        更新检测状态
        
        Args:
            current_pos: 当前位置
            
        Returns:
            是否卡住
        """
        if self.reference_pos is None:
            self.reference_pos = current_pos.copy()
            return False
        
        # 计算移动距离
        distance = euclidean_distance(current_pos, self.reference_pos)
        
        if distance < self.cfg.system.STUCK_THRESHOLD:
            self.stuck_count += 1
        else:
            # 移动了，重置
            self.stuck_count = 0
            self.reference_pos = current_pos.copy()
        
        return self.stuck_count > self.cfg.system.STUCK_FRAMES
    
    def reset(self):
        """重置检测器"""
        self.reference_pos = None
        self.stuck_count = 0


# =============================================================================
# 主控制器
# =============================================================================

class NavigationController:
    """
    导航控制器
    
    整合所有模块，实现完整的导航功能
    """
    
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or global_config
        
        # 感知与执行
        self.vision = Vision()
        self.action = Action()
        self.debugger = Debugger()
        
        # 导航模块
        self.state_machine = NavigationStateMachine(cfg)
        self.path_manager = PathManager(cfg)
        self.dwa_planner = DWAPlanner(cfg)
        self.motion_controller = MotionController(cfg)
        self.emergency_avoidance = EmergencyAvoidance(cfg)
        
        # 状态检测
        self.stuck_detector = StuckDetector(cfg)
        
        # 计数器
        self.frame_count = 0
        self.replan_counter = 0
    
    def run(self):
        """主循环"""
        print("导航系统启动...")
        time.sleep(self.cfg.system.STARTUP_DELAY)
        
        while True:
            try:
                self._update()
            except Exception as e:
                print(f"错误: {e}")
                self.action.sendCommand(vx=0, vw=0)
                time.sleep(0.1)
            
            # 控制频率
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
        
        # 构建机器人状态
        robot_state = RobotState(
            x=robot.x,
            y=robot.y,
            heading=robot_heading,
            vx=self.motion_controller.current_vx,
            vw=self.motion_controller.current_vw
        )
        
        # 提取障碍物
        obstacles = extract_obstacles(self.vision, include_teammates=True)
        
        # 获取当前目标
        goal = self.state_machine.get_current_goal()
        
        # -----------------------------------------------------------------
        # 2. 全局规划
        # -----------------------------------------------------------------
        should_replan = (
            self.path_manager.current_path is None or
            self.replan_counter >= self.cfg.system.REPLAN_INTERVAL
        )
        
        if should_replan and self.state_machine.is_navigating():
            success = self.path_manager.plan_new_path(robot_pos, goal, obstacles)
            if success:
                self.replan_counter = 0
                self.stuck_detector.reset()
                print(f"[帧 {self.frame_count}] 规划新路径，长度: {len(self.path_manager.current_path)}")
        
        self.replan_counter += 1
        
        # -----------------------------------------------------------------
        # 3. 紧急避障检查
        # -----------------------------------------------------------------
        is_emergency, emergency_cmd = self.emergency_avoidance.check_emergency(
            robot_state, obstacles
        )
        
        if is_emergency and emergency_cmd:
            vx, vw = emergency_cmd.vx, emergency_cmd.vw
            self.state_machine.set_state(NavigationState.EMERGENCY)
            self.emergency_counter = 0
        else:
            if self.state_machine.state == NavigationState.EMERGENCY:
                if not hasattr(self, 'emergency_counter'):
                    self.emergency_counter = 0
                self.emergency_counter += 1
                
                if self.emergency_counter > 20:
                    if self.state_machine._goal_switch_count % 2 == 0:
                        self.state_machine.set_state(NavigationState.TO_GOAL_A)
                    else:
                        self.state_machine.set_state(NavigationState.TO_GOAL_B)
                    self.emergency_counter = 0
            
            # -------------------------------------------------------------
            # 4. 局部规划 + 控制
            # -------------------------------------------------------------
            path = self.path_manager.get_remaining_path()
            
            if path and not path.is_empty:
                # 使用 DWA 计算速度
                target_point = self._get_target_point(path)
                cmd = self.dwa_planner.compute_velocity(
                    robot_state, target_point, path, obstacles
                )
                vx, vw = cmd.vx, cmd.vw
                
                # 更新路径点
                self._update_waypoint(robot_pos, path)
                
                # 调试：显示目标点
                if self.frame_count % 10 == 0:
                    print(f"  -> 跟踪路径点: ({target_point.x:.0f}, {target_point.y:.0f}), "
                          f"路径剩余: {len(path)} 点")
            else:
                # 直接控制到目标
                vx, vw = self.motion_controller.compute_command_to_point(
                    robot_pos, robot_heading, Point(goal[0], goal[1])
                )
                
                if self.frame_count % 10 == 0:
                    print(f"  -> 直接控制到目标: ({goal[0]:.0f}, {goal[1]:.0f})")
        
        # -----------------------------------------------------------------
        # 5. 到达检测 (需要连续多帧检测通过)
        # -----------------------------------------------------------------
        distance_to_goal = euclidean_distance(robot_pos, goal)
        
        if not hasattr(self, 'arrival_counter'):
            self.arrival_counter = 0
        
        if distance_to_goal < self.cfg.goal.ARRIVAL_THRESHOLD:
            self.arrival_counter += 1
        else:
            self.arrival_counter = 0
        
        if self.arrival_counter >= 10:
            if self.state_machine.state == NavigationState.COMPLETED:
                print(f"[帧 {self.frame_count}] 所有任务已完成，停止!")
                self.action.sendCommand(vx=0, vw=0)
                time.sleep(1.0)
                return
            
            print(f"[帧 {self.frame_count}] 到达目标! {self.state_machine.get_progress_info()}")
            self.state_machine.switch_goal()
            
            if self.state_machine.state == NavigationState.COMPLETED:
                print(f"=== 完成所有 {self.state_machine.max_trips} 次往返任务! ===")
                self.path_manager.current_path = None
                self.motion_controller.reset()
                self.stuck_detector.reset()
                vx, vw = 0, 0
                self.action.sendCommand(vx=vx, vw=vw)
                return
            
            self.path_manager.current_path = None
            self.motion_controller.reset()
            self.stuck_detector.reset()
            
            vx, vw = self.motion_controller.rotation_command()
            self.action.sendCommand(vx=vx, vw=vw)
            time.sleep(0.3)
            return
        
        # -----------------------------------------------------------------
        # 6. 接近目标时简化控制
        # -----------------------------------------------------------------
        approaching_goal = distance_to_goal < 500
        
        # -----------------------------------------------------------------
        # 7. 卡住检测 (接近目标时跳过卡住检测)
        # -----------------------------------------------------------------
        if not approaching_goal and self.stuck_detector.update(robot_pos):
            current_goal = self.state_machine.get_current_goal()
            dist_to_current_goal = euclidean_distance(robot_pos, current_goal)
            
            if dist_to_current_goal > 300:
                print(f"[帧 {self.frame_count}] 检测到卡住，触发重规划")
                self.path_manager.current_path = None
                self.stuck_detector.reset()
        
        # -----------------------------------------------------------------
        # 7. 执行命令
        # -----------------------------------------------------------------
        self.action.sendCommand(vx=vx, vw=vw)
        
        # 调试输出
        if self.frame_count % 10 == 0:
            progress = self.state_machine.get_progress_info()
            print(f"[帧 {self.frame_count}] 位置: ({robot.x:.0f}, {robot.y:.0f}), "
                  f"朝向: {math.degrees(robot_heading):.1f}°, "
                  f"速度: vx={vx:.0f}, vw={vw:.2f}, "
                  f"到目标距离: {distance_to_goal:.0f}, "
                  f"进度: {progress}")
        
        # -----------------------------------------------------------------
        # 8. 调试可视化
        # -----------------------------------------------------------------
        self._send_debug_info(robot_pos, goal, obstacles)
    
    def _get_target_point(self, path: Path) -> Point:
        """获取目标跟踪点"""
        if path.is_empty:
            return Point(0, 0)
        
        robot = self.vision.my_robot
        robot_pos = np.array([robot.x, robot.y])
        goal = self.state_machine.get_current_goal()
        dist_to_goal = euclidean_distance(robot_pos, goal)
        
        # 接近目标时，直接以目标点为跟踪点
        if dist_to_goal < 600:
            return Point(goal[0], goal[1])
        
        # 正常前视距离
        lookahead = min(600, self.motion_controller.current_vx * 0.2 + 200)
        
        for point in path:
            if euclidean_distance(robot_pos, point.to_array()) >= lookahead:
                return point
        
        return path.end
    
    def _update_waypoint(self, robot_pos: np.ndarray, path: Path):
        """更新路径点索引"""
        if path.is_empty:
            return
        
        current_wp = self.path_manager.get_current_waypoint()
        if current_wp:
            dist = euclidean_distance(robot_pos, current_wp.to_array())
            if dist < self.cfg.pure_pursuit.WAYPOINT_THRESHOLD:
                self.path_manager.advance_waypoint()
    
    def _send_debug_info(self, robot_pos: np.ndarray, goal: np.ndarray, 
                         obstacles: list):
        """发送调试信息"""
        package = Debug_Msgs()
        
        # 绘制机器人位置 (蓝色)
        self.debugger.draw_circle(package, robot_pos[0], robot_pos[1])
        
        # 绘制两个目标点
        self.debugger.draw_circle(package, self.cfg.goal.GOAL_A[0], self.cfg.goal.GOAL_A[1], radius=200)
        self.debugger.draw_circle(package, self.cfg.goal.GOAL_B[0], self.cfg.goal.GOAL_B[1], radius=200)
        
        # 绘制当前目标 (黄色高亮)
        self.debugger.draw_circle(package, goal[0], goal[1], radius=250)
        
        # 绘制路径 (绿色)
        if self.path_manager.current_path:
            path_x, path_y = self.path_manager.get_path_arrays()
            if path_x and path_y:
                self.debugger.draw_finalpath(package, path_x, path_y)
        
        # 绘制障碍物 (红色)
        for obs in obstacles:
            self.debugger.draw_circle(package, obs.center.x, obs.center.y, 
                                    radius=obs.radius)
        
        # 发送
        self.debugger.send(package)


# =============================================================================
# 主函数
# =============================================================================

def main():
    """主函数"""
    controller = NavigationController()
    controller.run()


if __name__ == '__main__':
    main()
