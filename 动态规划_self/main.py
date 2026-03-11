"""
机器人导航主程序
实现机器人在动态环境中的实时导航、避障与脱困规划。
采用 APF 人工势场法进行局部导航，PRM 概率路网进行脱困规划。
"""
import time
import numpy as np

from vision import Vision
from action import Action
from debug import Debugger
from plan import Navigator, vector_to_angle, normalize_angle
from move import MotionController, PositionTracker
from zss_debug_pb2 import Debug_Msgs
def main():
    """主函数"""
    # =========================================================================
    # 初始化
    # =========================================================================
    # 感知与通信模块
    vision = Vision()
    action = Action()
    debugger = Debugger()
    # 导航与运动控制
    navigator = Navigator()
    motion_ctrl = MotionController()
    pos_tracker = PositionTracker()
    # 启动阶段标志
    is_startup = True
    # =========================================================================
    # 主循环
    # =========================================================================
    while True:
        # ---------------------------------------------------------------------
        # 1. 获取机器人状态
        # ---------------------------------------------------------------------
        robot = vision.my_robot
        robot_pos = np.array([robot.x, robot.y])
        last_pos = pos_tracker.last_position
        # 记录位置
        pos_tracker.update(robot_pos)
        # 等待一帧
        time.sleep(0.01)
        # ---------------------------------------------------------------------
        # 2. 计算导航方向
        # ---------------------------------------------------------------------
        # 获取当前目标
        goal = navigator.get_current_goal()
        # 计算势场合力
        force = navigator.compute_navigation_force(robot_pos, vision)
        # 启动阶段不移动
        if is_startup:
            force = np.array([0.0, 0.0])
            is_startup = False
        # 放大力度作为期望速度
        target_velocity = force * 500
        # ---------------------------------------------------------------------
        # 3. 计算控制命令
        # ---------------------------------------------------------------------
        # 计算朝向
        if last_pos is not None:
            current_heading = motion_ctrl.estimate_heading(robot_pos, last_pos)
        else:
            current_heading = None
        target_heading = vector_to_angle(target_velocity)
        # 计算速度命令
        if current_heading is not None:
            # 计算角度误差
            heading_error = normalize_angle(target_heading - current_heading)
            cmd = motion_ctrl.compute_velocity_from_error(target_velocity, heading_error)
            # 调试输出
            print(f"目标朝向: {target_heading:.2f}, 当前朝向: {current_heading:.2f}, 误差: {heading_error:.2f}")
        else:
            cmd = motion_ctrl.stop_command()
        # ---------------------------------------------------------------------
        # 4. 状态切换与脱困处理
        # ---------------------------------------------------------------------
        if navigator.is_escaping():
            # 脱困状态：检查是否完成
            if navigator.check_escape_complete(robot_pos):
                navigator.end_escape()
        elif navigator.check_arrival(robot_pos):
            # 到达目标：切换目标点
            navigator.switch_goal()
            cmd = motion_ctrl.rotation_command()
            action.sendCommand(vx=cmd.vx, vw=cmd.vw)
            time.sleep(0.2)
        elif pos_tracker.check_stuck(threshold=10, min_count=200):
            # 卡住：启动脱困规划
            if navigator.start_escape(vision, robot_pos):
                cmd = motion_ctrl.rotation_command(direction=-1)
                pos_tracker.reset_stuck_count()
        # ---------------------------------------------------------------------
        # 5. 发送控制命令
        # ---------------------------------------------------------------------
        if not is_startup:
            action.sendCommand(vx=cmd.vx, vw=cmd.vw)
        # ---------------------------------------------------------------------
        # 6. 调试可视化
        # ---------------------------------------------------------------------
        package = Debug_Msgs()
        # 绘制机器人位置
        debugger.draw_circle(package, robot.x, robot.y)
        # 绘制目标点
        debugger.draw_circle(package, goal[0], goal[1])
        # 脱困状态时绘制路径
        if navigator.is_escaping() and len(navigator.path_x) > 0:
            debugger.draw_finalpath(package, navigator.path_x, navigator.path_y)
        debugger.send(package)
        # 控制循环频率
        time.sleep(0.005)

if __name__ == '__main__':
    main()