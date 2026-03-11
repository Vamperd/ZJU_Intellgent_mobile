from numpy import append
from vision import Vision
from action import Action
from debug import Debugger
import numpy as np
from zss_debug_pb2 import Debug_Msgs, Debug_Msg
from prm import PRM
import time
from func import Force, angle
import pid

# 目标点设置
goal_A = np.array([2400, 1500])   # A点
goal_B = np.array([-2400, -1500]) # B点

# 状态变量
flag = 1           # 1: 去B点, -1: 去A点
count = 0
w_real = -3.14
recode = np.array([0, 0])
repeat = 0

# 往返记录相关变量
round_trip_count = 0      # 已完成往返次数
max_round_trips = 5       # 最大往返次数
trip_start_time = None    # 单次行程开始时间
round_trip_times = []     # 记录每次往返的时间
current_leg_start = None  # 当前路段开始时间
total_start_time = None   # 总开始时间
is_return_trip = False    # 是否为返程

# 末端加速参数
approach_distance = 500   # 开始加速的距离阈值
speed_boost_factor = 2.5  # 速度提升因子

if __name__ == '__main__':
    vision = Vision()
    action = Action()
    debugger = Debugger()
    planner = PRM()
    w_pid = pid.PID()
    
    print("=" * 50)
    print("人工势场法路径规划 - 往返测试")
    print("=" * 50)
    
    while True:
        # 1. 获取机器人位置信息
        robot = vision.my_robot
        robot_last_x = robot.x
        robot_last_y = robot.y
        recode = np.array([robot_last_x, robot_last_y])
        time.sleep(0.01)
        
        # 设置目标点
        if flag == 1:
            goal = goal_B
        elif flag == -1:
            goal = goal_A
        
        # 计算人工势场力（返回详细力信息用于可视化）
        force_result = Force([robot.x, robot.y], vision, goal)
        F_total, Fatt, Frep, obs_positions, Frep_list = force_result
        
        d_goal = np.linalg.norm(np.array([robot.x, robot.y]) - goal)
        
        if count == 0:
            F_display = np.array([0, 0])
            count = 1
        else:
            F_display = F_total
            count = 2
        
        # 末端加速：当靠近目标点时增加速度
        speed_multiplier = 1.0
        if d_goal < approach_distance:
            # 距离越近，速度提升越大
            speed_multiplier = 1.0 + (approach_distance - d_goal) / approach_distance * (speed_boost_factor - 1.0)
            speed_multiplier = min(speed_multiplier, speed_boost_factor)  # 限制最大倍数
        
        totalv = F_display * 500 * speed_multiplier
        
        v_real = [robot.x - robot_last_x, robot.y - robot_last_y]
        if v_real[0] != 0:
            w_real = angle(v_real)
        
        w_expect = angle(totalv) if np.linalg.norm(totalv) > 0.001 else w_real
        
        err = w_expect - w_real
        if err > np.pi:
            err = err - np.pi * 2
        elif err < -np.pi:
            err = err + np.pi * 2
        
        w_pid.SetPoint = 0
        pid.PID.update(w_pid, err)
        
        # 打印调试信息
        if count == 2:
            print(f"目标: {goal}, 距离: {d_goal:.1f}, 速度倍数: {speed_multiplier:.2f}, 往返次数: {round_trip_count}/{max_round_trips}")
        
        # 到达目标点检测
        if flag == 0:
            if d_goal < d_prm:
                flag = _flag
        elif d_goal < 150:  # 到达目标点阈值
            # 记录到达时间
            if current_leg_start is not None:
                leg_time = time.time() - current_leg_start
                print(f"  -> 到达目标点! 耗时: {leg_time:.2f}秒")
            
            # 切换目标点
            flag = -1 * flag
            action.sendCommand(vx=0, vw=10)
            time.sleep(0.2)
            
            # 往返计数逻辑
            if flag == 1:  # 刚刚到达A点，准备返回B点（完成一次往返）
                if is_return_trip and round_trip_count < max_round_trips:
                    round_trip_count += 1
                    if trip_start_time is not None:
                        trip_time = time.time() - trip_start_time
                        round_trip_times.append(trip_time)
                        print(f"*** 完成第 {round_trip_count} 次往返! 耗时: {trip_time:.2f}秒 ***")
                is_return_trip = True
                trip_start_time = time.time()
                print(f"--- 开始第 {round_trip_count + 1} 次往返 ---")
            
            current_leg_start = time.time()
            
            # 检查是否完成所有往返
            if round_trip_count >= max_round_trips:
                print("=" * 50)
                print("完成所有往返测试!")
                print("=" * 50)
                total_time = time.time() - total_start_time if total_start_time else 0
                print(f"总时间: {total_time:.2f}秒")
                print("各次往返时间:")
                for i, t in enumerate(round_trip_times, 1):
                    print(f"  第{i}次: {t:.2f}秒")
                if round_trip_times:
                    print(f"平均往返时间: {np.mean(round_trip_times):.2f}秒")
                print("=" * 50)
                action.sendCommand(vx=0, vw=0)
                break
        
        # 2. 发送控制命令
        if count == 2:
            vx = np.linalg.norm(totalv) * 2 / (1 + 2 * abs(w_pid.output))
            vw = -w_pid.output * 10
            action.sendCommand(vx=vx, vw=vw)
        
        # 检测是否卡住
        d_move = np.linalg.norm(np.array([vision.my_robot.x, vision.my_robot.y]) - recode)
        if d_move < 10:
            repeat += 1
            if repeat > 200:
                # 使用PRM规划逃脱
                path_x, path_y, road_map, sample_x, sample_y = planner.plan(
                    vision=vision,
                    start_x=vision.my_robot.x,
                    start_y=vision.my_robot.y,
                    goal_x=goal[0],
                    goal_y=goal[1]
                )
                l = len(path_x)
                if l >= 2:
                    goal = np.array([path_x[l-2], path_y[l-2]])
                action.sendCommand(vx=0, vw=-w_pid.output * 10)
                
                d_prm = np.linalg.norm(recode - goal) / 2
                if d_prm < 300:
                    d_prm = 999999
                elif d_prm < 1000:
                    d_prm = 500
                
                if flag != 0:
                    _flag = flag
                flag = 0
                repeat = 0
        else:
            repeat = 0
        
        # 3. 绘制调试信息 - 所有元素添加到同一个package，最后统一发送
        package = Debug_Msgs()
        
        # 绘制目标点加速区域（绿色圆圈表示末端加速区域）
        debugger.draw_goal_area(package, goal[0], goal[1], radius=approach_distance, color=Debug_Msg.GREEN)
        
        # 绘制起点和终点标记（固定位置）
        debugger.draw_circle(package, goal_A[0], goal_A[1], radius=150)
        debugger.draw_circle(package, goal_B[0], goal_B[1], radius=150)
        
        # 绘制障碍物斥力影响范围（品红色圆圈）
        for i in range(0, 15):
            obs = vision.yellow_robot[i]
            if obs.visible:
                d_obs = np.linalg.norm(np.array([robot.x, robot.y]) - np.array([obs.x, obs.y]))
                if d_obs < 1500:  # 只绘制附近的障碍物影响范围
                    debugger.draw_obstacle_radius(package, obs.x, obs.y, radius=800, color=Debug_Msg.PURPLE)
        
        # 绘制PRM路径（如果处于逃脱模式）
        if flag == 0 and 'path_x' in locals() and len(path_x) > 0:
            debugger.draw_finalpath(package, path_x, path_y)
        
        # 绘制人工势场力向量（在机器人上方，避免被遮挡）
        debugger.draw_force_vectors(package, robot.x, robot.y, F_total, Fatt, Frep, scale=0.4)
        
        # 绘制机器人实际速度向量（白色）
        debugger.draw_velocity_vector(package, robot.x, robot.y, v_real, scale=3.0, color=Debug_Msg.WHITE)
        
        # 绘制机器人位置（黄色填充圆圈，最后绘制确保在最上层）
        debugger.draw_circle(package, vision.my_robot.x, vision.my_robot.y, radius=180)
        
        # 统一发送所有绘制信息
        debugger.send(package)
        
        # 初始化总开始时间
        if total_start_time is None and round_trip_count == 0 and flag == -1:
            total_start_time = time.time()
        
        time.sleep(0.005)
