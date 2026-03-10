from vision import Vision
from action import Action
from debug import Debugger
from prm import PRM
from rrt import RRT
from Track import Track
from move import Move,cal_dist
import time
import math
import numpy as np


if __name__ == '__main__':
    vision = Vision()
    action = Action()
    debugger = Debugger()
    planner = RRT()
    goal_x, goal_y = -2400, -1500
    loop_count = 5 # 往返次数
    path_x = []
    while True:
        # 1. path planning & velocity planning
        # path planning
        while(path_x == []):
            start_x, start_y = vision.my_robot.x, vision.my_robot.y # 记录起点
            while start_x < -10000:
                start_x, start_y = vision.my_robot.x, vision.my_robot.y # 记录起点
            print(start_x, start_y)
            
            # 创建可视化回调函数，展示规划过程
            visual_callback = debugger.create_visual_callback(start_x, start_y, goal_x, goal_y)
            
            path_x, path_y, road_map, sample_x, sample_y = planner.plan(
                vision=vision, 
                start_x=start_x, 
                start_y=start_y, 
                goal_x=goal_x, 
                goal_y=goal_y,
                visual_callback=visual_callback
            )

        # 呈现路径（已在规划过程中通过可视化回调显示）
        print('get path')

        # velocity planning
        point_num = len(path_x)-1
        for i in range(loop_count): # 循环5次，5个来回
            Next_point_index = point_num
            # 正向
            move = Move()
            while True:
                current_x, current_y = vision.my_robot.x, vision.my_robot.y
                dist_of_next_point = cal_dist(current_x, current_y, path_x[Next_point_index], path_y[Next_point_index])
                if dist_of_next_point < 200:
                    Next_point_index -= 1

                    if Next_point_index < 0:   # 到达最终目标点
                        action.sendCommand(vx=0, vw=0)
                        time.sleep(0.2)
                        Next_point_index = 0
                        break
                vx, vw = move.get_action(vision, path_x, path_y, Next_point_index, direction=1)
                # 发送指令
                action.sendCommand(vx=1.2*vx/(1+0.5*abs(vw)), vw=vw)
                time.sleep(0.05)  # 减小延时，提高控制频率

            move.turn_arround(action, vision)
            # 反向
            move = Move()
            while True:
                current_x, current_y = vision.my_robot.x, vision.my_robot.y
                dist_of_next_point = cal_dist(current_x, current_y, path_x[Next_point_index], path_y[Next_point_index])
                if dist_of_next_point < 200:
                    Next_point_index += 1

                    if Next_point_index == point_num+1:   # 到达最终目标点
                        action.sendCommand(vx=0, vw=0)
                        time.sleep(0.2)
                        Next_point_index = point_num
                        break
                vx, vw = move.get_action(vision, path_x, path_y, Next_point_index, direction=-1)
                # 发送指令
                action.sendCommand(vx=1.2*vx/(1+0.5*abs(vw)), vw=vw)
                time.sleep(0.1)
            move.turn_arround(action, vision)      


        break
    print('over')