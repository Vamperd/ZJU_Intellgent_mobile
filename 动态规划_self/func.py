from numpy import append
from vision import Vision
from action import Action
from debug import Debugger
import numpy as np
from zss_debug_pb2 import Debug_Msgs
import time
import pid

d_min = 160
k_a = 0.00055
k_b = 7e7
d_o = 800
step = 0.01
d_a = 1000
Fi = 1
totalv = 0
count = 0
w_a = 30
w = 0
theta_real = 0

def Force(robot_position, vision, goal):
    """
    计算人工势场力
    返回: (合力, 引力, 斥力, 障碍物位置列表, 各障碍物斥力列表)
    """
    robot_position = np.array(robot_position)
    Position_Yellow = np.array([[vision.yellow_robot[i].x, vision.yellow_robot[i].y] for i in range(0, 16)])
    Position = Position_Yellow
    
    d_goal = np.linalg.norm(robot_position - goal)
    d_frep = np.array([np.linalg.norm(robot_position - Position[i]) for i in range(0, 15)])
    
    Frep_sum = np.array([0.0, 0.0])
    Frep_list = []  # 存储每个障碍物的斥力
    obs_positions = []  # 存储产生斥力的障碍物位置
    
    # 计算引力
    if d_goal <= d_a:
        Fatt = -2 * k_a * (robot_position - goal)
    else:
        Fatt = -2 * k_a * d_a * (robot_position - goal) / d_goal
    
    # 计算斥力
    for i in range(0, 15):
        if d_frep[i] <= d_o:
            Frep1 = k_b * (1.0 / d_frep[i] - 1.0 / d_o) * 1.0 / d_frep[i] / d_frep[i] * (robot_position - Position[i]) / d_frep[i]
            Frep_sum = Frep_sum + Frep1
            Frep_list.append(Frep1)
            obs_positions.append(Position[i])
        else:
            Frep_list.append(np.array([0.0, 0.0]))
    
    F_total = Frep_sum + Fatt
    return F_total, Fatt, Frep_sum, obs_positions, Frep_list


def angle(totalv):
	if totalv[0]<0:
		if np.arctan(totalv[1]/totalv[0])<0:
			theta = np.arctan(totalv[1]/totalv[0])+np.pi
		else:
			theta = np.arctan(totalv[1]/totalv[0])-np.pi
	elif totalv[0]>0:
		theta = np.arctan(totalv[1]/totalv[0])
	else:
		if totalv[1]>0:
			theta = np.pi/2
		else:
			theta = -np.pi/2
	return theta

