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
def Force(robot_position,vision,goal):
		robot_position = np.array(robot_position)
		#Position_Blue = np.array([[vision.blue_robot[i].x,vision.blue_robot[i].y] for i in range(1,8)])
		Position_Yellow = np.array([[vision.yellow_robot[i].x,vision.yellow_robot[i].y] for i in range (0,16)])
		#Position = np.append(Position_Blue,Position_Yellow,axis=0)
		Position = Position_Yellow	#dynamic
		d_goal = np.linalg.norm(robot_position - goal)		#distance from my robot to goal
		d_frep = np.array([np.linalg.norm(robot_position - Position[i]) for i in range(0,15)])	#distance from my robot to all the other robot
		Frep_sum = np.array([0,0])							#compute the sum frep of all robot
		if d_goal<=d_a:
			Fatt = -2*k_a*(robot_position - goal)
		else:
			Fatt = -2*k_a*d_a*(robot_position-goal)/d_goal
		for i in range(0,15):
			if d_frep[i]<=d_o:
				Frep1 = k_b*(1.0/d_frep[i]-1.0/d_o)*1.0/d_frep[i]/d_frep[i]*(robot_position - Position[i])/d_frep[i]
			else:
				Frep1 = 0
				
			Frep_sum = Frep_sum + Frep1
		#print(Frep_sum,Fatt)
		return Frep_sum + Fatt


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

