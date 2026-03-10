from numpy import append
from vision import Vision
from action import Action
from debug import Debugger
import numpy as np
from zss_debug_pb2 import Debug_Msgs
from prm import PRM
import time
from func import Force,angle
import pid

goal = np.array([-2400,-1500])
flag=1
count = 0
w_real = -3.14
recode = np.array([0,0])
repeat=0


if __name__ == '__main__':
	vision = Vision()
	action = Action()
	debugger = Debugger()
	planner = PRM()
	w_pid = pid.PID()
	while True:
		# 1. path planning & velocity planning
		# Do something
		robot = vision.my_robot
		robot_last_x = robot.x
		robot_last_y = robot.y
		recode=np.array([robot_last_x,robot_last_y])
		time.sleep(0.01)
		if flag==1:
			goal = np.array([-2400,-1500])
		elif flag == -1:
			goal = np.array([2400,1500])
		F = Force([robot.x,robot.y],vision,goal)
		d_goal = np.linalg.norm(np.array([robot.x,robot.y]) - goal)
		if count == 0:
			F = np.array([0,0])
			count = 1
		else:
			count = 2
		totalv = F * 500
		v_real = [robot.x-robot_last_x,robot.y-robot_last_y]
		if v_real[0]!=0:
			w_real = angle(v_real)
		w_expect = angle(totalv)
		err = w_expect - w_real
		if err > np.pi:
			err = err - np.pi*2
		elif err <-np.pi:
			err = err + np.pi*2
		w_pid.SetPoint = 0
		pid.PID.update(w_pid,err)
		print(w_expect,w_real,err)
		if flag == 0:
			if d_goal < d_prm:
				flag = _flag
		elif d_goal<100:
			flag = -1*flag
			action.sendCommand(vx=0, vw=10)
			time.sleep(0.2)
		# 2. send command
		if count == 2:
			action.sendCommand(vx=np.linalg.norm(totalv)*2/(1+2*abs(w_pid.output)), vw=-w_pid.output*10)
		d_move=np.linalg.norm(np.array([vision.my_robot.x,vision.my_robot.y]) - recode)
		if d_move<10:
			repeat += 1
			if repeat > 200:
				path_x, path_y, road_map, sample_x, sample_y = planner.plan(vision=vision,
																			start_x=vision.my_robot.x,
																			start_y=vision.my_robot.y,
																			goal_x=goal[0],
																			goal_y=goal[1])
				l=len(path_x)
				goal=np.array([path_x[l-2],path_y[l-2]])
				action.sendCommand(vx=0, vw=-w_pid.output*10)
				#d_prm是状态结束时离目标点goal的距离
				d_prm=np.linalg.norm(recode - goal)/2
				if d_prm < 300:#目标点过近无法摆脱自锁状态
					d_prm = 999999
				elif d_prm < 1000:#目标点较近时，需要状态结束时离目标点更近一点
					d_prm = 500
				if flag != 0 :
					_flag=flag
				flag=0
				repeat=0
		else:
			repeat=0


		# 3. draw debug msg
		package = Debug_Msgs()
		debugger.draw_circle(package, vision.my_robot.x, vision.my_robot.y)
		if flag==0:
			debugger.draw_finalpath(package, path_x, path_y)
		debugger.draw_circle(package,goal[0],goal[1])
		debugger.send(package)

		time.sleep(0.005)
