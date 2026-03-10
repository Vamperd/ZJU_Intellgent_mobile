from scipy.spatial import KDTree
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import time

class Node(object):
    def __init__(self, x, y, cost, parent):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent

class RRT(object):
    def __init__(self, N_SAMPLE=100, KNN=10, MAX_EDGE_LEN=5000):
        self.N_SAMPLE = N_SAMPLE
        self.KNN = KNN
        self.MAX_EDGE_LEN = MAX_EDGE_LEN
        self.minx = -4500
        self.maxx = 4500
        self.miny = -3000
        self.maxy = 3000
        self.robot_size = 200
        self.avoid_dist = 200


    def plan(self, vision, start_x, start_y, goal_x, goal_y):
        # Obstacles
        obstacle_x = [-9999]
        obstacle_y = [-9999]
        for robot_blue in vision.blue_robot:
            if robot_blue.visible and robot_blue.id > 0:
                obstacle_x.append(robot_blue.x)
                obstacle_y.append(robot_blue.y)
        for robot_yellow in vision.yellow_robot:
            if robot_yellow.visible:
                obstacle_x.append(robot_yellow.x)
                obstacle_y.append(robot_yellow.y)
        # Obstacle KD Tree
        obstree = KDTree(np.vstack((obstacle_x, obstacle_y)).T)
        # Sampling
        print('start searching')
        road_x, road_y, parent_index, sample_x, sample_y = self.sampling(start_x, start_y, goal_x, goal_y, obstree)
        print("finished")
        # Generate Roadmap
        road_map = self.generate_roadmap(sample_x, sample_y, obstree)
        # Get path
        path_x, path_y = self.get_path(start_x, start_y, parent_index, sample_x, sample_y)

        return path_x, path_y, road_map, sample_x, sample_y

    def sampling(self,start_x, start_y, goal_x, goal_y, obstree):
        sample_x, sample_y = [], []
        road_x, road_y = [], []
        parent_index = [-1]
        sample_x.append(start_x)
        sample_y.append(start_y)
        if_continue = 1
        while (if_continue == 1):
            Nodetree = KDTree(np.vstack((sample_x, sample_y)).T)
            random_num = random.random()
            if random_num < 0.1:
                Q_rand_x = goal_x
                Q_rand_y = goal_y
            else:
                Q_rand_x = (random.random() * (self.maxx - self.minx)) + self.minx
                Q_rand_y = (random.random() * (self.maxy - self.miny)) + self.miny
            distance, index = Nodetree.query(np.array([Q_rand_x, Q_rand_y]))    # 查找与随机采样点最近的点
            Q_nearest_x,Q_nearest_y  = sample_x[index],sample_y[index]
            # 拓展节点，Qnew
            Q_new_x, Q_new_y = self.extend(Q_nearest_x,Q_nearest_y,Q_rand_x,Q_rand_y)

            # 碰撞检测
            col_distance,_ = obstree.query(np.array([Q_new_x, Q_new_y]))

            if not self.check_obs(Q_nearest_x, Q_nearest_y, Q_new_x, Q_new_y, obstree):     # 无碰撞情况下
                if col_distance >= self.robot_size + self.avoid_dist:
                    road_x.append(Q_new_x)
                    road_y.append(Q_new_y)
                    sample_x.append(Q_new_x)
                    sample_y.append(Q_new_y)
                    parent_index.append(index)
                    goal_x_dist = Q_new_x - goal_x
                    goal_y_dist = Q_new_y - goal_y

                    goal_distance = math.hypot(goal_x_dist, goal_y_dist)  # 与目标点的距离
                    # print(goal_distance)

                    if not self.check_obs(Q_new_x, Q_new_y, goal_x, goal_y, obstree) or goal_distance < 400: #检测是否达到终点
                        # print('4')
                        road_x.append(goal_x)
                        road_y.append(goal_y)
                        sample_x.append(goal_x)
                        sample_y.append(goal_y)
                        parent_index.append(len(sample_x)-2)
                        if_continue = 0
                        
        return road_x, road_y, parent_index, sample_x, sample_y

    def extend(self,Q_nearest_x,Q_nearest_y,Q_rand_x,Q_rand_y):
        dy = Q_rand_y - Q_nearest_y
        dx = Q_rand_x - Q_nearest_x
        theta = math.atan2(dy, dx)
        Q_new_x = Q_nearest_x + 1000 * math.cos(theta)
        Q_new_y = Q_nearest_y + 1000 * math.sin(theta)
        return Q_new_x, Q_new_y

    def generate_roadmap(self, sample_x, sample_y, obstree):
        road_map = []
        nsample = len(sample_x)
        sampletree = KDTree(np.vstack((sample_x, sample_y)).T)

        for (i, ix, iy) in zip(range(nsample), sample_x, sample_y):
            distance, index = sampletree.query(np.array([ix, iy]), k=nsample)
            edges = []
            # print(len(index))

            for ii in range(1, len(index)):
                nx = sample_x[index[ii]]
                ny = sample_y[index[ii]]

                # check collision
                if not self.check_obs(ix, iy, nx, ny, obstree):
                    edges.append(index[ii])

                if len(edges) >= self.KNN:
                    break

            road_map.append(edges)

        return road_map

    def get_path(self, start_x, start_y, parent_index, sample_x, sample_y):
        path_x, path_y = [], []
        last_index = len(parent_index) - 1
        while parent_index[last_index]!=-1:
            path_x.append(sample_x[last_index])
            path_y.append(sample_y[last_index])
            last_index = parent_index[last_index]
        path_x.append(start_x)
        path_y.append(start_y)
        return path_x, path_y

    def check_obs(self, ix, iy, nx, ny, obstree):
        x = ix
        y = iy
        dx = nx - ix
        dy = ny - iy
        angle = math.atan2(dy, dx)
        dis = math.hypot(dx, dy)

        if dis > self.MAX_EDGE_LEN:
            return True

        step_size = self.robot_size + self.avoid_dist
        steps = round(dis/step_size)
        for i in range(steps):
            distance, index = obstree.query(np.array([x, y]))
            if distance <= self.robot_size + self.avoid_dist:
                return True
            x += step_size * math.cos(angle)
            y += step_size * math.sin(angle)

        # check for goal point
        distance, index = obstree.query(np.array([nx, ny]))
        if distance <= 100:
            return True

        return False
    def maxin(self,parent_index):
        max = 0
        for i in range(len(parent_index)):
            if parent_index[i]>max:
                max = parent_index
        return max

