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
    def __init__(self, N_SAMPLE=100, KNN=10, MAX_EDGE_LEN=5000, SEARCH_RADIUS=1500, MAX_ITER=3000):
        self.N_SAMPLE = N_SAMPLE
        self.KNN = KNN
        self.MAX_EDGE_LEN = MAX_EDGE_LEN
        self.SEARCH_RADIUS = SEARCH_RADIUS  # RRT* 邻居搜索半径
        self.MAX_ITER = MAX_ITER  # 最大迭代次数
        self.minx = -4500
        self.maxx = 4500
        self.miny = -3000
        self.maxy = 3000
        self.robot_size = 200
        self.avoid_dist = 200

    # 可视化回调函数的默认实现（什么都不做）
    def _default_visual_callback(self, event, data):
        pass

    def plan(self, vision, start_x, start_y, goal_x, goal_y, visual_callback=None):
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
        
        # 设置可视化回调
        self.visual_callback = visual_callback if visual_callback else self._default_visual_callback
        
        # Sampling
        print('start searching')
        road_x, road_y, parent_index, sample_x, sample_y, goal_node_idx = self.sampling(
            start_x, start_y, goal_x, goal_y, obstree
        )
        print("finished")
        # Generate Roadmap
        road_map = self.generate_roadmap(sample_x, sample_y, obstree)
        # Get path
        path_x, path_y = self.get_path(start_x, start_y, parent_index, sample_x, sample_y, goal_node_idx)
        
        # 路径后处理：剪枝 + 插值平滑
        print("Smoothing path...")
        path_x, path_y = self.smooth_path(path_x, path_y, obstree, visual_callback=self.visual_callback)
        path_x, path_y = self.interpolate_path(path_x, path_y, obstree, interval=400)
        
        # 计算路径总长度
        if len(path_x) > 1:
            path_length = sum(math.hypot(path_x[i+1]-path_x[i], path_y[i+1]-path_y[i]) 
                             for i in range(len(path_x)-1))
            print(f"Path nodes: {len(path_x)}, Path length: {path_length:.0f}")
        else:
            print("Warning: No valid path found")

        # 最终可视化：显示路径周围的采样点和树结构
        if len(path_x) > 1:
            print("Showing final visualization...")
            self.visual_callback('final_result', {
                'path_x': path_x,
                'path_y': path_y
            })
            time.sleep(0.5)  # 暂停让用户看清最终结果

        return path_x, path_y, road_map, sample_x, sample_y

    def sampling(self, start_x, start_y, goal_x, goal_y, obstree):
        sample_x, sample_y = [], []
        parent_index = [-1]
        cost = [0.0]  # 起点代价为0
        sample_x.append(start_x)
        sample_y.append(start_y)
        
        goal_node_idx = -1  # 目标点在树中的索引
        best_goal_cost = float('inf')
        iteration = 0
        
        # 可视化更新频率
        visual_update_interval = 10  # 每10次迭代更新一次可视化
        
        # 找到目标后继续优化的迭代次数
        post_goal_iterations = 300
        
        while iteration < self.MAX_ITER:
            iteration += 1
            Nodetree = KDTree(np.vstack((sample_x, sample_y)).T)
            
            # 目标偏置采样
            random_num = random.random()
            if random_num < 0.15:
                Q_rand_x = goal_x
                Q_rand_y = goal_y
            else:
                Q_rand_x = (random.random() * (self.maxx - self.minx)) + self.minx
                Q_rand_y = (random.random() * (self.maxy - self.miny)) + self.miny
            
            # 找最近节点
            distance, nearest_idx = Nodetree.query(np.array([Q_rand_x, Q_rand_y]))
            nearest_idx = int(nearest_idx)
            Q_nearest_x, Q_nearest_y = sample_x[nearest_idx], sample_y[nearest_idx]
            
            # 扩展新节点
            Q_new_x, Q_new_y = self.extend(Q_nearest_x, Q_nearest_y, Q_rand_x, Q_rand_y)
            
            # 碰撞检测
            col_distance, _ = obstree.query(np.array([Q_new_x, Q_new_y]))
            
            if not self.check_obs(Q_nearest_x, Q_nearest_y, Q_new_x, Q_new_y, obstree):
                if col_distance >= self.robot_size + self.avoid_dist:
                    # RRT*: 搜索邻居节点
                    neighbor_indices = self.find_near_nodes(Nodetree, Q_new_x, Q_new_y, len(sample_x))
                    
                    if nearest_idx not in neighbor_indices:
                        neighbor_indices.append(nearest_idx)
                    
                    # RRT*: 选择最优父节点
                    best_parent, best_cost = self.choose_parent(
                        Q_new_x, Q_new_y, neighbor_indices, 
                        sample_x, sample_y, cost, obstree
                    )
                    
                    if best_parent is None:
                        best_parent = nearest_idx
                        best_cost = cost[nearest_idx] + math.hypot(Q_new_x - Q_nearest_x, Q_new_y - Q_nearest_y)
                    
                    # 添加新节点
                    sample_x.append(Q_new_x)
                    sample_y.append(Q_new_y)
                    parent_index.append(best_parent)
                    cost.append(best_cost)
                    
                    new_node_idx = len(sample_x) - 1
                    
                    # RRT*: 重布线
                    self.rewire(
                        Q_new_x, Q_new_y, new_node_idx, neighbor_indices,
                        sample_x, sample_y, cost, parent_index, obstree
                    )
                    
                    # 尝试从新节点连接到目标
                    goal_dist = math.hypot(Q_new_x - goal_x, Q_new_y - goal_y)
                    if not self.check_obs(Q_new_x, Q_new_y, goal_x, goal_y, obstree):
                        new_goal_cost = cost[new_node_idx] + goal_dist
                        
                        if goal_node_idx == -1:
                            # 第一次到达目标，添加目标点
                            goal_node_idx = len(sample_x)
                            sample_x.append(goal_x)
                            sample_y.append(goal_y)
                            parent_index.append(new_node_idx)
                            cost.append(new_goal_cost)
                            best_goal_cost = new_goal_cost
                            print(f"Goal reached at iteration {iteration}, cost: {best_goal_cost:.0f}")
                            
                            # 可视化：目标到达
                            self.visual_callback('goal_reached', {
                                'tree_x': sample_x.copy(),
                                'tree_y': sample_y.copy(),
                                'parent_index': parent_index.copy(),
                                'goal_x': goal_x,
                                'goal_y': goal_y,
                                'iteration': iteration
                            })
                        elif new_goal_cost < best_goal_cost:
                            # 找到更好的路径
                            best_goal_cost = new_goal_cost
                            parent_index[goal_node_idx] = new_node_idx
                            cost[goal_node_idx] = new_goal_cost
                    
                    # 定期更新可视化（树扩展过程）
                    if iteration % visual_update_interval == 0:
                        self.visual_callback('tree_expand', {
                            'tree_x': sample_x.copy(),
                            'tree_y': sample_y.copy(),
                            'parent_index': parent_index.copy(),
                            'new_x': Q_new_x,
                            'new_y': Q_new_y,
                            'parent_x': sample_x[best_parent],
                            'parent_y': sample_y[best_parent],
                            'iteration': iteration,
                            'goal_x': goal_x,
                            'goal_y': goal_y
                        })
                    
                    # 到达目标后继续优化
                    if goal_node_idx != -1:
                        post_goal_iterations -= 1
                        if post_goal_iterations <= 0:
                            print(f"Optimization finished. Final cost: {cost[goal_node_idx]:.0f}")
                            break
        
        if goal_node_idx == -1:
            print(f"Warning: Goal not reached after {iteration} iterations")
        
        road_x = sample_x.copy()
        road_y = sample_y.copy()
        
        return road_x, road_y, parent_index, sample_x, sample_y, goal_node_idx

    def find_near_nodes(self, tree, x, y, n):
        """查找新节点周围的邻居节点"""
        # RRT* 标准动态半径公式: gamma * (log(n)/n)^(1/d), d=2为维度
        # gamma应大于场地对角线/根号pi
        gamma = 5000
        radius = gamma * math.sqrt(math.log(n + 1) / (n + 1))
        radius = min(radius, self.SEARCH_RADIUS)  # 不超过最大搜索半径
        if radius < 300:
            radius = 300
        
        indices = tree.query_ball_point([x, y], radius)
        return list(indices)

    def choose_parent(self, new_x, new_y, neighbor_indices, sample_x, sample_y, cost, obstree):
        """在邻居中选择代价最小的作为父节点"""
        best_parent = None
        best_cost = float('inf')
        
        for idx in neighbor_indices:
            px, py = sample_x[idx], sample_y[idx]
            
            # 检查连线是否无碰撞
            if not self.check_obs(px, py, new_x, new_y, obstree):
                edge_cost = math.hypot(new_x - px, new_y - py)
                total_cost = cost[idx] + edge_cost
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_parent = idx
        
        return best_parent, best_cost

    def rewire(self, new_x, new_y, new_idx, neighbor_indices, sample_x, sample_y, cost, parent_index, obstree):
        """重布线：检查邻居是否通过新节点可以获得更小代价"""
        for idx in neighbor_indices:
            if idx == new_idx:
                continue
            
            px, py = sample_x[idx], sample_y[idx]
            edge_cost = math.hypot(new_x - px, new_y - py)
            potential_cost = cost[new_idx] + edge_cost
            
            # 如果通过新节点代价更小，且连线无碰撞
            if potential_cost < cost[idx]:
                if not self.check_obs(new_x, new_y, px, py, obstree):
                    cost[idx] = potential_cost
                    parent_index[idx] = new_idx

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
            
            # 确保index是数组格式
            if nsample == 1:
                index = np.array([index])
            elif not hasattr(index, '__len__'):
                index = np.array([index])
            
            for ii in range(1, len(index)):
                nx = sample_x[index[ii]]
                ny = sample_y[index[ii]]

                # check collision
                if not self.check_obs(ix, iy, nx, ny, obstree):
                    edges.append(int(index[ii]))

                if len(edges) >= self.KNN:
                    break

            road_map.append(edges)

        return road_map

    def get_path(self, start_x, start_y, parent_index, sample_x, sample_y, goal_node_idx):
        path_x, path_y = [], []
        
        if goal_node_idx == -1:
            print("Error: No goal node found")
            return path_x, path_y
        
        last_index = goal_node_idx
        while parent_index[last_index] != -1:
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
    
    def smooth_path(self, path_x, path_y, obstree, visual_callback=None):
        """路径剪枝：跳过可直接连通的中间节点，减少转角"""
        if len(path_x) <= 2:
            return path_x, path_y
        
        if visual_callback is None:
            visual_callback = self._default_visual_callback
        
        smoothed_x, smoothed_y = [path_x[0]], [path_y[0]]
        i = 0
        prune_count = 0
        
        # 可视化：初始路径
        visual_callback('prune_start', {
            'original_x': path_x.copy(),
            'original_y': path_y.copy()
        })
        
        while i < len(path_x) - 1:
            # 从远到近尝试跳过中间节点
            for j in range(len(path_x) - 1, i, -1):
                if not self.check_obs(path_x[i], path_y[i], path_x[j], path_y[j], obstree):
                    # 找到了可以跳过的节点，可视化剪枝
                    prune_count += 1
                    visual_callback('prune_step', {
                        'smoothed_x': smoothed_x.copy() + [path_x[j]],
                        'smoothed_y': smoothed_y.copy() + [path_y[j]],
                        'pruned_segment_x': path_x[i:j+1],
                        'pruned_segment_y': path_y[i:j+1],
                        'keep_start': (path_x[i], path_y[i]),
                        'keep_end': (path_x[j], path_y[j])
                    })
                    
                    smoothed_x.append(path_x[j])
                    smoothed_y.append(path_y[j])
                    i = j
                    break
            else:
                i += 1
                if i < len(path_x):
                    smoothed_x.append(path_x[i])
                    smoothed_y.append(path_y[i])
        
        # 可视化：剪枝完成
        visual_callback('prune_done', {
            'smoothed_x': smoothed_x,
            'smoothed_y': smoothed_y,
            'prune_count': prune_count
        })
        
        return smoothed_x, smoothed_y

    def interpolate_path(self, path_x, path_y, obstree, interval=300):
        """路径插值平滑：在路径节点间插入中间点，减少转角"""
        if len(path_x) <= 1:
            return path_x, path_y
        
        new_x, new_y = [path_x[0]], [path_y[0]]
        
        for i in range(len(path_x) - 1):
            x1, y1 = path_x[i], path_y[i]
            x2, y2 = path_x[i + 1], path_y[i + 1]
            
            dist = math.hypot(x2 - x1, y2 - y1)
            if dist > interval:
                # 插入中间点
                num_points = int(dist / interval)
                for j in range(1, num_points + 1):
                    ratio = j / (num_points + 1)
                    interp_x = x1 + ratio * (x2 - x1)
                    interp_y = y1 + ratio * (y2 - y1)
                    # 检查插值点是否安全
                    col_dist, _ = obstree.query([interp_x, interp_y])
                    if col_dist >= self.robot_size + self.avoid_dist:
                        new_x.append(interp_x)
                        new_y.append(interp_y)
            
            new_x.append(x2)
            new_y.append(y2)
        
        return new_x, new_y

