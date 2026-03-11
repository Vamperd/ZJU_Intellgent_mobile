"""
A* 路径规划算法实现
将场地离散化为网格，使用 A* 算法搜索最优路径
"""

import numpy as np
import math
import heapq
from scipy.spatial import KDTree


class AStar:
    def __init__(self, grid_resolution=100, diagonal_move=True):
        """
        初始化 A* 规划器
        
        Args:
            grid_resolution: 网格分辨率（mm），越小路径越精确但计算量越大
            diagonal_move: 是否允许对角移动
        """
        self.grid_resolution = grid_resolution
        self.diagonal_move = diagonal_move
        
        # 场地范围（mm）
        self.minx = -4500
        self.maxx = 4500
        self.miny = -3000
        self.maxy = 3000
        
        # 机器人参数
        self.robot_size = 200
        self.avoid_dist = 200
        
        # 网格尺寸
        self.grid_width = int((self.maxx - self.minx) / grid_resolution) + 1
        self.grid_height = int((self.maxy - self.miny) / grid_resolution) + 1
        
        # 可视化回调
        self.visual_callback = None
        
    def _default_visual_callback(self, event, data):
        """默认可视化回调（什么都不做）"""
        pass
    
    def plan(self, vision, start_x, start_y, goal_x, goal_y, visual_callback=None):
        """
        规划从起点到终点的路径
        
        Args:
            vision: 视觉数据对象
            start_x, start_y: 起点坐标
            goal_x, goal_y: 终点坐标
            visual_callback: 可视化回调函数
            
        Returns:
            path_x, path_y: 路径点坐标列表
            grid_map: 网格地图（用于可视化）
            came_from: 搜索树（用于可视化）
        """
        self.visual_callback = visual_callback if visual_callback else self._default_visual_callback
        
        print('A* Planning started...')
        
        # 1. 提取障碍物
        obstacle_x, obstacle_y = self._extract_obstacles(vision)
        obstree = KDTree(np.vstack((obstacle_x, obstacle_y)).T)
        
        # 2. 创建网格地图
        grid_map = self._create_grid_map(obstacle_x, obstacle_y, obstree)
        
        # 可视化：显示网格地图
        self.visual_callback('grid_map', {
            'grid_map': grid_map,
            'resolution': self.grid_resolution,
            'minx': self.minx,
            'miny': self.miny,
            'start': (start_x, start_y),
            'goal': (goal_x, goal_y)
        })
        
        # 3. 转换坐标到网格索引
        start_grid = self._world_to_grid(start_x, start_y)
        goal_grid = self._world_to_grid(goal_x, goal_y)
        
        print(f'Start grid: {start_grid}, Goal grid: {goal_grid}')
        
        # 4. A* 搜索
        path_grid, came_from, open_set_history = self._astar_search(
            grid_map, start_grid, goal_grid
        )
        
        if not path_grid:
            print('A* failed to find a path!')
            return [], [], grid_map, came_from
        
        # 5. 转换路径到世界坐标
        path_x, path_y = self._grid_path_to_world(path_grid)
        
        # 可视化：显示找到的原始路径
        self.visual_callback('path_found', {
            'path_x': path_x,
            'path_y': path_y,
            'came_from': came_from,
            'grid_map': grid_map
        })
        
        # 6. 路径平滑处理
        print('Smoothing path...')
        path_x, path_y = self._smooth_path(path_x, path_y, obstree)
        
        # 可视化：显示平滑后的路径
        self.visual_callback('path_smoothed', {
            'path_x': path_x,
            'path_y': path_y,
            'original_x': self._grid_path_to_world(path_grid)[0],
            'original_y': self._grid_path_to_world(path_grid)[1]
        })
        
        # 7. 路径插值（增加路径点密度）
        path_x, path_y = self._interpolate_path(path_x, path_y, obstree, interval=400)
        
        # 计算路径总长度
        if len(path_x) > 1:
            path_length = sum(math.hypot(path_x[i+1]-path_x[i], path_y[i+1]-path_y[i]) 
                             for i in range(len(path_x)-1))
            print(f'A* Path found! Nodes: {len(path_x)}, Length: {path_length:.0f}mm')
        
        # 最终可视化
        self.visual_callback('final_result', {
            'path_x': path_x,
            'path_y': path_y
        })
        
        return path_x, path_y, grid_map, came_from
    
    def _extract_obstacles(self, vision):
        """从视觉数据中提取障碍物位置"""
        obstacle_x = [-9999]
        obstacle_y = [-9999]
        
        # 蓝方机器人（排除己方0号车）
        for robot_blue in vision.blue_robot:
            if robot_blue.visible and robot_blue.id > 0:
                obstacle_x.append(robot_blue.x)
                obstacle_y.append(robot_blue.y)
        
        # 黄方机器人
        for robot_yellow in vision.yellow_robot:
            if robot_yellow.visible:
                obstacle_x.append(robot_yellow.x)
                obstacle_y.append(robot_yellow.y)
        
        return obstacle_x, obstacle_y
    
    def _create_grid_map(self, obstacle_x, obstacle_y, obstree):
        """
        创建网格地图
        0 = 可通行, 1 = 障碍物
        """
        grid_map = np.zeros((self.grid_height, self.grid_width), dtype=np.int8)
        
        # 障碍物膨胀半径
        inflate_radius = self.robot_size + self.avoid_dist
        
        # 遍历所有网格点
        for i in range(self.grid_height):
            for j in range(self.grid_width):
                # 网格中心的世界坐标
                wx, wy = self._grid_to_world(j, i)
                
                # 检查是否与障碍物碰撞
                dist, _ = obstree.query([wx, wy])
                if dist <= inflate_radius:
                    grid_map[i, j] = 1
        
        return grid_map
    
    def _world_to_grid(self, x, y):
        """世界坐标转网格索引"""
        gx = int((x - self.minx) / self.grid_resolution)
        gy = int((y - self.miny) / self.grid_resolution)
        
        # 边界限制
        gx = max(0, min(gx, self.grid_width - 1))
        gy = max(0, min(gy, self.grid_height - 1))
        
        return (gx, gy)
    
    def _grid_to_world(self, gx, gy):
        """网格索引转世界坐标（返回网格中心）"""
        x = self.minx + gx * self.grid_resolution + self.grid_resolution / 2
        y = self.miny + gy * self.grid_resolution + self.grid_resolution / 2
        return (x, y)
    
    def _grid_path_to_world(self, path_grid):
        """网格路径转世界坐标路径"""
        path_x = []
        path_y = []
        for (gx, gy) in path_grid:
            x, y = self._grid_to_world(gx, gy)
            path_x.append(x)
            path_y.append(y)
        return path_x, path_y
    
    def _heuristic(self, a, b):
        """
        启发式函数：欧几里得距离
        """
        return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    
    def _get_neighbors(self, current, grid_map):
        """获取当前节点的邻居节点"""
        neighbors = []
        x, y = current
        
        # 8方向移动（包括对角线）
        if self.diagonal_move:
            directions = [
                (1, 0), (-1, 0), (0, 1), (0, -1),  # 水平垂直
                (1, 1), (1, -1), (-1, 1), (-1, -1)  # 对角线
            ]
        else:
            directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # 边界检查
            if 0 <= nx < self.grid_width and 0 <= ny < self.grid_height:
                # 障碍物检查
                if grid_map[ny, nx] == 0:
                    neighbors.append((nx, ny))
        
        return neighbors
    
    def _get_move_cost(self, current, neighbor):
        """计算移动代价"""
        dx = abs(neighbor[0] - current[0])
        dy = abs(neighbor[1] - current[1])
        
        # 对角线移动代价为 sqrt(2)，水平/垂直为 1
        if dx + dy == 2:
            return math.sqrt(2)
        else:
            return 1.0
    
    def _astar_search(self, grid_map, start, goal):
        """
        A* 搜索算法核心
        
        Returns:
            path: 路径网格索引列表
            came_from: 搜索树（用于可视化）
            open_set_history: 开放集历史（用于可视化）
        """
        # 开放集（优先队列）
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        # 记录每个节点的来源
        came_from = {}
        
        # g_score: 从起点到当前节点的实际代价
        g_score = {start: 0}
        
        # f_score: g_score + 启发式估计
        f_score = {start: self._heuristic(start, goal)}
        
        # 开放集历史（用于可视化）
        open_set_history = [set([start])]
        
        # 已访问节点集合
        visited = set()
        
        # 可视化更新频率
        visual_update_interval = 50
        iteration = 0
        
        while open_set:
            iteration += 1
            
            # 取出 f 值最小的节点
            _, current = heapq.heappop(open_set)
            
            # 检查是否到达目标
            if current == goal:
                # 重建路径
                path = self._reconstruct_path(came_from, current)
                print(f'A* found path at iteration {iteration}, path length: {len(path)}')
                
                # 可视化：找到路径
                self.visual_callback('search_complete', {
                    'came_from': came_from,
                    'path': path,
                    'visited': visited,
                    'grid_map': grid_map
                })
                
                return path, came_from, open_set_history
            
            # 如果已经访问过则跳过
            if current in visited:
                continue
            visited.add(current)
            
            # 遍历邻居
            for neighbor in self._get_neighbors(current, grid_map):
                if neighbor in visited:
                    continue
                
                # 计算通过当前节点到达邻居的代价
                tentative_g_score = g_score[current] + self._get_move_cost(current, neighbor)
                
                # 如果找到更优路径
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self._heuristic(neighbor, goal)
                    
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
            
            # 定期可视化更新
            if iteration % visual_update_interval == 0:
                current_open = set(node for _, node in open_set)
                open_set_history.append(current_open)
                
                self.visual_callback('searching', {
                    'visited': visited.copy(),
                    'open_set': current_open,
                    'current': current,
                    'grid_map': grid_map,
                    'iteration': iteration,
                    'start': start,
                    'goal': goal
                })
        
        # 未找到路径
        print(f'A* failed to find path after {iteration} iterations')
        return [], came_from, open_set_history
    
    def _reconstruct_path(self, came_from, current):
        """从搜索树重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
    
    def _smooth_path(self, path_x, path_y, obstree):
        """
        路径平滑：跳过可直接连通的中间节点
        """
        if len(path_x) <= 2:
            return path_x, path_y
        
        smoothed_x, smoothed_y = [path_x[0]], [path_y[0]]
        i = 0
        prune_count = 0
        
        while i < len(path_x) - 1:
            # 从远到近尝试跳过中间节点
            for j in range(len(path_x) - 1, i, -1):
                if self._check_line_collision(path_x[i], path_y[i], 
                                              path_x[j], path_y[j], obstree):
                    continue
                
                # 找到可以跳过的节点
                prune_count += 1
                smoothed_x.append(path_x[j])
                smoothed_y.append(path_y[j])
                i = j
                break
            else:
                i += 1
                if i < len(path_x):
                    smoothed_x.append(path_x[i])
                    smoothed_y.append(path_y[i])
        
        print(f'Path smoothed: {len(path_x)} -> {len(smoothed_x)} nodes')
        return smoothed_x, smoothed_y
    
    def _check_line_collision(self, x1, y1, x2, y2, obstree):
        """
        检查两点之间的连线是否与障碍物碰撞
        返回 True 表示有碰撞
        """
        dist = math.hypot(x2 - x1, y2 - y1)
        angle = math.atan2(y2 - y1, x2 - x1)
        
        step_size = self.robot_size + self.avoid_dist
        steps = int(dist / step_size) + 1
        
        x, y = x1, y1
        for _ in range(steps):
            distance, _ = obstree.query([x, y])
            if distance <= self.robot_size + self.avoid_dist:
                return True
            x += step_size * math.cos(angle)
            y += step_size * math.sin(angle)
        
        # 检查终点
        distance, _ = obstree.query([x2, y2])
        if distance <= self.robot_size + self.avoid_dist:
            return True
        
        return False
    
    def _interpolate_path(self, path_x, path_y, obstree, interval=400):
        """路径插值：在路径节点间插入中间点"""
        if len(path_x) <= 1:
            return path_x, path_y
        
        new_x, new_y = [path_x[0]], [path_y[0]]
        
        for i in range(len(path_x) - 1):
            x1, y1 = path_x[i], path_y[i]
            x2, y2 = path_x[i + 1], path_y[i + 1]
            
            dist = math.hypot(x2 - x1, y2 - y1)
            if dist > interval:
                num_points = int(dist / interval)
                for j in range(1, num_points + 1):
                    ratio = j / (num_points + 1)
                    interp_x = x1 + ratio * (x2 - x1)
                    interp_y = y1 + ratio * (y2 - y1)
                    
                    # 检查插值点安全性
                    col_dist, _ = obstree.query([interp_x, interp_y])
                    if col_dist >= self.robot_size + self.avoid_dist:
                        new_x.append(interp_x)
                        new_y.append(interp_y)
            
            new_x.append(x2)
            new_y.append(y2)
        
        return new_x, new_y
