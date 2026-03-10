import socket
import sys
import time

from zss_debug_pb2 import Debug_Msgs, Debug_Msg, Debug_Arc

class Debugger(object):
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.debug_address = ('localhost', 20001)

    def draw_circle(self, package, x, y, radius=300):
        msg = package.msgs.add()
        msg.type = Debug_Msg.ARC
        msg.color = Debug_Msg.WHITE
        arc = msg.arc
        arc.rectangle.point1.x = x - radius
        arc.rectangle.point1.y = y - radius
        arc.rectangle.point2.x = x + radius
        arc.rectangle.point2.y = y + radius
        arc.start = 0
        arc.end = 360
        arc.FILL = True

    def draw_line(self, package, x1, y1, x2, y2, color=Debug_Msg.WHITE):
        msg = package.msgs.add()
        msg.type = Debug_Msg.LINE
        msg.color = color
        line = msg.line
        line.start.x = x1
        line.start.y = y1
        line.end.x = x2
        line.end.y = y2
        line.FORWARD = True
        line.BACK = True
    
    def draw_lines(self, package, x1, y1, x2, y2, color=Debug_Msg.WHITE):
        for i in range(len(x1)):
            msg = package.msgs.add()
            msg.type = Debug_Msg.LINE
            msg.color = color
            line = msg.line
            line.start.x = x1[i]
            line.start.y = y1[i]
            line.end.x = x2[i]
            line.end.y = y2[i]
            line.FORWARD = True
            line.BACK = True

    def draw_point(self, package, x, y, color=Debug_Msg.WHITE):
        msg = package.msgs.add()
        # line 1
        msg.type = Debug_Msg.LINE
        msg.color = color
        line = msg.line
        line.start.x = x + 50
        line.start.y = y + 50
        line.end.x = x - 50
        line.end.y = y - 50
        line.FORWARD = True
        line.BACK = True
        # line 2
        msg = package.msgs.add()
        msg.type = Debug_Msg.LINE
        msg.color = color
        line = msg.line
        line.start.x = x - 50
        line.start.y = y + 50
        line.end.x = x + 50
        line.end.y = y - 50
        line.FORWARD = True
        line.BACK = True

    def draw_points(self, package, x, y, color=Debug_Msg.WHITE):
        for i in range(len(x)):
            # line 1
            msg = package.msgs.add()
            msg.type = Debug_Msg.LINE
            msg.color = color
            line = msg.line
            line.start.x = x[i] + 50
            line.start.y = y[i] + 50
            line.end.x = x[i] - 50
            line.end.y = y[i] - 50
            line.FORWARD = True
            line.BACK = True
            # line 2
            msg = package.msgs.add()
            msg.type = Debug_Msg.LINE
            msg.color = color
            line = msg.line
            line.start.x = x[i] - 50
            line.start.y = y[i] + 50
            line.end.x = x[i] + 50
            line.end.y = y[i] - 50
            line.FORWARD = True
            line.BACK = True
    
    def draw_roadmap(self, package, sample_x, sample_y, road_map):
        for (i, edges) in zip(range(len(road_map)), road_map):
            for edge in edges:
                msg = package.msgs.add()
                msg.type = Debug_Msg.LINE
                msg.color = Debug_Msg.WHITE
                line = msg.line
                line.start.x = sample_x[i]
                line.start.y = sample_y[i]
                line.end.x = sample_x[edge]
                line.end.y = sample_y[edge]
                line.FORWARD = True
                line.BACK = True

    def draw_finalpath(self, package, x, y, color=Debug_Msg.GREEN):
        for i in range(len(x)-1):
            msg = package.msgs.add()
            msg.type = Debug_Msg.LINE
            msg.color = color
            line = msg.line
            line.start.x = x[i]
            line.start.y = y[i]
            line.end.x = x[i+1]
            line.end.y = y[i+1]
            line.FORWARD = True
            line.BACK = True

    def draw_tree(self, package, tree_x, tree_y, parent_index, color=Debug_Msg.WHITE):
        """绘制树结构"""
        for i in range(1, len(tree_x)):
            parent_i = parent_index[i]
            if parent_i >= 0:
                msg = package.msgs.add()
                msg.type = Debug_Msg.LINE
                msg.color = color
                line = msg.line
                line.start.x = tree_x[parent_i]
                line.start.y = tree_y[parent_i]
                line.end.x = tree_x[i]
                line.end.y = tree_y[i]
                line.FORWARD = True
                line.BACK = True

    def draw_all(self, sample_x, sample_y, road_map, path_x, path_y):
        package = Debug_Msgs()
        # 限制采样点数量，避免UDP缓冲区溢出
        max_points = 200
        if len(sample_x) > max_points:
            # 只绘制部分采样点
            step = len(sample_x) // max_points
            indices = list(range(0, len(sample_x), step))[:max_points]
            draw_sample_x = [sample_x[i] for i in indices]
            draw_sample_y = [sample_y[i] for i in indices]
            self.draw_points(package, draw_sample_x, draw_sample_y)
        else:
            self.draw_points(package, sample_x, sample_y)
        
        # 只绘制最终路径，不绘制完整路网（路网太大会导致缓冲区溢出）
        self.draw_finalpath(package, path_x, path_y)
        self.sock.sendto(package.SerializeToString(), self.debug_address)
    
    def send(self, package):
        self.sock.sendto(package.SerializeToString(), self.debug_address)
    
    def send_empty(self):
        """发送空包，清除显示"""
        package = Debug_Msgs()
        self.sock.sendto(package.SerializeToString(), self.debug_address)
    
    def create_visual_callback(self, start_x, start_y, goal_x, goal_y):
        """创建可视化回调函数，用于RRT*规划过程的实时可视化"""
        import math
        
        # 存储当前状态
        self.current_tree_x = []
        self.current_tree_y = []
        self.current_parent_index = []
        self.current_path_x = []
        self.current_path_y = []
        self.current_goal_reached = False
        
        def is_near_path(px, py, path_x, path_y, threshold=800):
            """判断点是否在路径附近"""
            for i in range(len(path_x)):
                dist = math.hypot(px - path_x[i], py - path_y[i])
                if dist < threshold:
                    return True
            return False
        
        def visual_callback(event, data):
            package = Debug_Msgs()
            
            if event == 'tree_expand':
                # 更新当前状态
                self.current_tree_x = data['tree_x']
                self.current_tree_y = data['tree_y']
                self.current_parent_index = data['parent_index']
                
                # 绘制完整的树结构
                tree_x = self.current_tree_x
                tree_y = self.current_tree_y
                parent_index = self.current_parent_index
                
                # 限制绘制的树边数量
                max_edges = 150
                edge_count = 0
                
                for i in range(1, len(tree_x)):
                    if edge_count >= max_edges:
                        break
                    parent_i = parent_index[i]
                    if parent_i >= 0:
                        msg = package.msgs.add()
                        msg.type = Debug_Msg.LINE
                        msg.color = Debug_Msg.WHITE
                        line = msg.line
                        line.start.x = tree_x[parent_i]
                        line.start.y = tree_y[parent_i]
                        line.end.x = tree_x[i]
                        line.end.y = tree_y[i]
                        line.FORWARD = True
                        line.BACK = True
                        edge_count += 1
                
                # 绘制新添加的边（黄色高亮）
                new_x = data['new_x']
                new_y = data['new_y']
                parent_x = data['parent_x']
                parent_y = data['parent_y']
                msg = package.msgs.add()
                msg.type = Debug_Msg.LINE
                msg.color = Debug_Msg.YELLOW
                line = msg.line
                line.start.x = parent_x
                line.start.y = parent_y
                line.end.x = new_x
                line.end.y = new_y
                line.FORWARD = True
                line.BACK = True
                
                # 绘制起点（蓝色）
                self.draw_point(package, start_x, start_y, Debug_Msg.BLUE)
                
                # 绘制目标点（红色）
                self.draw_point(package, goal_x, goal_y, Debug_Msg.RED)
                
            elif event == 'goal_reached':
                # 目标到达，更新状态
                self.current_goal_reached = True
                self.current_tree_x = data['tree_x']
                self.current_tree_y = data['tree_y']
                self.current_parent_index = data['parent_index']
                
                tree_x = self.current_tree_x
                tree_y = self.current_tree_y
                parent_index = self.current_parent_index
                
                # 绘制树（白色）
                max_edges = 100
                edge_count = 0
                for i in range(1, len(tree_x)):
                    if edge_count >= max_edges:
                        break
                    parent_i = parent_index[i]
                    if parent_i >= 0:
                        msg = package.msgs.add()
                        msg.type = Debug_Msg.LINE
                        msg.color = Debug_Msg.WHITE
                        line = msg.line
                        line.start.x = tree_x[parent_i]
                        line.start.y = tree_y[parent_i]
                        line.end.x = tree_x[i]
                        line.end.y = tree_y[i]
                        line.FORWARD = True
                        line.BACK = True
                        edge_count += 1
                
                # 绘制起点和目标点
                self.draw_point(package, start_x, start_y, Debug_Msg.BLUE)
                self.draw_point(package, goal_x, goal_y, Debug_Msg.RED)
                
            elif event == 'prune_start':
                # 剪枝开始：绘制原始路径（黄色）
                original_x = data['original_x']
                original_y = data['original_y']
                self.current_path_x = original_x
                self.current_path_y = original_y
                
                # 绘制树（白色背景）
                if self.current_tree_x:
                    for i in range(1, min(len(self.current_tree_x), 100)):
                        parent_i = self.current_parent_index[i] if i < len(self.current_parent_index) else -1
                        if parent_i >= 0:
                            msg = package.msgs.add()
                            msg.type = Debug_Msg.LINE
                            msg.color = Debug_Msg.WHITE
                            line = msg.line
                            line.start.x = self.current_tree_x[parent_i]
                            line.start.y = self.current_tree_y[parent_i]
                            line.end.x = self.current_tree_x[i]
                            line.end.y = self.current_tree_y[i]
                            line.FORWARD = True
                            line.BACK = True
                
                # 绘制原始路径（黄色）
                for i in range(len(original_x) - 1):
                    msg = package.msgs.add()
                    msg.type = Debug_Msg.LINE
                    msg.color = Debug_Msg.YELLOW
                    line = msg.line
                    line.start.x = original_x[i]
                    line.start.y = original_y[i]
                    line.end.x = original_x[i + 1]
                    line.end.y = original_y[i + 1]
                    line.FORWARD = True
                    line.BACK = True
                
                # 起点和目标点
                self.draw_point(package, start_x, start_y, Debug_Msg.BLUE)
                self.draw_point(package, goal_x, goal_y, Debug_Msg.RED)
                
            elif event == 'prune_step':
                # 剪枝过程
                smoothed_x = data['smoothed_x']
                smoothed_y = data['smoothed_y']
                pruned_x = data['pruned_segment_x']
                pruned_y = data['pruned_segment_y']
                
                self.current_path_x = smoothed_x
                self.current_path_y = smoothed_y
                
                # 绘制树（白色背景）
                if self.current_tree_x:
                    for i in range(1, min(len(self.current_tree_x), 80)):
                        parent_i = self.current_parent_index[i] if i < len(self.current_parent_index) else -1
                        if parent_i >= 0:
                            msg = package.msgs.add()
                            msg.type = Debug_Msg.LINE
                            msg.color = Debug_Msg.WHITE
                            line = msg.line
                            line.start.x = self.current_tree_x[parent_i]
                            line.start.y = self.current_tree_y[parent_i]
                            line.end.x = self.current_tree_x[i]
                            line.end.y = self.current_tree_y[i]
                            line.FORWARD = True
                            line.BACK = True
                
                # 绘制被剪枝的段（红色）
                if len(pruned_x) > 1:
                    for i in range(len(pruned_x) - 1):
                        msg = package.msgs.add()
                        msg.type = Debug_Msg.LINE
                        msg.color = Debug_Msg.RED
                        line = msg.line
                        line.start.x = pruned_x[i]
                        line.start.y = pruned_y[i]
                        line.end.x = pruned_x[i + 1]
                        line.end.y = pruned_y[i + 1]
                        line.FORWARD = True
                        line.BACK = True
                
                # 绘制平滑后的路径（绿色）
                for i in range(len(smoothed_x) - 1):
                    msg = package.msgs.add()
                    msg.type = Debug_Msg.LINE
                    msg.color = Debug_Msg.GREEN
                    line = msg.line
                    line.start.x = smoothed_x[i]
                    line.start.y = smoothed_y[i]
                    line.end.x = smoothed_x[i + 1]
                    line.end.y = smoothed_y[i + 1]
                    line.FORWARD = True
                    line.BACK = True
                
                # 起点和目标点
                self.draw_point(package, start_x, start_y, Debug_Msg.BLUE)
                self.draw_point(package, goal_x, goal_y, Debug_Msg.RED)
                
            elif event == 'prune_done':
                # 剪枝完成
                smoothed_x = data['smoothed_x']
                smoothed_y = data['smoothed_y']
                
                self.current_path_x = smoothed_x
                self.current_path_y = smoothed_y
                
                # 绘制树（白色背景）
                if self.current_tree_x:
                    for i in range(1, min(len(self.current_tree_x), 80)):
                        parent_i = self.current_parent_index[i] if i < len(self.current_parent_index) else -1
                        if parent_i >= 0:
                            msg = package.msgs.add()
                            msg.type = Debug_Msg.LINE
                            msg.color = Debug_Msg.WHITE
                            line = msg.line
                            line.start.x = self.current_tree_x[parent_i]
                            line.start.y = self.current_tree_y[parent_i]
                            line.end.x = self.current_tree_x[i]
                            line.end.y = self.current_tree_y[i]
                            line.FORWARD = True
                            line.BACK = True
                
                # 绘制最终路径（绿色）
                for i in range(len(smoothed_x) - 1):
                    msg = package.msgs.add()
                    msg.type = Debug_Msg.LINE
                    msg.color = Debug_Msg.GREEN
                    line = msg.line
                    line.start.x = smoothed_x[i]
                    line.start.y = smoothed_y[i]
                    line.end.x = smoothed_x[i + 1]
                    line.end.y = smoothed_y[i + 1]
                    line.FORWARD = True
                    line.BACK = True
                
                # 起点和目标点
                self.draw_point(package, start_x, start_y, Debug_Msg.BLUE)
                self.draw_point(package, goal_x, goal_y, Debug_Msg.RED)
                
                print(f"Pruning completed, removed {data['prune_count']} nodes")
            
            elif event == 'final_result':
                # 最终结果：只显示路径周围的采样点和树结构
                path_x = data['path_x']
                path_y = data['path_y']
                tree_x = self.current_tree_x
                tree_y = self.current_tree_y
                parent_index = self.current_parent_index
                
                near_threshold = 1000  # 路径附近的阈值
                
                # 绘制路径附近的树边（浅蓝色）
                if tree_x and parent_index:
                    edge_count = 0
                    max_edges = 100
                    for i in range(1, len(tree_x)):
                        if edge_count >= max_edges:
                            break
                        parent_i = parent_index[i] if i < len(parent_index) else -1
                        if parent_i >= 0:
                            # 检查边是否在路径附近
                            if (is_near_path(tree_x[i], tree_y[i], path_x, path_y, near_threshold) or
                                is_near_path(tree_x[parent_i], tree_y[parent_i], path_x, path_y, near_threshold)):
                                msg = package.msgs.add()
                                msg.type = Debug_Msg.LINE
                                msg.color = Debug_Msg.CYAN  # 浅蓝色
                                line = msg.line
                                line.start.x = tree_x[parent_i]
                                line.start.y = tree_y[parent_i]
                                line.end.x = tree_x[i]
                                line.end.y = tree_y[i]
                                line.FORWARD = True
                                line.BACK = True
                                edge_count += 1
                
                # 绘制路径附近的采样点（黄色小点）
                if tree_x:
                    point_count = 0
                    max_points = 50
                    for i in range(len(tree_x)):
                        if point_count >= max_points:
                            break
                        if is_near_path(tree_x[i], tree_y[i], path_x, path_y, near_threshold):
                            # 绘制小叉号表示采样点
                            msg = package.msgs.add()
                            msg.type = Debug_Msg.LINE
                            msg.color = Debug_Msg.YELLOW
                            line = msg.line
                            line.start.x = tree_x[i] + 30
                            line.start.y = tree_y[i] + 30
                            line.end.x = tree_x[i] - 30
                            line.end.y = tree_y[i] - 30
                            line.FORWARD = True
                            line.BACK = True
                            
                            msg = package.msgs.add()
                            msg.type = Debug_Msg.LINE
                            msg.color = Debug_Msg.YELLOW
                            line = msg.line
                            line.start.x = tree_x[i] - 30
                            line.start.y = tree_y[i] + 30
                            line.end.x = tree_x[i] + 30
                            line.end.y = tree_y[i] - 30
                            line.FORWARD = True
                            line.BACK = True
                            point_count += 1
                
                # 绘制最终路径（绿色加粗效果 - 绘制两次）
                for i in range(len(path_x) - 1):
                    msg = package.msgs.add()
                    msg.type = Debug_Msg.LINE
                    msg.color = Debug_Msg.GREEN
                    line = msg.line
                    line.start.x = path_x[i]
                    line.start.y = path_y[i]
                    line.end.x = path_x[i + 1]
                    line.end.y = path_y[i + 1]
                    line.FORWARD = True
                    line.BACK = True
                
                # 绘制路径节点（白色点）
                for i in range(len(path_x)):
                    msg = package.msgs.add()
                    msg.type = Debug_Msg.LINE
                    msg.color = Debug_Msg.WHITE
                    line = msg.line
                    line.start.x = path_x[i] + 80
                    line.start.y = path_y[i] + 80
                    line.end.x = path_x[i] - 80
                    line.end.y = path_y[i] - 80
                    line.FORWARD = True
                    line.BACK = True
                    
                    msg = package.msgs.add()
                    msg.type = Debug_Msg.LINE
                    msg.color = Debug_Msg.WHITE
                    line = msg.line
                    line.start.x = path_x[i] - 80
                    line.start.y = path_y[i] + 80
                    line.end.x = path_x[i] + 80
                    line.end.y = path_y[i] - 80
                    line.FORWARD = True
                    line.BACK = True
                
                # 起点（蓝色大点）
                msg = package.msgs.add()
                msg.type = Debug_Msg.LINE
                msg.color = Debug_Msg.BLUE
                line = msg.line
                line.start.x = start_x + 100
                line.start.y = start_y + 100
                line.end.x = start_x - 100
                line.end.y = start_y - 100
                line.FORWARD = True
                line.BACK = True
                msg = package.msgs.add()
                msg.type = Debug_Msg.LINE
                msg.color = Debug_Msg.BLUE
                line = msg.line
                line.start.x = start_x - 100
                line.start.y = start_y + 100
                line.end.x = start_x + 100
                line.end.y = start_y - 100
                line.FORWARD = True
                line.BACK = True
                
                # 目标点（红色大点）
                msg = package.msgs.add()
                msg.type = Debug_Msg.LINE
                msg.color = Debug_Msg.RED
                line = msg.line
                line.start.x = goal_x + 100
                line.start.y = goal_y + 100
                line.end.x = goal_x - 100
                line.end.y = goal_y - 100
                line.FORWARD = True
                line.BACK = True
                msg = package.msgs.add()
                msg.type = Debug_Msg.LINE
                msg.color = Debug_Msg.RED
                line = msg.line
                line.start.x = goal_x - 100
                line.start.y = goal_y + 100
                line.end.x = goal_x + 100
                line.end.y = goal_y - 100
                line.FORWARD = True
                line.BACK = True
                
                print("Final result visualization completed")
            
            # 发送数据包
            try:
                self.sock.sendto(package.SerializeToString(), self.debug_address)
            except OSError:
                pass
            
            # 添加延时
            time.sleep(0.05)
        
        return visual_callback


if __name__ == '__main__':
    debugger = Debugger()
    package = Debug_Msgs()
    debugger.draw_circle(package, x=0, y=500)
    debugger.draw_line(package, x1=0, y1=2500, x2=600, y2=2500)
    debugger.draw_lines(package, x1=[0,0], y1=[0,2000], x2=[2000,2000], y2=[0,2000])
    debugger.draw_point(package, x=500, y=500)
    debugger.draw_points(package, x=[1000, 2000], y=[3000, 3000])
    debugger.send(package)
