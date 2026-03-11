import socket
import sys
import time
import numpy as np

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

    def draw_line(self, package, x1, y1, x2, y2):
        msg = package.msgs.add()
        msg.type = Debug_Msg.LINE
        msg.color = Debug_Msg.WHITE
        line = msg.line
        line.start.x = x1
        line.start.y = y1
        line.end.x = x2
        line.end.y = y2
        line.FORWARD = True
        line.BACK = True
    
    def draw_lines(self, package, x1, y1, x2, y2):
        for i in range(len(x1)):
            msg = package.msgs.add()
            msg.type = Debug_Msg.LINE
            msg.color = Debug_Msg.WHITE
            line = msg.line
            line.start.x = x1[i]
            line.start.y = y1[i]
            line.end.x = x2[i]
            line.end.y = y2[i]
            line.FORWARD = True
            line.BACK = True

    def draw_finalpath(self, package, x, y):
        for i in range(len(x)-1):
            msg = package.msgs.add()
            msg.type = Debug_Msg.LINE
            msg.color = Debug_Msg.GREEN
            line = msg.line
            line.start.x = x[i]
            line.start.y = y[i]
            line.end.x = x[i+1]
            line.end.y = y[i+1]
            line.FORWARD = True
            line.BACK = True

    def draw_point(self, package, x, y):
        msg = package.msgs.add()
        # line 1
        msg.type = Debug_Msg.LINE
        msg.color = Debug_Msg.WHITE
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
        msg.color = Debug_Msg.WHITE
        line = msg.line
        line.start.x = x - 50
        line.start.y = y + 50
        line.end.x = x + 50
        line.end.y = y - 50
        line.FORWARD = True
        line.BACK = True

    def draw_points(self, package, x, y):
        for i in range(len(x)):
            # line 1
            msg = package.msgs.add()
            msg.type = Debug_Msg.LINE
            msg.color = Debug_Msg.WHITE
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
            msg.color = Debug_Msg.WHITE
            line = msg.line
            line.start.x = x[i] - 50
            line.start.y = y[i] + 50
            line.end.x = x[i] + 50
            line.end.y = y[i] - 50
            line.FORWARD = True
            line.BACK = True
    
    def send(self, package):
        self.sock.sendto(package.SerializeToString(), self.debug_address)

    def draw_arrow(self, package, x1, y1, x2, y2, color=Debug_Msg.RED, width=5):
        """绘制带箭头的向量"""
        msg = package.msgs.add()
        msg.type = Debug_Msg.LINE
        msg.color = color
        line = msg.line
        line.start.x = x1
        line.start.y = y1
        line.end.x = x2
        line.end.y = y2
        line.FORWARD = True
        line.BACK = False

    def draw_force_vectors(self, package, robot_x, robot_y, F_total, Fatt, Frep, scale=0.5):
        """
        绘制人工势场力的向量
        robot_x, robot_y: 机器人位置
        F_total: 合力
        Fatt: 引力
        Frep: 斥力
        scale: 缩放因子
        """
        # 绘制引力（青色 - CYAN）
        if np.linalg.norm(Fatt) > 0.001:
            end_x = robot_x + Fatt[0] * scale
            end_y = robot_y + Fatt[1] * scale
            self.draw_arrow(package, robot_x, robot_y, end_x, end_y, Debug_Msg.CYAN, 8)
        
        # 绘制斥力（红色）
        if np.linalg.norm(Frep) > 0.001:
            end_x = robot_x + Frep[0] * scale
            end_y = robot_y + Frep[1] * scale
            self.draw_arrow(package, robot_x, robot_y, end_x, end_y, Debug_Msg.RED, 8)
        
        # 绘制合力（黄色 - 更醒目）
        if np.linalg.norm(F_total) > 0.001:
            end_x = robot_x + F_total[0] * scale
            end_y = robot_y + F_total[1] * scale
            self.draw_arrow(package, robot_x, robot_y, end_x, end_y, Debug_Msg.YELLOW, 12)
            
    def draw_obstacle_radius(self, package, obs_x, obs_y, radius=800, color=Debug_Msg.PURPLE):
        """绘制障碍物斥力影响范围"""
        msg = package.msgs.add()
        msg.type = Debug_Msg.ARC
        msg.color = color
        arc = msg.arc
        arc.rectangle.point1.x = obs_x - radius
        arc.rectangle.point1.y = obs_y - radius
        arc.rectangle.point2.x = obs_x + radius
        arc.rectangle.point2.y = obs_y + radius
        arc.start = 0
        arc.end = 360
        arc.FILL = False
        
    def draw_velocity_vector(self, package, robot_x, robot_y, velocity, scale=2.0, color=Debug_Msg.WHITE):
        """绘制机器人实际速度向量"""
        if np.linalg.norm(velocity) > 0.001:
            end_x = robot_x + velocity[0] * scale
            end_y = robot_y + velocity[1] * scale
            self.draw_arrow(package, robot_x, robot_y, end_x, end_y, color, 6)
            
    def draw_goal_area(self, package, goal_x, goal_y, radius=500, color=Debug_Msg.GREEN):
        """绘制目标点区域（末端加速区域）"""
        msg = package.msgs.add()
        msg.type = Debug_Msg.ARC
        msg.color = color
        arc = msg.arc
        arc.rectangle.point1.x = goal_x - radius
        arc.rectangle.point1.y = goal_y - radius
        arc.rectangle.point2.x = goal_x + radius
        arc.rectangle.point2.y = goal_y + radius
        arc.start = 0
        arc.end = 360
        arc.FILL = False

    def draw_text_info(self, package, x, y, text, color=Debug_Msg.WHITE):
        """绘制文本信息（用点阵模拟）"""
        # 由于protobuf可能没有文本功能，我们用一个小圆点表示信息位置
        msg = package.msgs.add()
        msg.type = Debug_Msg.ARC
        msg.color = color
        arc = msg.arc
        arc.rectangle.point1.x = x - 30
        arc.rectangle.point1.y = y - 30
        arc.rectangle.point2.x = x + 30
        arc.rectangle.point2.y = y + 30
        arc.start = 0
        arc.end = 360
        arc.FILL = True


if __name__ == '__main__':
    debugger = Debugger()
    package = Debug_Msgs()
    debugger.draw_circle(package, x=0, y=500)
    debugger.draw_line(package, x1=0, y1=2500, x2=600, y2=2500)
    debugger.draw_lines(package, x1=[0,0], y1=[0,2000], x2=[2000,2000], y2=[0,2000])
    debugger.draw_point(package, x=500, y=500)
    debugger.draw_points(package, x=[1000, 2000], y=[3000, 3000])
    debugger.send(package)
