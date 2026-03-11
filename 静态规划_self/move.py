import math
import time
def cal_dist(x1,y1,x2,y2):
    dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    return dist

def cal_theta(x1,y1,x2,y2):
    if x1 >= x2 and y1 >= y2:
        return math.atan((y1 - y2) / (x1 - x2)) - math.pi
    if x1 >= x2 and y1 < y2:
        return math.atan((y1 - y2) / (x1 - x2)) + math.pi
    if x1 < x2:
        return math.atan((y1 - y2) / (x1 - x2))

def cal_theta_error(theta1, vision):
    theta_error = theta1 - vision.my_robot.orientation
    while theta_error > math.pi:
        theta_error = theta_error - 2 * math.pi
    while theta_error < -math.pi:
        theta_error = theta_error + 2 * math.pi
    return theta_error

def function(x):
    if x > 3000:
        x = 3000
    return -1/3000*x*(x-6000)

def PID_vx(vx_error, vx_error_sum, vx_error_last, flag):
    Kp = 1.5
    Ki = 0.01
    Kd = 0
    
    if vx_error_sum > 10000:
        vx_error_sum = 10000
    output = Kp * vx_error + Ki * vx_error_sum + Kd * vx_error_last
    if flag:
        output = function(output)
        output = max(output, 1200)
    else:
        output = function(output)
        output = max(output, 450)
    if output > 4000:
        output = 4000
    return output

def PID_vw(vw_error, vw_error_sum, vw_error_last):
    Kp = 5
    Ki = 0.015
    Kd = 0.015
    if vw_error_sum > 40:
        vw_error_sum = 40
    output = Kp * vw_error + Ki * vw_error_sum + Kd * vw_error_last
    if output > 5:
        output = 5
    return output

class Move(object):
    def __init__(self):
        print('go')
        self.position_error_last = 0
        self.position_error = 0
        self.position_error_sum = 0
        self.theta_error_last = 0
        self.theta_error = 0
        self.theta_error_sum = 0

    def get_action(self, vision, path_x, path_y, Next_point_index, direction=1):
        current_x, current_y = vision.my_robot.x, vision.my_robot.y
        # 计算当前点与下一个目标点的距离
        dist_error1 = cal_dist(current_x,current_y, path_x[Next_point_index], path_y[Next_point_index])
        # 计算当前位置与相邻目标点的距离（用于平滑误差跳变）
        if direction == 1:
            # 正向移动：检查是否有下一个点
            if Next_point_index + 1 < len(path_x):
                dist_error2 = cal_dist(current_x,current_y, path_x[Next_point_index+1], path_y[Next_point_index+1])
            else:
                dist_error2 = dist_error1  # 边界情况，直接使用当前距离
        else:
            # 反向移动：检查是否有上一个点
            if Next_point_index - 1 >= 0:
                dist_error2 = cal_dist(current_x,current_y, path_x[Next_point_index-1], path_y[Next_point_index-1])
            else:
                dist_error2 = dist_error1  # 边界情况，直接使用当前距离
        # 计算当前点与下一个目标点的角度
        theta_error = cal_theta(current_x,current_y, path_x[Next_point_index], path_y[Next_point_index])

        dist_error = min(dist_error1,dist_error2)

        self.position_error_last = self.position_error
        self.position_error = dist_error
        self.position_error_sum = self.position_error_sum + self.position_error

        self.theta_error_last = self.theta_error
        self.theta_error = cal_theta_error(theta_error, vision)
        self.theta_error_sum = self.theta_error_sum + self.theta_error

        if Next_point_index == 0 or Next_point_index == len(path_x)-1:
            flag = False
        else:
            flag = True
        vx = PID_vx(self.position_error, self.position_error_sum, self.position_error_last, flag)
        vw = PID_vw(self.theta_error, self.theta_error_sum, self.theta_error_last)
        return vx, vw

    def turn_arround(self, action, vision):
        start_theta = vision.my_robot.orientation
        theta_change = 0
        while theta_change < math.pi-0.2:
            if theta_change > 2*math.pi/3:
                action.sendCommand(vx=0, vw=2.5)
            else:
                action.sendCommand(vx=0, vw=5)
            time.sleep(0.1)
            current_theta = vision.my_robot.orientation
            theta_change = abs(current_theta-start_theta)
