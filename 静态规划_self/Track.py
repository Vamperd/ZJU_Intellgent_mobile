import math
class Track(object):
    def __init__(self, target_flag = 0, value_of_k_v = 0.1, value_of_k_w = -0.1):
        self.target_flag = target_flag
        self.value_of_k_v = value_of_k_v
        self.value_of_k_w = value_of_k_w
    def test(self, vx, vy, start_x, start_y, goal_x, goal_y, path_x, path_y):
        if (start_x == goal_x) and (start_y == goal_y):
            robot_v = 0
            robot_vw = 0
            return robot_v, robot_vw
        robot_v = self.get_v(start_x, start_y, path_x, path_y)
        robot_vw = self.get_vw(vx, vy, start_x, start_y, path_x, path_y)

        return robot_v, robot_vw

    def get_v(self, start_x, start_y, path_x, path_y):
        delta_x = path_x[-1] - start_x
        delta_y = path_y[-1] - start_y
        robot_v = self.value_of_k_v*(abs(delta_x) + abs(delta_y))
        return robot_v

    def get_vw(self, vx, vy, start_x, start_y, path_x, path_y):
        delta_x = path_x[-1] - start_x
        delta_y = path_y[-1] - start_y
        angle_xy = math.atan2(delta_y, delta_x)
        # angle_xy = angle_xy * 180/math.pi
        angle_v = math.atan2(vy, vx)
        # angle_v = angle_v * 180/math.pi
        delta_angele = angle_xy - angle_v
        dw = self.value_of_k_w*delta_angele
        return dw