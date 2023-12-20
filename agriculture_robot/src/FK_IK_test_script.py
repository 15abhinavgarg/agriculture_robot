from sympy import *
import numpy as np
from numpy import pi
from tqdm import tqdm
import csv
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import cmath
from math import cos as m_cos
from math import sin as m_sin
from math import atan2 as m_atan2
from math import acos as m_acos
from math import asin as m_asin
from math import sqrt as m_sqrt
global mat
mat=np.matrix
# ****** Coefficients ******
global d1, a2, a3, a7, d4, d5, d6
d1 =  0.1273
a2 = -0.612
a3 = -0.5723
d4 =  0.163941
d5 =  0.1157
d6 =  0.0922 + 0.0922

global d, a, alph

#d = mat([0.089159, 0, 0, 0.10915, 0.09465, 0.0823]) ur5
d = mat([d1, 0, 0, d4, d5, d6])#ur10 mm
# a =mat([0 ,-0.425 ,-0.39225 ,0 ,0 ,0]) ur5
a =mat([0 ,a2 , a3 ,0 ,0 ,0])#ur10 mm
#alph = mat([math.pi/2, 0, 0, math.pi/2, -math.pi/2, 0 ])  #ur5
alph = mat([pi/2, 0, 0, pi/2, -pi/2, 0 ]) # ur10

from numpy import linalg


theta1, theta2, theta3, theta4, theta5, theta6 = symbols(
    "θ₁ θ₂ θ₃ θ₄ θ₅ θ₆")

# This class handles the insertion of dh parameters and calculation of transforms for each joint


class forwardKinematicsSolver:
    def __init__(self):
        self.rows = {
            'row1': {'a': None, 'alpha': None, 'd': None, 'theta': None}}
        self.configs = {'homePosition': {theta1: 0, theta2: 0,
                                         theta3: 0, theta4: 0, theta5: 0, theta6: 0}}
        self.transforms = {'1->2': None}
        self.final_transform = np.eye(4)

    def insert_dh_parameters(self, row_name, a, alpha, d, theta):
        self.rows[row_name] = {'a': a, 'alpha': alpha, 'd': d, 'theta': theta}

    def show_dh_table(self):
        headers = ['frames', 'a', 'alpha', 'd', 'theta']
        data = []

        for row, transform in zip(self.rows.values(), self.transforms):
            data.append([transform, row['a'], row['alpha'],
                         row['d'], row['theta']])

        fig, ax = plt.subplots()
        ax.axis('off')
        ax.set_title("DH Table for UR10e Robot",
                     fontsize=16, fontweight='bold')

        table_data = [headers] + data
        table = ax.table(cellText=table_data, loc='center',
                         cellLoc='center', colLabels=None, cellColours=[['#f2f2f2'] * len(headers)] * len(table_data),
                         bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(14)

        for key, cell in table.get_celld().items():
            cell.set_linewidth(1.0)
            if key[0] == 0:
                cell.set_text_props(weight='bold')

        plt.subplots_adjust(left=0.2, top=0.8)
        plt.show()

    def compute_transforms(self):
        curr_row = 1
        for row in self.rows:
            T = Matrix([[cos(self.rows[row]['theta']), -sin(self.rows[row]['theta'])*cos(self.rows[row]['alpha'] * (pi/180)),
                         sin(self.rows[row]['theta'])*sin(self.rows[row]['alpha']*(pi/180)), self.rows[row]['a']*cos(self.rows[row]['theta'])],
                        [sin(self.rows[row]['theta']), cos(self.rows[row]['theta']) * cos(self.rows[row]['alpha'] * (pi/180)),
                         -cos(self.rows[row]['theta'])*sin(self.rows[row]['alpha']*(pi/180)), self.rows[row]['a']*sin(self.rows[row]['theta'])],
                        [0, sin(self.rows[row]['alpha']*(pi/180)),
                         cos(self.rows[row]['alpha']*(pi/180)), self.rows[row]['d']],
                        [0, 0, 0, 1]])
            transform_name = str(curr_row) + "->" + str(curr_row+1)
            curr_row += 1
            self.transforms[transform_name] = T

        for transform in self.transforms:
            # multiplying matrices that were added from first to last ie T1@T2@T3...T6
            self.final_transform = self.final_transform @ self.transforms[transform]
        print("Individual joint transforms were computed from dh parameters")

    def rotation_to_euler(self, matrix):
        # follows yaw pitch roll convention
        rotation = Rotation.from_matrix(matrix[:3, :3])
        rpy_angles = rotation.as_euler('ZYX', degrees=True)

        return rpy_angles[2], rpy_angles[1], rpy_angles[0]

    def compute_FK(self, config):
        if isinstance(config, str):
            Tf = self.final_transform.subs([(theta1, self.configs[config]['theta1']), (theta2, self.configs[config]['theta2']), (theta3, self.configs[config]['theta3']), (
                theta4, self.configs[config]['theta4']), (theta5, self.configs[config]['theta5']), (theta6, self.configs[config]['theta6'])])
            T_evaluated = Tf.evalf()
            # print(
                # f"\n The estimated transformation matrix for the config, {config}:{list(self.configs[config].values())} is:\n")
        else:
            Tf = self.final_transform.subs([(theta1, config[0]), (theta2, config[1]), (theta3, config[2]), (
                theta4, config[3]), (theta5, config[4]), (theta6, config[5])])
            T_evaluated = Tf.evalf()
            # print(f"\n The estimated transformation matrix for {config} is:\n")
        # pprint(T_evaluated)
        print(
            f'End effector x-pos:{round(T_evaluated[0,3],2)}, y-pos:{round(T_evaluated[1,3],2)}, z-pos:{round(T_evaluated[2,3],2)}')
        roll, pitch, yaw = self.rotation_to_euler(T_evaluated)
        print(f"Roll: {roll}, Pitch: {pitch}, Yaw: {yaw}")

        return T_evaluated

    def insert_geometric_configs(self, config_name, angles):
        self.configs[config_name] = {'theta1': angles[0], 'theta2': angles[1],
                                     'theta3': angles[2], 'theta4': angles[3], 'theta5': angles[4], 'theta6': angles[5]}

    def get_transforms(self):
        return list(self.transforms.values())

    @ staticmethod
    def z_to_script(raw_z):
        # UR10 CAD model has the following sign convention
        # theta 2,3,4,and 6 direction of z axis is reversed
        return [raw_z[0], -raw_z[1], -raw_z[2], -raw_z[3], raw_z[4], -raw_z[5]]

    @staticmethod
    def z_to_gazebo(raw_z):
        # converts back to gazebo CAD convention before sending to controller
        return [raw_z[0], -raw_z[1], -raw_z[2], -raw_z[3], raw_z[4], -raw_z[5]]

    def write_to_csv(self, thetas):
        with open('live_data.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6'])
            csv_writer.writerow(thetas)

# This class handles the computation of the inverse kinematics calculation to draw a circle in 20 seconds


class inverseKinematicsSolver:
    def __init__(self, all_transforms):
        self.transforms = all_transforms
        self.T_0 = []

        # building the transformation matrices from base frame to all joints (T1@T2@T3...T6)
        T = Matrix.eye(4, 4)
        for frame in range(len(self.transforms)):
            for transform in self.transforms[0:frame+1]:
                T = T @ transform  #
            self.T_0.append(T)
            T = Matrix.eye(4, 4)

        self.J_symbolic = None

        self.A1 = self.transforms[0]
        self.A2 = self.transforms[1]
        self.A3 = self.transforms[2]
        self.A4 = self.transforms[3]
        self.A5 = self.transforms[4]
        self.A6 = self.transforms[5]

        self.A01 = self.A1
        self.A02 = self.A1@self.A2
        self.A03 = self.A02 @ self.A3
        self.A04 = self.A03 @ self.A4
        self.A05 = self.A04 @ self.A5
        self.A06 = self.A05 @ self.A6
        print("Base transforms for each joint was calculated")

        self.base_transforms = [self.A01, self.A02,
                                self.A03, self.A04, self.A05, self.A06]

    def get_base_transforms(self):
        return self.base_transforms

    def get_jacobian(self):
        return self.J_symbolic

    def jacobian_from_partials(self):
        Z0 = Matrix([0, 0, 1])
        Z1 = self.A01[:3, 2]
        Z2 = self.A02[:3, 2]
        Z3 = self.A03[:3, 2]
        Z4 = self.A04[:3, 2]
        Z5 = self.A05[:3, 2]
        Z6 = self.A06[:3, 2]

        O0 = Matrix([0, 0, 0])
        O1 = self.A01[:3, 3]
        O2 = self.A02[:3, 3]
        O3 = self.A03[:3, 3]
        O4 = self.A04[:3, 3]
        O5 = self.A05[:3, 3]
        O6 = self.A06[:3, 3]

        px = self.A06[0, 3]
        py = self.A06[1, 3]
        pz = self.A06[2, 3]

        a11 = diff(px, theta1)
        a12 = diff(px, theta2)
        a13 = diff(px, theta3)
        a14 = diff(px, theta4)
        a15 = diff(px, theta5)
        a16 = diff(px, theta6)

        a21 = diff(py, theta1)
        a22 = diff(py, theta2)
        a23 = diff(py, theta3)
        a24 = diff(py, theta4)
        a25 = diff(py, theta5)
        a26 = diff(py, theta6)

        a31 = diff(pz, theta1)
        a32 = diff(pz, theta2)
        a33 = diff(pz, theta3)
        a34 = diff(pz, theta4)
        a35 = diff(pz, theta5)
        a36 = diff(pz, theta6)

        J = Matrix([[a11, a12, a13, a14, a15, a16], [a21, a22, a23, a24, a25, a26], [
                   a31, a32, a33, a34, a35, a36], [Z1, Z2, Z3, Z4, Z5, Z6]])
        print("Symbolic jacobian matrix was calcuated using partial derivative method")
        print("\nThe final Jacobian matrix size is: ", J.shape)

        self.J_symbolic = J

    def circle_trajectory_IK(self):
        theta_joint = Matrix([-180, -90, 2, -90, -2, -90])*(pi/180)

        figure, ax = plt.subplots()
        ax = plt.axes(projection='3d')
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')

        timestep = 0.1
        x_data = []
        y_data = []
        z_data = []
        thetas = []
        # dets = []

        damping_factor = 0.5
        sim_time = 20  # in seconds, max 20.
        radius = 600

        print("Computing end effector positions using inverse kinematics to draw the circle...")
        for t in tqdm(np.arange(0, sim_time+timestep, timestep)):
            x_dot = radius/10 * np.pi * cos(t*np.pi/10)
            z_dot = -radius/10 * np.pi * sin(t*np.pi/10)

            V = Matrix([x_dot, 0.0, z_dot, 0.0, 0.0, 0.0])
            J = self.J_symbolic.evalf(3, subs={
                                      theta1: theta_joint[0], theta2: theta_joint[1], theta3: theta_joint[2], theta4: theta_joint[3], theta5: theta_joint[4], theta6: theta_joint[5]})
            # dets.append(J.det())
            J = np.array(J, dtype=np.float32)
            # J_inv = np.linalg.pinv(J)
            J_inv = np.transpose(J) @ np.linalg.inv(J @ np.transpose(J) + damping_factor**2 * np.eye(J.shape[0]))

            theta_dot = J_inv*V  # inverse calculation messes up the joint angles
            theta_joint = theta_joint + theta_dot * timestep
            thetas.append(theta_joint)

            T = self.base_transforms[5].evalf(3, subs={theta1: theta_joint[0], theta2: theta_joint[1],
                                              theta3: theta_joint[2], theta4: theta_joint[3], theta5: theta_joint[4], theta6: theta_joint[5]})
            ax.scatter3D(T[0, 3], T[1, 3], T[2, 3])
            P_7 = Matrix([0, 0, 0, 1])
            P = T * P_7

            # Appending position data to the arrays
            x_data.append(P[0, 0])
            y_data.append(P[1, 0])
            z_data.append(P[2, 0])

        plt.show()

        with open('scatter_data.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['X', 'Y', 'Z'])
            csv_writer.writerows(zip(x_data, y_data, z_data))

        with open('joint_data.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6'])
            csv_writer.writerows(thetas)

    def line_trajectory_IK(self, initial_position, goal_position):
        theta_joint = Matrix([-180, -90, 2, -90, -2, -90])*(pi/180)

        # initial_position = np.array([-1.13,-348.34,1427.30])
        # goal_position = np.array([500,500,500])
        initial_position = np.array(initial_position)
        goal_position = np.array(goal_position)

        distance = np.linalg.norm(goal_position-initial_position)

        T = 20 # Time period
        V_x = x_dot = (goal_position[0]-initial_position[0])/T
        V_y = y_dot = (goal_position[1]-initial_position[1])/T
        V_z = z_dot = (goal_position[2]-initial_position[2])/T

        figure, ax = plt.subplots()
        ax = plt.axes(projection='3d')
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')

        timestep = 0.1
        x_data = []
        y_data = []
        z_data = []
        thetas = []
        # dets = []

        damping_factor = 0.5
        sim_time = T  # in seconds, max 20.

        print("Computing end effector positions using inverse kinematics to draw the circle...")
        for t in tqdm(np.arange(0, sim_time+timestep, timestep)):
            # x_dot = radius/10 * np.pi * cos(t*np.pi/10)
            # z_dot = -radius/10 * np.pi * sin(t*np.pi/10)

            V = Matrix([x_dot, y_dot, z_dot, 0.0, 0.0, 0.0])
            J = self.J_symbolic.evalf(3, subs={
                                      theta1: theta_joint[0], theta2: theta_joint[1], theta3: theta_joint[2], theta4: theta_joint[3], theta5: theta_joint[4], theta6: theta_joint[5]})
            # dets.append(J.det())
            J = np.array(J, dtype=np.float32)
            # J_inv = np.linalg.pinv(J)
            J_inv = np.transpose(J) @ np.linalg.inv(J @ np.transpose(J) + damping_factor**2 * np.eye(J.shape[0]))

            theta_dot = J_inv*V  # inverse calculation messes up the joint angles
            theta_joint = theta_joint + theta_dot * timestep
            thetas.append(theta_joint)

            T = self.base_transforms[5].evalf(3, subs={theta1: theta_joint[0], theta2: theta_joint[1],
                                              theta3: theta_joint[2], theta4: theta_joint[3], theta5: theta_joint[4], theta6: theta_joint[5]})
            ax.scatter3D(T[0, 3], T[1, 3], T[2, 3])
            P_7 = Matrix([0, 0, 0, 1])
            P = T * P_7

            # Appending position data to the arrays
            x_data.append(P[0, 0])
            y_data.append(P[1, 0])
            z_data.append(P[2, 0])

        plt.show()

        with open('scatter_data.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['X', 'Y', 'Z'])
            csv_writer.writerows(zip(x_data, y_data, z_data))

        with open('joint_data.csv', 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(
                ['theta1', 'theta2', 'theta3', 'theta4', 'theta5', 'theta6'])
            csv_writer.writerows(thetas)

    def AH(self, n, th, c):

        T_a = mat(np.identity(4), copy=False)
        T_a[0, 3] = a[0, n-1]
        T_d = mat(np.identity(4), copy=False)
        T_d[2, 3] = d[0, n-1]

        Rzt = mat([[m_cos(th[n-1, c]), -m_sin(th[n-1, c]), 0, 0],
                   [m_sin(th[n-1, c]),  m_cos(th[n-1, c]), 0, 0],
                   [0,               0,              1, 0],
                   [0,               0,              0, 1]], copy=False)

        Rxa = mat([[1, 0,                 0,                  0],
                   [0, m_cos(alph[0, n-1]), -m_sin(alph[0, n-1]),   0],
                   [0, m_sin(alph[0, n-1]),  m_cos(alph[0, n-1]),   0],
                   [0, 0,                 0,                  1]], copy=False)

        A_i = T_d * Rzt * T_a * Rxa

        return A_i

    def HTrans(self, th, c):
        A_1 = AH(1, th, c)
        A_2 = AH(2, th, c)
        A_3 = AH(3, th, c)
        A_4 = AH(4, th, c)
        A_5 = AH(5, th, c)
        A_6 = AH(6, th, c)

        T_06 = A_1*A_2*A_3*A_4*A_5*A_6

        return T_06

    def compute_position_IK(self, desired_pos):
        th = mat(np.zeros((6, 8)))
        P_05 = (desired_pos * mat([0, 0, -d6, 1]).T-mat([0, 0, 0, 1]).T)

        # solving for theta1
        psi = m_atan2(P_05[2-1, 0], P_05[1-1, 0])
        phi = m_acos(d4 / m_sqrt(P_05[2-1, 0] *
                    P_05[2-1, 0] + P_05[1-1, 0]*P_05[1-1, 0]))
        # The two solutions for theta1 correspond to the shoulder left and right
        th[0, 0:4] = pi/2 + psi + phi
        th[0, 4:8] = pi/2 + psi - phi
        th = th.real

        # solving for theta2
        cl = [0, 4]  # wrist up or down
        for i in range(0, len(cl)):
            c = cl[i]
            T_10 = linalg.inv(self.AH(1, th, c))
            T_16 = T_10 * desired_pos
            th[4, c:c+2] = + m_acos((T_16[2, 3]-d4)/d6)
            th[4, c+2:c+4] = - m_acos((T_16[2, 3]-d4)/d6)

        th = th.real

        # solving for theta6
        # theta6 is not well-defined when m_sin(theta5) = 0 or when T16(1,3), T16(2,3) = 0.
        cl = [0, 2, 4, 6]
        for i in range(0, len(cl)):
            c = cl[i]
            T_10 = linalg.inv(self.AH(1, th, c))
            T_16 = linalg.inv(T_10 * desired_pos)
            th[5, c:c+2] = m_atan2((-T_16[1, 2]/m_sin(th[4, c])),
                                (T_16[0, 2]/m_sin(th[4, c])))

        th = th.real

        # solving for theta3
        cl = [0, 2, 4, 6]
        for i in range(0, len(cl)):
            c = cl[i]
            T_10 = linalg.inv(self.AH(1, th, c))
            T_65 = self.AH(6, th, c)
            T_54 = self.AH(5, th, c)
            T_14 = (T_10 * desired_pos) * linalg.inv(T_54 * T_65)
            P_13 = T_14 * mat([0, -d4, 0, 1]).T - mat([0, 0, 0, 1]).T
            t3 = cmath.acos((linalg.norm(P_13)**2 - a2**2 - a3**2)/(2 * a2 * a3))  
            th[2, c] = t3.real
            th[2, c+1] = -t3.real

        # solving for theta2 and theta4
        cl = [0, 1, 2, 3, 4, 5, 6, 7]
        for i in range(0, len(cl)):
            c = cl[i]
            T_10 = linalg.inv(self.AH(1, th, c))
            T_65 = linalg.inv(self.AH(6, th, c))
            T_54 = linalg.inv(self.AH(5, th, c))
            T_14 = (T_10 * desired_pos) * T_65 * T_54
            P_13 = T_14 * mat([0, -d4, 0, 1]).T - mat([0, 0, 0, 1]).T

            # theta 2
            th[1, c] = -m_atan2(P_13[1], -P_13[0]) + \
                m_asin(a3 * m_sin(th[2, c])/linalg.norm(P_13))
            # theta 4
            T_32 = linalg.inv(self.AH(3, th, c))
            T_21 = linalg.inv(self.AH(2, th, c))
            T_34 = T_32 * T_21 * T_14
            th[3, c] = m_atan2(T_34[1, 0], T_34[0, 0])
        th = th.real

        thetas = np.array(th)

        all_solutions= []
        for solution in range(8):
            solution_list = []
            # print(f"Solution: {solution}\n")
            for theta in range(6):
                solution_list.append(thetas[theta][solution])
            # print(solution_list)
            all_solutions.append(solution_list)

        return all_solutions

def main():
    ur10_FK = forwardKinematicsSolver()

    # inserting dh parameters: a, alpla, d, theta
    # https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
    gripper_length = 92.2
    ur10_FK.insert_dh_parameters('row1', 0, 90, 127.3, theta1)
    ur10_FK.insert_dh_parameters('row2', -612.7, 0, 0, theta2)  # -pi/2)
    ur10_FK.insert_dh_parameters('row3', -571.6, 0, 0, theta3)
    ur10_FK.insert_dh_parameters('row4', 0, 90, 163.941, theta4)  # -pi/2)
    ur10_FK.insert_dh_parameters('row5', 0, -90, 115.7, theta5)
    ur10_FK.insert_dh_parameters('row6', 0, 0, 92.2+gripper_length, theta6)
    ur10_FK.compute_transforms()

    ur10_FK.insert_geometric_configs(
        'pickupBoxPose', ur10_FK.z_to_script([0.0, -0.0, -0.16, 1.87-0.16, -1.57, -0.0]))
    ur10_FK.insert_geometric_configs(
        'pickedBoxPose', ur10_FK.z_to_script([0.0, 0.7, -0.3, 1.17, -1.57, -0.0]))
    ur10_FK.insert_geometric_configs(
        'placingBoxPose', ur10_FK.z_to_script([3.14, 0.7, -0.3, 1.17, -1.57, -0.0]))
    ur10_FK.insert_geometric_configs(
        'placeBoxPose', ur10_FK.z_to_script([3.14, -0.0, -0.3, 1.87, -1.57, -0.0]))
    ur10_FK.insert_geometric_configs(
        'homePosition', [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    ur10_FK.insert_geometric_configs('uprightPosition', ur10_FK.z_to_script(
        [0.0, 1.57, 0.0, 1.57, 0.0, 0.0]))

    print("The Final Transformation Matrix for the End Effector:")
    print("Simplifying...")
    # pprint(simplify(ur10_FK.final_transform))

    # Use these for FK validation of the 5 poses the robot shifts from in order to perform the pick and place
    # ur10_FK.compute_FK("homePosition")
    # ur10_FK.compute_FK('uprightPosition')
    # ur10_FK.compute_FK('pickupBoxPose')
    # ur10_FK.compute_FK('pickedBoxPose')
    # ur10_FK.compute_FK('placingBoxPose')
    # ur10_FK.compute_FK('placeBoxPose')

    # Use these when the ros2 warehouse publisher is running to publish these values to the /joint_states and see the UR10 live simulation in gazebo
    # ur10.write_to_csv(
    #     ur10.z_to_gazebo(list(ur10.configs["homePosition"].values())))
    # ur10.write_to_csv(
    #      ur10.z_to_gazebo(list(ur10.configs["pickupPosition"].values())))
    # ur10.write_to_csv(
    #      ur10.z_to_gazebo(list(ur10.configs["uprightPosition"].values())))

    # This section verifies the velocity inverse kinematics solution for a circle trajectory
    ur10_IK = inverseKinematicsSolver(ur10_FK.get_transforms())
    ur10_IK.jacobian_from_partials()

    print("Jacobian Matrix for Inverse Kinematics: ")
    print("Simplifying...")
    # pprint(simplify(ur10_IK.J_symbolic))
    # ur10_IK.circle_trajectory_IK()

    # This section verifies the velocity inverse kinematics solution for a line trajectory 
    initial_position = [-1.13,-348.34,1427.30]
    goal_position = [-1288.84,-164.09,-150.53]
    # ur10_IK.line_trajectory_IK(initial_position=initial_position, goal_position=goal_position)

    # ur10_FK.compute_FK('uprightPosition')

    # Add the joint angles in the below to see live update of gazebo simulation 
    ur10_FK.write_to_csv(
        [0.0,1.57,0.0,1.57,0.0,0.0]) # upright pose 
        # [1.57, -0.0, -0.16, 1.87-0.16, -1.57, -0.0]) #loc 1 pickup
        # [1.57, 0.7, -0.3, 1.17, -1.57, -0.0]) # loc 1 picked
        # [0.0, -0.0, -0.16, 1.87-0.16, -1.57, -0.0]) # place pose
        # [3.14, -0.52, -0.3, 1.87+0.52, -1.57, -0.0]) # loc 2 pickup
        # [3.14, 0.785, -0.3, 1.87-0.785, -1.57, -0.0]) # loc 2 picked 


    # # This section was used to get the best solution from the position IK and verification 
    # # It prints out the FK solution for a known configuration and then verifies whether the Ik solver 
    # # can get similar joint angles for the x,y,z roll, pitch, yaw input 
    # verification_angles = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0] # upright position
    # print("\nPosition and Orientation to match")
    # T_evaluated = ur10_FK.compute_FK(verification_angles)
    # T_evaluated = np.array(T_evaluated,dtype=np.float32)

    # all_solutions = ur10_IK.compute_IK(T_evaluated)
    
    # print("\nSolutions from IK:")
    # for solution in all_solutions:
    #     ur10_FK.compute_FK(solution)



if __name__ == "__main__":
    main()
