#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Quaternion
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from scipy.spatial.transform import Rotation
import math
import csv
import numpy as np

import sys
from std_srvs.srv import SetBool

class TrajecotryPublisherNode(Node):

    def __init__(self):
        super().__init__('trajectory_publisher_node')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription = self.create_subscription(JointState, '/joint_states', self.get_joint_states_callback,qos_profile)
        self.joint_position_pub = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)

        self.cli = self.create_client(SetBool, '/warehouse_gripper/custom_switch')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')
        self.req = SetBool.Request()

        timer_period = 0.05  # in seconds
        self.timer = self.create_timer(timer_period, self.joint_positions_publisher)
        
        self.current_joint_1 = None
        self.current_joint_2 = None
        self.current_joint_3 = None
        self.current_joint_4 = None
        self.current_joint_5 = None
        self.current_joint_6 = None

        self.joint_1 = None
        self.joint_2 = None
        self.joint_3 = None
        self.joint_4 = None
        self.joint_5 = None
        self.joint_6 = None

        self.number_of_messages = 157 
        self.i = 0
        self.j = 0
        self.k = 0
        self.joint_value_counter = 0

        self.pickup = 0
        self.pickedup = 0
        self.placing = 0
        self.place = 0

        self.phase1 = True
        self.phase2 = True
        self.phase3 = True
        self.phase4 = True
        self.phase5 = True
        self.phase6 = True
        self.phase7 = True
        self.joint_value_counter = 0

        # joint positions in sequence calculated using position inverse kinematics
        self.start =   [0.0,1.57,0.0,1.57,0.0,0.0]
        self.loc1_pickup = [1.57, -0.0, -0.16, 1.87-0.16, -1.57, -0.0]
        self.loc1_picked = [1.57, 0.7, -0.3, 1.17, -1.57, -0.0]
        self.place =  [0.0, -0.0, -0.16, 1.87-0.16, -1.57, -0.0]
        self.loc2_pickup = [3.14, -0.52, -0.3, 1.87+0.52, -1.57, -0.0]
        self.loc2_picked = [3.14, 0.785, -0.3, 1.87-0.785, -1.57, -0.0]

        self.joint_1_phase1 = np.linspace(self.start[0], self.loc1_pickup[0], 100)
        self.joint_2_phase1 = np.linspace(self.start[1], self.loc1_pickup[1], 100)
        self.joint_3_phase1 = np.linspace(self.start[2], self.loc1_pickup[2], 100)
        self.joint_4_phase1 = np.linspace(self.start[3], self.loc1_pickup[3], 100)
        self.joint_5_phase1 = np.linspace(self.start[4], self.loc1_pickup[4], 100)
        self.joint_6_phase1 = np.linspace(self.start[5], self.loc1_pickup[5], 100)
        
        self.joint_1_phase2 = np.linspace(self.loc1_pickup[0], self.loc1_picked[0], 100)
        self.joint_2_phase2 = np.linspace(self.loc1_pickup[1], self.loc1_picked[1], 100)
        self.joint_3_phase2 = np.linspace(self.loc1_pickup[2], self.loc1_picked[2], 100)
        self.joint_4_phase2 = np.linspace(self.loc1_pickup[3], self.loc1_picked[3], 100)
        self.joint_5_phase2 = np.linspace(self.loc1_pickup[4], self.loc1_picked[4], 100)
        self.joint_6_phase2 = np.linspace(self.loc1_pickup[5], self.loc1_picked[5], 100)
        
        self.joint_1_phase3 = np.linspace(self.loc1_picked[0], self.place[0], 100)
        self.joint_2_phase3 = np.linspace(self.loc1_picked[1], self.place[1], 100)
        self.joint_3_phase3 = np.linspace(self.loc1_picked[2], self.place[2], 100)
        self.joint_4_phase3 = np.linspace(self.loc1_picked[3], self.place[3], 100)
        self.joint_5_phase3 = np.linspace(self.loc1_picked[4], self.place[4], 100)
        self.joint_6_phase3 = np.linspace(self.loc1_picked[5], self.place[5], 100)
        
        self.joint_1_phase4 = np.linspace(self.place[0], self.loc2_pickup[0], 100)
        self.joint_2_phase4 = np.linspace(self.place[1], self.loc2_pickup[1], 100)
        self.joint_3_phase4 = np.linspace(self.place[2], self.loc2_pickup[2], 100)
        self.joint_4_phase4 = np.linspace(self.place[3], self.loc2_pickup[3], 100)
        self.joint_5_phase4 = np.linspace(self.place[4], self.loc2_pickup[4], 100)
        self.joint_6_phase4 = np.linspace(self.place[5], self.loc2_pickup[5], 100)

        self.joint_1_phase5 = np.linspace(self.loc2_pickup[0], self.loc2_picked[0], 100)
        self.joint_2_phase5 = np.linspace(self.loc2_pickup[1], self.loc2_picked[1], 100)
        self.joint_3_phase5 = np.linspace(self.loc2_pickup[2], self.loc2_picked[2], 100)
        self.joint_4_phase5 = np.linspace(self.loc2_pickup[3], self.loc2_picked[3], 100)
        self.joint_5_phase5 = np.linspace(self.loc2_pickup[4], self.loc2_picked[4], 100)
        self.joint_6_phase5 = np.linspace(self.loc2_pickup[5], self.loc2_picked[5], 100)
        
        self.joint_1_phase6 = np.linspace(self.loc2_picked[0], self.place[0], 100)
        self.joint_2_phase6 = np.linspace(self.loc2_picked[1], self.place[1], 100)
        self.joint_3_phase6 = np.linspace(self.loc2_picked[2], self.place[2], 100)
        self.joint_4_phase6 = np.linspace(self.loc2_picked[3], self.place[3], 100)
        self.joint_5_phase6 = np.linspace(self.loc2_picked[4], self.place[4], 100)
        self.joint_6_phase6 = np.linspace(self.loc2_picked[5], self.place[5], 100)

        self.joint_1_phase7 = np.linspace(self.place[0], self.start[0], 100)
        self.joint_2_phase7 = np.linspace(self.place[1], self.start[1], 100)
        self.joint_3_phase7 = np.linspace(self.place[2], self.start[2], 100)
        self.joint_4_phase7 = np.linspace(self.place[3], self.start[3], 100)
        self.joint_5_phase7 = np.linspace(self.place[4], self.start[4], 100)
        self.joint_6_phase7 = np.linspace(self.place[5], self.start[5], 100)

    def send_request(self, status):
        self.req.data = status
        self.future = self.cli.call_async(self.req)
        # rclpy.spin_until_future_complete(self, self.future)
        return self.future.result()

    def get_joint_from_csv(self,path):
        with open(path, 'r') as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader) 
            joint_data = np.array(list(csv_reader), dtype=float)

        self.joint_1 = joint_data[:,0]
        self.joint_2 = joint_data[:,1]
        self.joint_3 = joint_data[:,2]
        self.joint_4 = joint_data[:,3]
        self.joint_5 = joint_data[:,4]
        self.joint_6 = joint_data[:,5]

        self.joint_values_len = len(self.joint_1)

        self.joints_computed = False

    def get_joint_states_callback(self, data):
        joint_positions = Float64MultiArray()

        joints = data.position
        self.current_joint_1 = joints[0]
        self.current_joint_2 = joints[1]
        self.current_joint_3 = joints[2]
        self.current_joint_4 = joints[3]
        self.current_joint_5 = joints[4]
        self.current_joint_6 = joints[5]
        # self.get_logger().info("Subscribed to \"/joint_states\""+"Current theta1: "+str(self.current_joint_1))

    def joint_positions_publisher(self):
        joint_positions = Float64MultiArray()

        if self.phase1 and self.joint_value_counter<100:
            joint_positions.data = [self.joint_1_phase1[self.joint_value_counter], self.joint_2_phase1[self.joint_value_counter], self.joint_3_phase1[self.joint_value_counter], self.joint_4_phase1[self.joint_value_counter], self.joint_5_phase1[self.joint_value_counter], self.joint_6_phase1[self.joint_value_counter]]
            self.get_logger().info('Now publishing phase1: "%s"' % joint_positions.data)
            self.joint_position_pub.publish(joint_positions)
            self.joint_value_counter = self.joint_value_counter + 1 
            if self.joint_value_counter == 99:
                self.joint_value_counter = 0
                self.phase1 = False
                self.get_logger().info('Bool phase1: "%s"' % self.phase1)
                response = self.send_request(True)
                self.get_logger().info('Server response after phase1: "%s"' % str(response))
        
        elif self.phase2 and self.joint_value_counter<100:
            joint_positions.data = [self.joint_1_phase2[self.joint_value_counter], self.joint_2_phase2[self.joint_value_counter], self.joint_3_phase2[self.joint_value_counter], self.joint_4_phase2[self.joint_value_counter], self.joint_5_phase2[self.joint_value_counter], self.joint_6_phase2[self.joint_value_counter]]
            self.get_logger().info('Now publishing phase2: "%s"' % joint_positions.data)
            self.joint_position_pub.publish(joint_positions)
            self.joint_value_counter = self.joint_value_counter + 1 
            if self.joint_value_counter == 99:
                self.joint_value_counter = 0
                self.phase2 = False
         
        elif self.phase3 and self.joint_value_counter<100:
            joint_positions.data = [self.joint_1_phase3[self.joint_value_counter], self.joint_2_phase3[self.joint_value_counter], self.joint_3_phase3[self.joint_value_counter], self.joint_4_phase3[self.joint_value_counter], self.joint_5_phase3[self.joint_value_counter], self.joint_6_phase3[self.joint_value_counter]]
            self.get_logger().info('Now publishing phase3: "%s"' % joint_positions.data)
            self.joint_position_pub.publish(joint_positions)
            self.joint_value_counter = self.joint_value_counter + 1 
            if self.joint_value_counter == 99:
                self.joint_value_counter = 0
                self.phase3 = False
                response = self.send_request(False)
                self.get_logger().info('Server response after phase4: "%s"' % response)

        elif self.phase4 and self.joint_value_counter<100:
            joint_positions.data = [self.joint_1_phase4[self.joint_value_counter], self.joint_2_phase4[self.joint_value_counter], self.joint_3_phase4[self.joint_value_counter], self.joint_4_phase4[self.joint_value_counter], self.joint_5_phase4[self.joint_value_counter], self.joint_6_phase4[self.joint_value_counter]]
            self.get_logger().info('Now publishing phase4: "%s"' % joint_positions.data)
            self.joint_position_pub.publish(joint_positions)
            self.joint_value_counter = self.joint_value_counter + 1 
            if self.joint_value_counter == 99:
                self.joint_value_counter = 0
                self.phase4 = False
                response = self.send_request(True)
                self.get_logger().info('Server response after phase4: "%s"' % response)

        elif self.phase5 and self.joint_value_counter<100:
            joint_positions.data = [self.joint_1_phase5[self.joint_value_counter], self.joint_2_phase5[self.joint_value_counter], self.joint_3_phase5[self.joint_value_counter], self.joint_4_phase5[self.joint_value_counter], self.joint_5_phase5[self.joint_value_counter], self.joint_6_phase5[self.joint_value_counter]]
            self.get_logger().info('Now publishing phase5: "%s"' % joint_positions.data)
            self.joint_position_pub.publish(joint_positions)
            self.joint_value_counter = self.joint_value_counter + 1 
            if self.joint_value_counter == 99:
                self.joint_value_counter = 0
                self.phase5 = False

        elif self.phase6 and self.joint_value_counter<100:
            joint_positions.data = [self.joint_1_phase6[self.joint_value_counter], self.joint_2_phase6[self.joint_value_counter], self.joint_3_phase6[self.joint_value_counter], self.joint_4_phase6[self.joint_value_counter], self.joint_5_phase6[self.joint_value_counter], self.joint_6_phase6[self.joint_value_counter]]
            self.get_logger().info('Now publishing phase6: "%s"' % joint_positions.data)
            self.joint_position_pub.publish(joint_positions)
            self.joint_value_counter = self.joint_value_counter + 1 
            if self.joint_value_counter == 99:
                self.joint_value_counter = 0
                self.phase6 = False
                response = self.send_request(False)
                self.get_logger().info('Server response after phase6: "%s"' % response)

        elif self.phase7 and self.joint_value_counter<100:
            joint_positions.data = [self.joint_1_phase7[self.joint_value_counter], self.joint_2_phase7[self.joint_value_counter], self.joint_3_phase7[self.joint_value_counter], self.joint_4_phase7[self.joint_value_counter], self.joint_5_phase7[self.joint_value_counter], self.joint_6_phase7[self.joint_value_counter]]
            self.get_logger().info('Now publishing phase7: "%s"' % joint_positions.data)
            self.joint_position_pub.publish(joint_positions)
            self.joint_value_counter = self.joint_value_counter + 1 
            if self.joint_value_counter == 99:
                self.joint_value_counter = 0
                self.phase7 = False
       
        # go to joint angles read from live csv file 
        else:
            self.get_joint_from_csv("/home/fierra/ros2_uws/src/agriculture_robot/src/live_data.csv")
            joint_positions.data = [self.joint_1[0], self.joint_2[0], self.joint_3[0], self.joint_4[0], self.joint_5[0], self.joint_6[0]]
            self.get_logger().info('Now publishing from live csv: "%s"' % joint_positions.data)
            self.joint_position_pub.publish(joint_positions)

def main(args=None):
    rclpy.init(args=args)
    node = TrajecotryPublisherNode()

    rclpy.spin(node)
    node.joint_positions_publisher()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
