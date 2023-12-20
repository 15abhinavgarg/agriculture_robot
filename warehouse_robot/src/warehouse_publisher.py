#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Quaternion
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState, Imu
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from scipy.spatial.transform import Rotation
import math
import numpy as np


class IMUSubscriberNode(Node):

    def __init__(self):
        super().__init__('imu_subscriber_node')

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=10
        )

        self.subscription = self.create_subscription(Imu,'/imu_plugin/out',self.imu_callback,qos_profile)

        self.joint_position_pub = self.create_publisher(Float64MultiArray, '/position_controller/commands', 10)
        self.wheel_velocities_pub = self.create_publisher(Float64MultiArray, '/velocity_controller/commands', 10)

        self.prev_time_x = 0
        self.prev_time_y = 0
        
        # Global Destination
        self.finalDestination_x = 10
        self.finalDestination_y = 10
        self.finalOrientation = math.atan(self.finalDestination_x/self.finalDestination_y)

        # Linear x
        self.prev_x = 0
        self.velocity_x = 0

        # Linear y
        self.prev_y = 0
        self.velocity_y = 0

    def imu_callback(self, data):
        # self.get_logger().info("Subscribed from \"/imu_plugin/out\" topic")
        
        joint_positions = Float64MultiArray()
        wheel_velocities = Float64MultiArray()

        orientation_q = data.orientation
        orientation_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        # self.get_logger().info(str(orientation_list))
        yaw = self.yaw_angle_calculation(orientation_list)
        # print yaw
        # yaw = (180/np.pi)*yaw
        # self.get_logger().info(str(yaw))   
        linear_acceleration = data.linear_acceleration
        time = data.header.stamp.sec + data.header.stamp.nanosec * (10**-9)
        # secs_only = secs_nsecs[0] + secs_nsecs[1]* (10**-9)
        x_curr = self.current_x(linear_acceleration,time)
        
        x_cordinate = x_curr*np.cos(yaw)
        y_cordinate = x_curr*np.sin(yaw)
        self.get_logger().info("Subscribed from \"/imu_plugin/out\" topic "+"Computed Robot Coordinates: "+str(x_cordinate)+" , "+str(y_cordinate))

        

        delta_x = x_cordinate - self.finalDestination_x
        delta_y = y_cordinate - self.finalDestination_y
        delta_theta = self.finalOrientation - yaw
        display_delta_theta = delta_theta * (180/np.pi)
        # self.get_logger().info(str(display_delta_theta))

        if abs( delta_x ) < 0.5 and abs( delta_y ) < 0.5:
            self.get_logger().warning("REACHED DESTINATION RADIUS")
            # wheel_velocities.data = []
            lin_vel = 0.0
            steer_angle = 0.0
            wheel_velocities.data = [-lin_vel,lin_vel,-lin_vel,lin_vel]
        
            joint_positions.data = [steer_angle,steer_angle]

            self.joint_position_pub.publish(joint_positions)
            self.wheel_velocities_pub.publish(wheel_velocities)

            # joint_positions.data = [steer_angle,steer_angle]
            # self.joint_position_pub.publish([0.0,0.0])
            # self.wheel_velocities_pub.publish([0.0,0.0])
            self.destroy_node()
            return

        lin_vel = 2*(( delta_x**2 + delta_y**2)**0.5)

        if lin_vel < 5.0:
            lin_vel = 5.0

        steer_angle = 2*(delta_theta/np.pi)

        self.get_logger().info(str(lin_vel)+" "+str(steer_angle))

        wheel_velocities.data = [-lin_vel,lin_vel,-lin_vel,lin_vel]
        
        joint_positions.data = [steer_angle,steer_angle]
        self.get_logger().info("Publishing Control output steering: "+str(steer_angle)+ " Linear Velocity: "+str(lin_vel))
        self.joint_position_pub.publish(joint_positions)
        self.wheel_velocities_pub.publish(wheel_velocities)

        # y_curr = self.current_y(linear_acceleration,time)
        # print(data)
        

    def current_x(self,acceleration,time):
        # self.get_logger().info(str(time))
        
        delta_time = time - self.prev_time_x
        val_x = acceleration.x
        val_x = round(val_x,4)
        if -0.0008 < val_x < 0.0008:
            val_x = 0

        val_x += 0.0073
        self.velocity_x = val_x * delta_time + self.velocity_x   
        self.velocity_x = round(self.velocity_x,2)
        if -0.0002 < self.velocity_x < 0.0002:
            self.velocity_x = 0

        self.prev_x = self.velocity_x * delta_time + self.prev_x    
        # self.get_logger().warning(str(self.prev_x))
        self.prev_time_x = time
        return self.prev_x
    
    def yaw_angle_calculation(self, quaternion):
        x = float(quaternion[0])
        y = float(quaternion[1])
        z = float(quaternion[2])
        w = float(quaternion[3])
        
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = np.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)
        # self.get_logger().info(str(roll))
        # self.get_logger().info(str(pitch))
        # self.get_logger().info(str(yaw))
        return yaw


def main(args=None):
    rclpy.init(args=args)
    node = IMUSubscriberNode()

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
