# -*- coding: utf-8 -*-

############################################
#       University of Laurentian
#     Authors: Guangtian GONG
#   Rao-Blackwellized Paricle Filter SLAM
#             RBPF SLAM Class
############################################

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion
from .utils import time_index_0
from .rbpf_SLAM import SLAM
from geometry_msgs.msg import Twist
import numpy as np
import math
from . import transformations as tf
from nav_msgs.msg import OccupancyGrid
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy

STATUS_TYPE_STAY = "stay"
STATUS_TYPE_FORWARD = "forward"
STATUS_TYPE_ROTATE_LEFT = "rotate_left"
STATUS_TYPE_ROTATE_RIGHT = "rotate_right"

mov_cov = np.array([[1e-8, 0, 0],
                    [0, 1e-8, 0],
                    [0, 0 , 1e-8]])

class LidarData:
    def __init__(self, scan_data):
        self.scan_time = scan_data.header.stamp.nanosec / 1e9
        self.lidar_data =  np.array(scan_data.ranges)        
        angle_min = scan_data.angle_min
        angle_max = scan_data.angle_max
        angle_increment = scan_data.angle_increment
        self.lidar_angles_ = np.arange(angle_min, angle_max, angle_increment)
        self.lidar_max_ = scan_data.range_max

    def _polar_to_cartesian(self, scan, pose = None):
        '''
            Converts polar scan to cartisian x,y coordinates
        '''
        scan[scan > self.lidar_max_] = self.lidar_max_ 
        lidar_ptx = scan * np.cos(self.lidar_angles_)
        lidar_pty = scan * np.sin(self.lidar_angles_)
        if pose is not None:
            T = tf.twoDTransformation(pose[0], pose[1], pose[2])
            pts = np.vstack((lidar_ptx, lidar_pty, np.ones(lidar_ptx.shape)))
            trans_pts = T@pts
            lidar_ptx, lidar_pty, _ = trans_pts
        return lidar_ptx, lidar_pty

class OdomData:
    def __init__(self, odom_data):
        self.odom_time = odom_data.header.stamp.nanosec / 1e9
        self.quaternion = odom_data.pose.pose.orientation
        self.x = odom_data.pose.pose.position.x
        self.y = odom_data.pose.pose.position.y
        self.theta = 0
        self.init_theta()
    
    def init_theta(self):
        x, y, z, w = self.quaternion.x, self.quaternion.y, self.quaternion.z, self.quaternion.w
        """
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        self.roll = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        self.pitch = math.asin(t2)
        """
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        self.theta = math.atan2(t3, t4)

    def get_distance(self, start_odo):
        return ((start_odo.x - self.x) ** 2 + (start_odo.y - self.y) ** 2) ** 0.5


class SLAMNavigationNode(Node):
    def __init__(self):
        super().__init__('slam_rbpf_nav')
        qos_profile = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        # Subscribe lidar data.
        self.lidar_subscription = self.create_subscription(
            LaserScan,
            '/scan',
            self.lidar_callback,
            qos_profile)
        # Subscribe camera data
        self.camera_subscription = self.create_subscription(
            Image,
            '/camera',
            self.camera_callback,
            10)
        # Subscribe odometer data
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/odometry',
            self.odom_callback,
            10)
        self.map_publisher = self.create_publisher(OccupancyGrid, '/map', 10)
        self.vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.start_odom = None
        self.angle_to_rotate = np.pi / 6  # 30 degrees
        self.distance_to_move = 0.44
        self.status = STATUS_TYPE_STAY
        self.turn_threshold = 0.107 # Turn threshold in radians
        self.move_threshold = 0.01
        self.slam_map = SLAM(mov_cov) # initiate a SLAM.
        self.time_index = time_index_0
        self.timer = self.create_timer(1.0, self.process_data)

    def lidar_callback(self, msg):
        self.get_logger().info('Received lidar_callback data')
        self.slam_map.add_lidar_data(LidarData(msg))

    def camera_callback(self, msg):
        # Camera data might be used or displayed, but not stored as per current requirement
        self.get_logger().info('Received camera data')

    # callback function for odometers. the moved distance and turned angles could be calculated with collected data from
    # the odometer.
    def odom_callback(self, msg):
        self.get_logger().info('Received odom_callback data')
        odom_data = OdomData(msg)
        self.slam_map.add_odo_data(odom_data)
        if self.start_odom is None:
            self.start_odom = odom_data
        if self.status == STATUS_TYPE_FORWARD:
            moved_distance = odom_data.get_distance(self.start_odom)
            if ((moved_distance >= self.distance_to_move)
                    or (abs(moved_distance - self.distance_to_move) <= self.move_threshold)):
                self.get_logger().info(f"----moved_distance = {moved_distance}")
                self.stop_moving()
        if self.status == STATUS_TYPE_ROTATE_LEFT or STATUS_TYPE_ROTATE_RIGHT:
            rotated_angle = abs(odom_data.theta - self.start_odom.theta)
            if (rotated_angle >= self.angle_to_rotate
                    or (abs(rotated_angle - self.angle_to_rotate) <= self.turn_threshold)):
                self.get_logger().info(f"----turned angles = {rotated_angle}")
                self.stop_moving()

    def rotate(self, angular_speed):
        self.start_odom = None
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        if angular_speed > 0:
            self.status = STATUS_TYPE_ROTATE_LEFT
            self.get_logger().info("-------------turn left-------------")
        else:
            self.status = STATUS_TYPE_ROTATE_RIGHT
            self.get_logger().info("-------------turn right-------------")
        twist_msg.angular.z = angular_speed
        self.vel_publisher.publish(twist_msg)

    def go_straight(self, speed):
        self.get_logger().info("-------------go_straight-------------")
        self.start_odom = None
        self.status = STATUS_TYPE_FORWARD
        twist_msg = Twist()
        twist_msg.linear.x = speed
        twist_msg.angular.z = 0.0
        self.vel_publisher.publish(twist_msg)

    def stop_moving(self):
        self.get_logger().info('-----stop_moving-----')
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.vel_publisher.publish(twist_msg)
        self.status = STATUS_TYPE_STAY

    def process_data(self):
        data_size = self.slam_map.get_shorter_length()
        if data_size == 0:
            return
        if self.time_index == time_index_0:
            self.slam_map._init_map_for_particles()
            index_start = 1
        else:
            index_start = 0
        grid_msg = self.slam_map._run_slam(index_start, data_size - 1)

        # Publish the map
        self.map_publisher.publish(grid_msg)

        self.time_index += 1
        self.slam_map.clear_data()

def main(args=None):
    rclpy.init(args=args)
    slam_navigation_node = SLAMNavigationNode()
    rclpy.spin(slam_navigation_node)
    slam_navigation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()