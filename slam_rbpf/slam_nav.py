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
from .rbpf_SLAM import SLAM
from geometry_msgs.msg import Twist
import numpy as np
import math
from . import transformations as tf
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import UInt8MultiArray
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from std_msgs.msg import Header
import time
from cv_bridge import CvBridge

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
        self.lidar_data[np.isinf(self.lidar_data)] = scan_data.range_max
        angle_min = scan_data.angle_min
        angle_max = scan_data.angle_max
        self.angle_increment = scan_data.angle_increment
        self.lidar_angles_ = np.arange(angle_min, angle_max, self.angle_increment)
        self.lidar_max_ = 8
        self.time = None

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
    
    def _detect_front_obstacles(self, angle_range=50, distance_threshold=1.0):
        angle_range_rad = np.radians(angle_range)
        start_index = max(int(((-angle_range_rad - self.lidar_angles_[0]) / self.angle_increment)), 0)
        end_index = min(int(((angle_range_rad - self.lidar_angles_[0]) / self.angle_increment)), len(self.lidar_angles_) - 1)
        print(f"_detect_front_obstacles -- Front angle scope is: lidar_angles_[{start_index}]={self.lidar_angles_[start_index]} ~ lidar_angles_[{end_index}]={self.lidar_angles_[end_index]}")
        front_distances = self.lidar_data[slice(start_index, end_index + 1)]
        obstacles = front_distances < distance_threshold
        if np.any(obstacles):
            min_distance = np.min(front_distances[obstacles])
            return True, min_distance
        return False, None

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
        self.bridge = CvBridge()
        self.map_publisher = self.create_publisher(Image, '/slam_map_image', 10)
        self.vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.start_odom = None
        self.lidar_data = None
        self.angle_to_rotate = np.pi / 2  # 30 degrees
        self.distance_to_move = 0.44
        self.status = STATUS_TYPE_STAY
        self.turn_threshold = 0.107 # Turn threshold in radians
        self.move_threshold = 0.01
        self.slam_map = SLAM(mov_cov) # initiate a SLAM.
        self.is_initialized = False
        self.timer = self.create_timer(1, self.process_data)
        self.timer = self.create_timer(15, self.initialtive_action)
        self.move_count = 0


    def lidar_callback(self, msg):
        #self.get_logger().info('Received lidar_callback data')
        self.lidar_data = LidarData(msg)
        self.slam_map.add_lidar_data(self.lidar_data)
        

    def camera_callback(self, msg):
        # Camera data might be used or displayed, but not stored as per current requirement
        self.get_logger().info('Received camera data')

    # callback function for odometers. the moved distance and turned angles could be calculated with collected data from
    # the odometer.
    def odom_callback(self, msg):
        #self.get_logger().info('Received odom_callback data')
        odom_data = OdomData(msg)
        #self.get_logger().info(f"--------odom_data:-----x={odom_data.x}, y={odom_data.y}, theta={odom_data.theta}-------------")
        self.slam_map.add_odo_data(odom_data)
        if self.start_odom is None:
            self.start_odom = odom_data
        if self.status == STATUS_TYPE_FORWARD:
            moved_distance = odom_data.get_distance(self.start_odom)
            if ((moved_distance >= self.distance_to_move)
                    or (abs(moved_distance - self.distance_to_move) <= self.move_threshold)):
                self.get_logger().info(f"----moved_distance = {moved_distance}")
                self.status = STATUS_TYPE_STAY
                self.stop_moving()
        if self.status == STATUS_TYPE_ROTATE_LEFT or STATUS_TYPE_ROTATE_RIGHT:
            rotated_angle = odom_data.theta - self.start_odom.theta
            rotated_angle = (rotated_angle + np.pi) % (2 * np.pi) - np.pi
            if (rotated_angle >= self.angle_to_rotate
                    or (abs(rotated_angle - self.angle_to_rotate) <= self.turn_threshold)):
                self.get_logger().info(f"----turned angles =  {odom_data.theta}-{self.start_odom.theta}={rotated_angle}")
                self.status = STATUS_TYPE_STAY
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
        if not self.slam_map.check_lidar_valid() or len(self.slam_map.odom_data_list) == 0:
            return
        if not self.is_initialized:
            self.slam_map._init_map_for_particles()
            self.is_initialized = True
        else:
            start_time = time.time()
            self.slam_map._run_slam()
            # Publish the map
            image_msg = self.bridge.cv2_to_imgmsg(self.slam_map.map_img, encoding='bgr8')
            self.map_publisher.publish(image_msg)
            elapsed_time = time.time() - start_time
            self.get_logger().info(f'-----map_publisher has published the grid_msg---- {elapsed_time:.2f} seconds')
            self.slam_map.clear_data()

    def initialtive_action(self):
        self.get_logger().info('-----initialtive_action-----')
        if self.lidar_data is None:
            return
        obstacle, distance = self.lidar_data._detect_front_obstacles()
        self.get_logger().info(f'-----_detect_front_obstacles returned obstacle = {obstacle} in the distance of {distance}-----')
        if obstacle is False and self.move_count < 4:
            self.go_straight(0.2)
            self.move_count += 1
        else :
            self.rotate(0.2)
            self.move_count = 0

def main(args=None):
    rclpy.init(args=args)
    slam_navigation_node = SLAMNavigationNode()
    rclpy.spin(slam_navigation_node)
    slam_navigation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()