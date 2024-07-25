# -*- coding: utf-8 -*-

############################################
#       University of Laurentian
#     Authors: Guangtian GONG
#   Rao-Blackwellized Paricle Filter SLAM
#             RBPF SLAM Class
############################################

import rclpy
from scipy.spatial.transform import Rotation as R
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Point, Quaternion
from .rbpf_SLAM import SLAM
from .rbpf_particle import Particle
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
import cv2
from geometry_msgs.msg import Pose, Point
from nav_msgs.msg import Path
from std_msgs.msg import Int32MultiArray


STATUS_TYPE_STAY = "stay"
STATUS_TYPE_FORWARD = "forward"
STATUS_TYPE_ROTATE = "rotating"
MAP_SIZE = 1200  # The original map size
GRID_SIZE = int(MAP_SIZE / 150)  # The original map should be separated in to grids of size 60x60.

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
        valid_indices = (self.lidar_angles_ >= -np.pi/2) & (self.lidar_angles_ <= np.pi/2)
        self.lidar_angles_ = self.lidar_angles_[valid_indices]
        self.lidar_data = self.lidar_data[valid_indices]
        self.lidar_max_ = 8
        self.lidar_data_range_min = scan_data.range_min
        self.lidar_data_range_max = scan_data.range_max
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


    def _detect_front_obstacles(self, distance_threshold=0.5, window_angle_rad=np.pi / 4):
        window_size = int(window_angle_rad / self.angle_increment)
        half_window_size = int(window_size / 2)
        center_index = len(self.lidar_data) // 2
        front_data = self.lidar_data[center_index - half_window_size : center_index + half_window_size + 1]

        side_window_width = int(np.pi/6 / self.angle_increment)
        right_index = center_index + half_window_size + side_window_width - 1
        left_data = self.lidar_data[0 : side_window_width + 1]
        right_data = self.lidar_data[right_index : len(self.lidar_angles_) - 1]
        #print(f"_detect_front_obstacles self.lidar_data = {self.lidar_data}")
        #print(f"_detect_front_obstacles angle:from {self.lidar_angles_[center_index - half_window_size]} to {self.lidar_angles_[center_index + half_window_size]}")
        print(f"_detect_front_obstacles left self.angle = {self.lidar_angles_[0 : side_window_width + 1]}")
        print(f"_detect_front_obstacles front self.angle = {self.lidar_angles_[center_index - half_window_size : center_index + half_window_size + 1]}")
        print(f"_detect_front_obstacles right self.angle = {self.lidar_angles_[right_index : len(self.lidar_angles_) - 1]}")
        front_min_distance = np.min(front_data)
        right_min_distance = np.min(left_data)
        left_min_distance = np.min(right_data)
        print(f"_detect_front_obstacles min_distance is {left_min_distance}, self.angle = {self.lidar_angles_[np.argmin(left_data)]}")
        print(f"_detect_front_obstacles min_distance is {front_min_distance}, self.angle = {self.lidar_angles_[np.argmin(front_data) + center_index-half_window_size]}")
        print(f"_detect_front_obstacles min_distance is {right_min_distance}, self.angle = {self.lidar_angles_[np.argmin(right_data) + right_index]}")
        return left_min_distance, front_min_distance, right_min_distance
        #if np.all(front_data > distance_threshold):
        #    return True, front_min_distance
        
        #left_min_distance = np.m
        #return False, front_min_distance
        

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
        self.path_subscription = self.create_subscription(
            Path,
            '/limo/path',
            self.path_callback,
            10)
        # Subscribe odometer data
        self.odom_subscription = self.create_subscription(
            Odometry,
            '/wheel/odom',
            self.odom_callback,
            10)
        self.bridge = CvBridge()
        self.map_pic_publisher = self.create_publisher(Image, '/slam_map_image', 10)
        self.process_publisher_ = self.create_publisher(Int32MultiArray, '/explore/process', 10)
        self.map_publisher = self.create_publisher(OccupancyGrid, 'map', 10)
        #self.vel_publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.vel_publisher = self.create_publisher(Twist, 'cmd_vel_input', 10)
        self.target_reached = True
        self.new_map_sent = False

        self.start_odom = None
        self.lidar_data = None
        self.angle_to_rotate = 0 # 90 degrees
        self.distance_to_move = 0.3
        self.status = STATUS_TYPE_STAY
        self.turn_threshold = 0.107 # Turn threshold in radians
        self.move_threshold = 0.01
        self.slam_map = SLAM(mov_cov) # initiate a SLAM.
        self.is_initialized = False
        self.timer = self.create_timer(0.5, self.process_data)
        self.timer = self.create_timer(3, self.move_path)
        self.move_count = 0
        self.current_pos = None
        self.theta = 0
        self.target_pos = None
        self.update_map = 0


    def lidar_callback(self, msg):
        #self.get_logger().info('Received lidar_callback data')
        self.lidar_data = LidarData(msg)
        self.slam_map.add_lidar_data(self.lidar_data)
        #self.get_logger().info(f"----self.lidar_data = {self.lidar_data.lidar_data}")

    def path_callback(self, path):
        # Camera data might be used or displayed, but not stored as per current requirement
        self.get_logger().info('Received path data')
        self.get_logger().info(f"path_callback self.target_reached={self.target_reached},"
                               f" self.new_map_sent={self.new_map_sent},"
                               f" self.status={self.status}")
        if self.target_reached and self.new_map_sent:
            if path and len(path.poses) > 0:
                pose = path.poses[0].pose
                self.target_pos = (int(pose.position.x), int(pose.position.y))
                self.target_reached = False
                self.get_logger().info(f"Received path data self.target_pos={self.target_pos}")

    # callback function for odometers. the moved distance and turned angles could be calculated with collected data from
    # the odometer.
    def odom_callback(self, msg):
        odom_data = OdomData(msg)
        self.get_logger().info(f"----odom_data = {odom_data.x}, {odom_data.y}, {odom_data.theta/np.pi * 180}")
        self.slam_map.add_odo_data(odom_data)
        self.theta = odom_data.theta
        if self.start_odom is None:
            self.start_odom = odom_data
        if self.status == STATUS_TYPE_FORWARD:
            moved_distance = odom_data.get_distance(self.start_odom)
            if ((moved_distance >= self.distance_to_move)
                    or (abs(moved_distance - self.distance_to_move) <= self.move_threshold)):
                self.get_logger().info(f"----moved_distance = {moved_distance}")
                self.stop_moving()
                self.status = STATUS_TYPE_STAY
        elif self.status == STATUS_TYPE_ROTATE:
            rotated_angle = odom_data.theta - self.start_odom.theta
            rotated_angle = abs((rotated_angle + np.pi) % (2 * np.pi) - np.pi)
            self.get_logger().info(f"----rotated_angle = {rotated_angle}, angle_to_rotate = {self.angle_to_rotate}")
            if (rotated_angle >= self.angle_to_rotate
                    or (abs(rotated_angle - self.angle_to_rotate) <= self.turn_threshold)):
                self.get_logger().info(f"----turned angles =  {odom_data.theta}-{self.start_odom.theta}={rotated_angle/np.pi * 180}")
                self.stop_moving()
                self.status = STATUS_TYPE_STAY

    def move_path(self):
        if self.status != STATUS_TYPE_STAY or self.target_reached:
            return
        if self.target_pos is None:
            return

        self.get_logger().info(f"position transition: "
                               f"({self.current_pos[0]}, f{self.current_pos[1]})"
                               f"->({self.target_pos[0]}, f{self.target_pos[1]})")
        left_obstacle, front_obstacle, right_obstacle = self.lidar_data._detect_front_obstacles()
        angle_to_turn = self.get_angle(self.current_pos, self.target_pos, self.theta)
        angle_to_rotate = abs(angle_to_turn)
        self.get_logger().info(f"angle_to_turn = {angle_to_turn}, left_obstacle={left_obstacle}, front_obstacle={front_obstacle}, right_obstacle={right_obstacle}")
        msg = Int32MultiArray()
        if (angle_to_rotate > self.turn_threshold):
            direction = angle_to_turn / angle_to_rotate
            self.rotate(angle_to_rotate, direction)
        else :
            distance_to_move = self.get_distance(self.current_pos, self.target_pos)
            safe_distance = front_obstacle * 0.86
            if safe_distance < 0.4:
                # Found an unexpected obstacle. it's either a mistake in path planning or dynamic change
                # either way, this trip should be terminated and wait for another map planning.
                self.get_logger().info(f"Obstacle found, this trip is interrupted.")
                if left_obstacle > right_obstacle:
                    self.rotate(np.pi/3, 1)
                else:
                    self.rotate(np.pi/3, -1)
                self.target_reached = True
                self.new_map_sent = False
                #msg.data = self.current_pos
                #self.process_publisher_.publish(msg)
            else:
                if left_obstacle <= 0.35 and right_obstacle > left_obstacle:
                    angular = -0.1 # Too close to the left, turn right a bit.
                elif right_obstacle <= 0.35 and left_obstacle > right_obstacle:
                    angular = 0.1  # Too close to the right, turn left a bit.
                else:
                    angular = 0.0
                distance_to_move = min(distance_to_move, safe_distance - 0.31) # 0.31 is the minimum distance where the obstacle can be detected.
                self.get_logger().info(f"obstacle_distance={front_obstacle}, distance_to_move:{distance_to_move}")
                self.go_straight(distance_to_move, angular)
                #msg.data = self.target_pos
                #self.process_publisher_.publish(msg)
                self.target_reached = True
                self.new_map_sent = False
            # Send current position and res to path planner to update the map record.

    # Take one step ahead along the path
    def get_angle(self, current_pose, target_pose, current_angle_rad):
        # Calculate the angle of the target point relative to the current position

        delta_x = target_pose[0] - current_pose[0]
        delta_y = -(target_pose[1]- current_pose[1])
        target_angle = math.atan2(delta_y, delta_x)
        # Calculate the angle required to rotate
        rotation_angle = target_angle - current_angle_rad
        # Make sure the rotation angle is between -π and π
        rotation_angle = (rotation_angle + np.pi) % (2 * np.pi) - np.pi
        self.get_logger().info(f"delta_y={delta_y}, delta_x={delta_x}, target_angle={target_angle}, current_angle_rad={current_angle_rad}, rotation_angle={rotation_angle}")
        return rotation_angle
    
    def get_distance(self, current_pose, target_pose):
        x1, y1 = current_pose
        x2, y2 = target_pose
        x1_world, y1_world = ((x1 + 1) * 0.05 - 30), ((y1 + 1) * 0.05 - 30)
        x2_world, y2_world = ((x2 + 1) * 0.05 - 30), ((y2 + 1) * 0.05 - 30)
        distance = math.sqrt((x2_world - x1_world) ** 2 + (y2_world - y1_world) ** 2)
        self.get_logger().info(f"current_pose={current_pose}, target_pose={target_pose}")
        self.get_logger().info(f"current_pose_world=[{x1_world},{y1_world}], target_pose_world=[{x2_world},{y2_world}], distance={distance}")
        return distance

    def rotate(self, angle_diff, direction):
        #self.angle_to_rotate = abs(angle_diff)
        self.angle_to_rotate = angle_diff
        angular_speed = direction * 0.5
        self.get_logger().info(f"------Start to rotate-----angle_to_rotate = {self.angle_to_rotate}--speed = {angular_speed}--------")
        self.start_odom = None
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        self.status = STATUS_TYPE_ROTATE
        if angular_speed > 0:
            self.get_logger().info("-------------turn left-------------")
        else:
            self.get_logger().info("-------------turn right-------------")
        twist_msg.angular.z = angular_speed
        self.vel_publisher.publish(twist_msg)

    def go_straight(self, distance, rotate):
        self.get_logger().info("-------------go straight-------------")
        self.distance_to_move = distance
        self.start_odom = None
        self.status = STATUS_TYPE_FORWARD
        twist_msg = Twist()
        twist_msg.linear.x = 0.5
        twist_msg.angular.z = rotate
        self.vel_publisher.publish(twist_msg)

    def stop_moving(self):
        self.get_logger().info("-------------stop moving-------------")
        twist_msg = Twist()
        twist_msg.linear.x = 0.0
        twist_msg.angular.z = 0.0
        self.vel_publisher.publish(twist_msg)
        self.status = STATUS_TYPE_STAY

    def process_data(self):
        if not self.slam_map.check_lidar_valid() or len(self.slam_map.odom_data_list) == 0:
            return
        self.get_logger().info(f"-----------process_data--------self.is_initialized = {self.is_initialized}-----")
        if not self.is_initialized:
            self.slam_map._init_map_for_particles()
            self.is_initialized = True
        else:
            start_time = time.time()
            particle = self.slam_map._run_slam()
            self.gen_map(particle)

            elapsed_time = time.time() - start_time
            self.get_logger().info(f'-----map_publisher has published the grid_msg---- {elapsed_time:.2f} seconds')
            self.slam_map.clear_data()


    def gen_map(self, particle):
        '''
            Generates and publishes the combined map with trajectory to the ROS topic
        '''
        #gen_init_time1 = time.time()
        log_odds = particle.log_odds_
        logodd_thresh = particle.logodd_thresh_
        traj = particle.traj_indices_
        self.get_logger().info(f"THE WHOLE trajectory should be: \n{traj}")
        MAP = particle.MAP_
        # Generate the visualization map
        x_wall_indices, y_wall_indices = np.where(log_odds > logodd_thresh)
        y_wall_indices_conv = MAP['sizey'] - 1 - y_wall_indices

        valid_x = (traj[0] >= 0) & (traj[0] < MAP['sizex'])
        valid_y = (traj[1] >= 0) & (traj[1] < MAP['sizey'])
        valid_indices = valid_x & valid_y   
        #MAP_2_display[abs(log_odds) < 1e-1, :] = [150, 150, 150]
        x_indices = traj[0][valid_indices]
        y_indices = traj[1][valid_indices]
        y_indices_conv = MAP['sizey'] - 1 - y_indices

        MAP_2_display = 255 * np.ones((MAP['sizex'], MAP['sizey'], 3), dtype=np.uint8)
        MAP_2_display[y_wall_indices_conv, x_wall_indices, :] = [0, 0, 0]
        MAP_2_display[y_indices_conv, x_indices] = [70, 70, 228]
        if self.target_pos is not None:
            MAP_2_display[MAP['sizey'] - 1 - int(self.target_pos[1]), int(self.target_pos[0])] = [0, 255, 0]
        map_img = cv2.resize(MAP_2_display, (700, 700))
        image_msg = self.bridge.cv2_to_imgmsg(map_img, encoding='bgr8')
        self.map_pic_publisher.publish(image_msg)
        self.update_map += 1
        if self.target_reached and not self.new_map_sent and  self.update_map > 5:
            update_time = time.time()
            grid = OccupancyGrid()
            grid.header = Header(frame_id="map")
            grid.info.width = MAP['sizex']
            grid.info.height = MAP['sizey']
            last_x = x_indices[-1] if x_indices.size > 0 else 0.0
            last_y = y_indices[-1] if y_indices.size > 0 else 0.0
            last_theta = particle.trajectory_[:, -1]
            rotation = R.from_euler('z', last_theta[2])
            quaternion = rotation.as_quat()
            self.current_pos = (int(last_x), int(last_y))
            grid.info.origin = Pose(position = Point(x = float(last_x), y = float(last_y), z = 0.0))
            grid.info.origin.orientation.x = quaternion[0]
            grid.info.origin.orientation.y = quaternion[1]
            grid.info.origin.orientation.z = quaternion[2]
            grid.info.origin.orientation.w = quaternion[3]
            grid.data = [-1] * (grid.info.width * grid.info.height)
            index = y_wall_indices * grid.info.width + x_wall_indices
            self.get_logger().info(f"grid map process last_x={last_x}, last_y={last_y}, last_theata={last_theta[2]}")
            for idx in index:
                grid.data[idx] = 100
            self.map_publisher.publish(grid)
            update_elapsed = time.time() - update_time
            self.get_logger().info(f"grid map process time consumed{update_elapsed:.5f}")
            self.new_map_sent = True

def main(args=None):
    rclpy.init(args=args)
    slam_navigation_node = SLAMNavigationNode()
    rclpy.spin(slam_navigation_node)
    slam_navigation_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
