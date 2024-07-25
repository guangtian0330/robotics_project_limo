import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np
import cv2
import math
import heapq
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import OccupancyGrid
from collections import deque
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Int32MultiArray
import random



class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher_node')
        self.publisher_ = self.create_publisher(Path, '/limo/path', 10)
        self.path_subscription = self.create_subscription(
            Int32MultiArray,
            '/explore/process',
            self.map_process_callback,
            10)
        self.subscription = self.create_subscription(
            OccupancyGrid,
            'map',
            self.map_callback,
            10)
        self.width = 1201
        self.height = 1201
        self.obstacle_pos = []
        self.obstacle_list = []
        self.robotic_pose = []
        self.target_pose = []
        self.path = []
        self.latest_trajectory = []
        self.global_record = None
        self.obstacle_map = None
        
    def publish_target(self):
        # Create a Path message
        path_msg = Path()
        path_msg.header.frame_id = "path"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().info(f"publish_path: path = {self.path}, len(self.path) = {len(self.path)}")
        # Convert the path to poses
        for position in self.path:
            pose = PoseStamped()
            pose.header.frame_id = "path"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(position[0])
            pose.pose.position.y = float(position[1])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # No rotation, facing straight up
            path_msg.poses.append(pose)

        # Publish the path
        self.publisher_.publish(path_msg)
        self.path.clear()
        self.get_logger().info('Published new path')

    # Publish all coordinates of obstacles and current pose.
    def map_callback(self, msg):
        self.get_logger().info('map_callback')
        data = msg.data
        data_array = np.array(data)
        self.width = msg.info.width
        self.height = msg.info.height
        self.get_logger().info(f"map_callback, self.width = {self.width}, self.height = {self.height}")

        if self.global_record is None:
            self.global_record = np.zeros((self.height, self.width), dtype=int)
        occupied_indices = np.where(data_array == 100)[0]
        y_wall_indices = occupied_indices // self.width
        x_wall_indices = occupied_indices % self.width
        origin_x = int(msg.info.origin.position.x)
        origin_y = int(msg.info.origin.position.y)
        orientation = msg.info.origin.orientation
        euler_angles = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w]).as_euler('xyz', degrees=False)
        yaw = euler_angles[2]
        self.path_planning(x_wall_indices, y_wall_indices, origin_x, origin_y, yaw)


    def map_process_callback(self, process_msg):
        pos = process_msg.data
        self.get_logger().info(f"map_process_callback: position = {pos}")
        self.latest_trajectory.append(pos)

    # update the 
    def path_planning(self, x_wall_indices, y_wall_indices, pose_x, pose_y, rotation):
        # Collect obstacles.
        self.get_logger().info(f"START path_planning....")
        combined_indices = np.vstack((x_wall_indices, y_wall_indices)).T
        self.obstacle_pos = np.unique(combined_indices, axis=0)
        if self.obstacle_map is None:
            self.obstacle_map = np.zeros((self.height, self.width), dtype=int)
        for pos in self.obstacle_pos:
            self.obstacle_map[pos[1], pos[0]] = 1
        self.robotic_pose = (pose_x, pose_y, rotation)
        # The last pose should be stored in latest_trajectory and the current pose is sent from slam_nav
        last_pose = []
        if len(self.latest_trajectory) > 0:
            last_pose = self.latest_trajectory[-1]
        if len(last_pose) > 0:
            y1, y2 = last_pose[1], self.robotic_pose[1]
            x1, x2 = last_pose[0], self.robotic_pose[0]
            if y1 > y2:
                y1, y2 = y2, y1
            if x1 > x2:
                x1, x2 = x2, x1
            self.global_record[y1:y2, x1:x2] += 50
            self.get_logger().info(f"Global record: {self.global_record[y1:y2, x1:x2]}")

        self.latest_trajectory.append(self.robotic_pose)
        self.global_record[pos[1], pos[0]] += 100
        self.get_logger().info(f"posey, posex = {pose_y, pose_x}")
        self.obstacle_map[pose_y, pose_x] = 2
        # Find a proper target.
        target_found, target_pos = self.find_random_free_space()
        if target_found: # If a new target is found
         self.target_pose = target_pos
        elif len(self.latest_trajectory) > 0:
            self.get_logger().info(f"No target pose found but latest_trajectory is not empty. reverse.")
            self.target_pose = self.latest_trajectory.pop()
        else:
            self.get_logger().info(f"No target pose found and latest_trajectory is empty. Exploring done.")
            return
        self.path.append(self.target_pose)
        self.publish_target()
        #local_map = self.obstacle_map[self.robotic_pose[1]:self.target_pose[1], self.robotic_pose[0]:self.target_pose[0]]
        #self.get_logger().info(f"Displaying the local map : {local_map}")
        # Exploration is stopped and the path has all been explored.
        # Mark this area as explored in the global map.
        

    # Find a random position in the nearest local map.
    def find_random_free_space(self):
        start_x, start_y, theta = self.robotic_pose
        total = np.sum(self.global_record)
        if total == 0:
            base_probabilities = np.ones(self.global_record.shape) / np.product(self.global_record.shape)
        else:
            base_probabilities = (total - self.global_record) / total

        y_coords, x_coords = np.indices(self.global_record.shape)
        self.get_logger().info(f"find_random_free_space SCOPE: x:{x_coords.min()}~{x_coords.max()}, y:{y_coords.min()}~{y_coords.max()}")

        distances = np.sqrt((x_coords - start_x)**2 + (y_coords - start_y)**2)

        angles = np.arctan2(y_coords - start_y, x_coords - start_x)
        angle_diffs = np.fmod(angles - theta + 3 * np.pi, 2 * np.pi) - np.pi
        angle_weights = ((np.cos(angle_diffs) + 1) / 2) ** 2
        combined_weights = base_probabilities * angle_weights

        self.get_logger().info(f"theta={theta}, angles={angles.min()}~{angles.max()}, angle_diffs={angle_diffs.min()}~{angle_diffs.max()}")
        angle_mask = np.abs(angle_diffs) <= np.pi / 3

        # Mask for selecting points within a distance of 100
        distance_mask = distances <= 30
        combined_mask = np.logical_and(angle_mask, distance_mask)
        self.get_logger().info(f"combined_mask={combined_mask.shape}")

        valid_indices = np.where(distance_mask.flatten())[0]
        probabilities = combined_weights.flatten()[valid_indices]
        probabilities /= np.sum(probabilities) # Normalize

        valid_y_coords = y_coords[combined_mask]
        valid_x_coords = x_coords[combined_mask]

        # 打印符合条件的坐标点范围
        if valid_y_coords.size > 0 and valid_x_coords.size > 0:
            y_min, y_max = valid_y_coords.min(), valid_y_coords.max()
            x_min, x_max = valid_x_coords.min(), valid_x_coords.max()
            self.get_logger().info(f"Valid points range: x:{x_min}~{x_max}, y:{y_min}~{y_max}")
        else:
            self.get_logger().info(f"No valid points found within the specified range")


        max_attempts = 10000
        attempts = 0
        while attempts < max_attempts:
            if probabilities.sum() == 0:
                break
            idx = np.random.choice(valid_indices, p=probabilities)
            end_y, end_x = np.unravel_index(idx, self.global_record.shape)
            if self.is_path_clear(start_x, start_y, end_x, end_y):
                return True, (end_x, end_y)
            attempts += 1
        self.get_logger().info(f"Tried everything but failed")
        return False, (start_x, start_y)

    # Bresenham's Line Algorithm
    # Return all points on the line between (x0, y0) and (x1, y1)
    def bresenham(self, x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1
        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy
        points.append((x, y))
        return points

    # Check if the way between the target and the start is clear.
    def is_path_clear(self, start_x, start_y, end_x, end_y):
        for x, y in self.bresenham(start_x, start_y, end_x, end_y):
            if self.obstacle_map[y][x] == 1:  # Assuming 1 is the obstacle
                return False
        return True


# Main function
def main(args=None):
    rclpy.init(args=args)       # Initialize ROS 2
    ros_node = PathPublisher()  # Create the ROS 2 node
    while rclpy.ok():
        rclpy.spin_once(ros_node, timeout_sec=0)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
