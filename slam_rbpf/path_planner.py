import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import Int32MultiArray, Float32
import matplotlib.pyplot as plt
import random

MAX_DISTANCE = 30

class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher_node')
        self.publisher_ = self.create_publisher(Path, '/limo/path', 10)
        self.loop_publisher_ = self.create_publisher(Float32, '/save_map', 10)

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
        self.save_path = '/home/agilex/slam_logs'
        self.timer = self.create_timer(10, self.exploration_detect)

        self.width = 1201
        self.height = 1201
        self.obstacle_pos = []
        self.obstacle_list = []
        self.robotic_pose = []
        self.target_pose = []
        self.path = []
        self.latest_trajectory = []
        self.global_record = None
        self.explored_area = None
        self.obstacle_map = None
        self.epsilon = 0.1  # ε-greedy factor

        self.explored_ratios = []
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], 'b-')
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 1)
        self.ax.set_xlabel('Time (iterations)')
        self.ax.set_ylabel('Exploration Ratio')
        
    def exploration_detect(self):
        if self.global_record is None or self.obstacle_map is None:
            return
        new_explored_area = self.find_min_submatrix_with_nonzeros()
        similar = False
        if self.explored_area is None:
            self.explored_area = new_explored_area
        else:
            similar = self.compare_matrices(new_explored_area)
        if similar:
            explored_ratio = np.count_nonzero(new_explored_area) / new_explored_area.size
            self.explored_ratios.append(explored_ratio)
            self.epsilon = min(1.0, 0.1 + 0.9 * explored_ratio)

            self.get_logger().info(f"exploration_detect = {explored_ratio}, self.epsilon = {self.epsilon}")
            if explored_ratio > 0.1:
                msg = Float32()
                msg.data = explored_ratio
                self.loop_publisher_.publish(msg)
            self.line.set_data(range(len(self.explored_ratios)), self.explored_ratios)
            self.ax.set_xlim(0, max(100, len(self.explored_ratios)))
            file_name = self.save_path + '/exploration_ratio_plot.png'
            plt.savefig(file_name)
            plt.figure(figsize=(10, 10))
            plt.close()
        color_map = plt.cm.get_cmap('Reds')
        color_map.set_under(color='white')
        plt.imshow(self.global_record, cmap=color_map, vmin=0.1)
        plt.colorbar()
        plt.title('Global Record Visualization')
        file_name = self.save_path + '/global_record_visualization.png'
        plt.savefig(file_name)
        plt.close()

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
        pos_data = process_msg.data
        current_pos = pos_data[:2]
        target_pos = pos_data[2:]
        # update global_record for weight.
        # The last pose should be stored in latest_trajectory and the current pose is sent from slam_nav

        if len(current_pos) > 0:
            y1, y2 = current_pos[1], target_pos[1]
            x1, x2 = current_pos[0], target_pos[0]
            if y1 > y2:
                y1, y2 = y2, y1
            if x1 > x2:
                x1, x2 = x2, x1
            self.global_record[y1:y2, x1:x2] += 100
            self.get_logger().info(f"Global record: {self.global_record[y1:y2, x1:x2]},\nGlobal record: scope x is {x1}~{x2}, y is {y1}~{y2}")

        self.get_logger().info(f"map_process_callback: position = {current_pos} -> {target_pos}")
        # if target_pose is already in the past trajectory, then delete the trajectories from that point.
        if target_pos in self.latest_trajectory:
            index = self.latest_trajectory.index(target_pos)
            self.latest_trajectory = self.latest_trajectory[:index]
        self.latest_trajectory.append(target_pos)

    def path_planning(self, x_wall_indices, y_wall_indices, pose_x, pose_y, rotation):
        # Collect obstacles.
        self.get_logger().info(f"START path_planning....")
        combined_indices = np.vstack((x_wall_indices, y_wall_indices)).T
        self.obstacle_pos = np.unique(combined_indices, axis=0)
        if self.obstacle_map is None:
            self.obstacle_map = np.zeros((self.height, self.width), dtype=int)
        for pos in self.obstacle_pos:
            self.obstacle_map[pos[1], pos[0]] = 1
        self.global_record[self.obstacle_map > 0] += self.obstacle_map[self.obstacle_map > 0] * 10
        self.robotic_pose = (pose_x, pose_y, rotation)

        self.get_logger().info(f"posey, posex = {pose_y, pose_x}")
        # Find a proper target.
        target_found, target_pos = self.find_random_free_space()
        if target_found: # If a new target is found
            self.target_pose = target_pos
        elif len(self.latest_trajectory) > 0:
            self.get_logger().info(f"No target pose found but latest_trajectory is not empty. reverse.")
            self.target_pose = self.latest_trajectory[-2]
            self.latest_trajectory = self.latest_trajectory[:-2]
        else:
            self.get_logger().info(f"No target pose found and latest_trajectory is empty. Exploring done.")
            return
        self.path.append(self.target_pose)
        self.publish_target()
        
    def decide_next_pos(self):
        res = True
        pose_to_compare = self.robotic_pose[:2]
        self.get_logger().info(f"pose_to_compare={pose_to_compare}, pose_to_compare len={len(pose_to_compare)}")
        if random.uniform(0, 1) > self.epsilon or len(self.latest_trajectory) > 0:
            self.get_logger().info(f"Go random search")
            res, next_pos = self.find_random_free_space()
        else:
            trajectory_array = np.array(self.latest_trajectory)
            positions = np.where((trajectory_array == pose_to_compare).all(axis=1))[0]
            if len(positions) > 0 and positions[0] + 1 < len(trajectory_array):
                res = True
                next_pos = trajectory_array[positions[0] + 1]
            else:
                res, next_pos = self.find_random_free_space()
        return res, next_pos

    # Find a random position in the nearest local map.
    def find_random_free_space(self):
        start_x, start_y, theta = self.robotic_pose
        total = np.sum(self.global_record)
        self.get_logger().info(f"Total value of self.global_record is {total}, average = {np.mean(self.global_record)}")
        if total == 0:
            base_probabilities = np.ones(self.global_record.shape) / np.product(self.global_record.shape)
        else:
            base_probabilities = (total - self.global_record) / total

        y_coords, x_coords = np.indices(self.global_record.shape)

        distances = np.sqrt((x_coords - start_x)**2 + (y_coords - start_y)**2)
        angles = np.arctan2(y_coords - start_y, x_coords - start_x)
        angle_diffs = np.fmod(angles - theta + 3 * np.pi, 2 * np.pi) - np.pi
        angle_weights = ((np.cos(angle_diffs) + 1) / 2) ** 2
        combined_weights = base_probabilities * angle_weights

        self.get_logger().info(f"theta={theta}, angles={angles.min()}~{angles.max()}, angle_diffs={angle_diffs.min()}~{angle_diffs.max()}")

        # Mask for selecting points within a angle difference of +-pi/3        
        angle_mask = np.abs(angle_diffs) <= np.pi / 3
        # Mask for selecting points within a distance of 100
        distance_mask = distances <= MAX_DISTANCE

        combined_mask = np.logical_and(angle_mask, distance_mask)
        self.get_logger().info(f"combined_mask={combined_mask.shape}")

        valid_indices = np.where(combined_mask.flatten())[0]
        probabilities = combined_weights.flatten()[valid_indices]
        probabilities /= np.sum(probabilities) # Normalize

        dist_valid_y_coords = y_coords[distance_mask]
        dist_valid_x_coords = x_coords[distance_mask]
        angle_valid_y_coords = y_coords[angle_mask]
        angle_valid_x_coords = x_coords[angle_mask]
        self.get_logger().info(f"find_random_free_space SCOPE:    x:{x_coords.min()}~{x_coords.max()}, y:{y_coords.min()}~{y_coords.max()}")
        self.get_logger().info(f"Valid points range for distance: x:{dist_valid_x_coords.min()}~{dist_valid_x_coords.max()}, y:{dist_valid_y_coords.min()}~{dist_valid_y_coords.max()}")
        self.get_logger().info(f"Valid points range for angles:   x:{angle_valid_x_coords.min()}~{angle_valid_x_coords.max()}, y:{angle_valid_y_coords.min()}~{angle_valid_y_coords.max()}")

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

    def find_min_submatrix_with_nonzeros(self):
        non_zero_indices = np.argwhere(self.global_record != 0)
        
        if non_zero_indices.size == 0:
            return np.array([[]]), (0, 0)
        
        min_row, min_col = np.min(non_zero_indices, axis=0)
        max_row, max_col = np.max(non_zero_indices, axis=0)
        
        min_submatrix = self.global_record[min_row:max_row + 1, min_col:max_col + 1]
        
        return min_submatrix

    def compare_matrices(self, new_explored_area, shape_threshold=0.1):

        shape1 = np.array(self.explored_area.shape)
        shape2 = np.array(new_explored_area.shape)
        
        shape_diff = np.abs(shape1 - shape2) / shape1
        max_shape_diff = np.max(shape_diff)
        
        res = max_shape_diff > shape_threshold
        return res


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
