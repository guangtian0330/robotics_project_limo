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


# Constants
MAP_SIZE = 1200  # The original map size
GRID_SIZE = int(MAP_SIZE / 150)  # The original map should be separated in to grids of size 60x60.

MAP_SIZE_PICT = 600
GRPD_SIZE_PICT = int(MAP_SIZE_PICT / 150)

OBSTACLE_SIZE = (GRID_SIZE, GRID_SIZE)  # in cm
GRID_ROWS = int(MAP_SIZE / GRID_SIZE)
GRID_COLS = int(MAP_SIZE / GRID_SIZE)


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
        self.bridge = CvBridge()
        self.width = 1201
        self.height = 1201
        self.min_x = 0
        self.min_y = 0
        self.cell_size_x = GRID_SIZE
        self.cell_size_y = GRID_SIZE
        self.map_image = None
        self.obstacle_pos = []
        self.obstacle_list = []
        self.robotic_pose = []
        self.target_pose = []
        self.path = []
        self.latest_trajectory = []
        self.explore_route = []
        self.global_record = np.zeros((MAP_SIZE, MAP_SIZE), dtype=int)
        self.local_map = None
        self.draw_map_pict()
        
    def publish_path(self, path):
        # Create a Path message
        path_msg = Path()
        path_msg.header.frame_id = "path"
        path_msg.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().info(f"publish_path: path = {path}, len(self.path) = {len(path)}")
        # Convert the path to poses
        for position in path:
            pose = PoseStamped()
            pose.header.frame_id = "path"
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = float(position[1])
            pose.pose.position.y = float(position[0])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0  # No rotation, facing straight up
            path_msg.poses.append(pose)

        # Publish the path
        self.publisher_.publish(path_msg)
        self.get_logger().info('Published new path')

    def map_callback(self, msg):
        data = msg.data
        data_array = np.array(data)
        self.width = msg.info.width
        self.height = msg.info.height
        occupied_indices = np.where(data_array == 100)[0]
        y_wall_indices = occupied_indices // self.width
        x_wall_indices = occupied_indices % self.width
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y
        orientation = msg.info.origin.orientation
        euler_angles = R.from_quat([orientation.x, orientation.y, orientation.z, orientation.w]).as_euler('xyz', degrees=False)
        theta = round(-euler_angles[2]/(np.pi/2)) * (np.pi/2) 
        self.update_map(x_wall_indices, y_wall_indices, origin_x, origin_y, theta)

        self.update_map_picture()
        self.update_path_picture()

    def map_process_callback(self, process_msg):
        index, res = process_msg.data
        self.get_logger().info(f"map_process_callback: index = {index}, self.path = {(self.path)}, res={res}")
        x, y = self.path[index]
        self.global_record[y, x] = 50
        self.update_path_process((x, y))

    # Convert the wall indices and postion of the robotics to a grid map.
    def update_map(self, x_wall_indices, y_wall_indices, pose_x, pose_y, rotation):
        grid_x_indices = x_wall_indices // self.cell_size_x
        grid_y_indices = y_wall_indices // self.cell_size_y
        combined_indices = np.vstack((grid_x_indices, grid_y_indices)).T
        self.obstacle_pos = np.unique(combined_indices, axis=0)
        #self.get_logger().info(f"update_map. grid_x_indices = {grid_x_indices}, grid_y_indices = {grid_y_indices},"
        #                       f"combined_indices = {combined_indices}, self.obstacle_pos = {self.obstacle_pos}")

        self.robotic_pose = [int(pose_x // self.cell_size_x), int(pose_y // self.cell_size_y), rotation]
        average_score = self.create_minimal_grid_map()

        new_place, parents = self.find_random_free_space(self.robotic_pose)
        if new_place is None:
            self.get_logger().info(f"No target pose found and explore_rout is empty. Exploration done.")
            return
        self.target_pose = new_place
        self.path = self.build_path(parents, self.robotic_pose, self.target_pose)
        self.publish_path(self.path)
        # Exploration is stopped and the path has all been explored.
        # Mark this area as explored in the global map.
        #for map_entry in self.local_maps:
        #self.get_logger().info(f"map_process_callback: min_x, min_y = {min_x, min_y}, local_map = {local_map}")
        rows, cols = self.local_map.shape
        start_x, start_y = self.min_x, self.min_y
        end_x, end_y = start_x + cols, start_y + rows
        self.global_record[start_y:end_y, start_x:end_x] += 1

    def draw_map_pict(self):
        self.map_image = np.ones((MAP_SIZE_PICT, MAP_SIZE_PICT, 3), dtype=np.uint8) * 200
        color = (100, 100, 100)
        for x in range(0, self.map_image.shape[1], GRPD_SIZE_PICT):
            cv2.line(self.map_image, (x, 0), (x, self.map_image.shape[0]), color, 1)
        for y in range(0, self.map_image.shape[0], GRPD_SIZE_PICT):
            cv2.line(self.map_image, (0, y), (self.map_image.shape[1], y), color, 1)
        cv2.imshow('Gridworld: Limo with Obstacles', self.map_image)

    def update_path_process(self, pos):
        x, y  = pos
        center_x = int((x + 0.5) * GRPD_SIZE_PICT)
        center_y = int((y + 0.5) * GRPD_SIZE_PICT)
        radius = int(GRPD_SIZE_PICT / 3)
        cv2.circle(self.map_image, (center_x, center_y), radius, (0, 200, 200), -1)
        cv2.imshow('Gridworld: Limo with Obstacles', self.map_image)

    def update_path_picture(self):
        for pos in self.path:
            col, row = pos
            center_x = int((col + 0.5) * GRPD_SIZE_PICT)
            center_y = int((row + 0.5) * GRPD_SIZE_PICT)
            radius = int(GRPD_SIZE_PICT / 3)
            cv2.circle(self.map_image, (center_x, center_y), radius, (0, 0, 0), -1)
        cv2.imshow('Gridworld: Limo with Obstacles', self.map_image)

    def update_map_picture(self):
        if self.map_image is None:
            self.map_image = np.ones((MAP_SIZE_PICT, MAP_SIZE_PICT, 3), dtype=np.uint8) * 200
        for obstacle in self.obstacle_pos:
            col, row = obstacle
            x_start = col * GRPD_SIZE_PICT
            y_start = row * GRPD_SIZE_PICT
            x_end = x_start + GRPD_SIZE_PICT
            y_end = y_start + GRPD_SIZE_PICT
            cv2.rectangle(self.map_image, (x_start, y_start), (x_end, y_end), (0, 0, 200), -1)

        col, row, theta = self.robotic_pose
        center_x = int((col + 0.5) * GRPD_SIZE_PICT)
        center_y = int((row + 0.5) * GRPD_SIZE_PICT)
        radius = int(GRPD_SIZE_PICT / 3)
        cv2.circle(self.map_image, (center_x, center_y), radius, (0, 0, 0), -1)

        target_center_x = int((self.target_pose[0] + 0.5) * GRPD_SIZE_PICT)
        target_center_y = int((self.target_pose[1] + 0.5) * GRPD_SIZE_PICT)
        cv2.circle(self.map_image, (target_center_x, target_center_y), radius, (0, 200, 0), -1)
        cv2.imshow('Gridworld: Limo with Obstacles', self.map_image)

    def find_random_free_space(self, start):
        min_x = self.min_x
        min_y = self.min_y
        rows, cols = self.local_map.shape
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        visited = set()
        queue = deque([(int(start[0]) - min_x, int(start[1]) - min_y, 0)])
        free_spaces = []
        parents = {}
        explore_score = {}
        chosen_space = None
        while queue:
            x, y, dist = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                if 0 <= nx < cols and 0 <= ny < rows and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    parents[(nx, ny)] = (x, y)
                    global_coord = (nx + min_x, ny + min_y)
                    # If this place has been visited before, get its explore record.
                    # Calculate the score for each point, and record it in explore_score
                    if (x + min_x, y + min_y) in explore_score:
                        explore_score[global_coord] = explore_score[(x + min_x, y + min_y)] + self.global_record[ny + min_y, nx + min_x]
                    else:
                        explore_score[global_coord] = self.global_record[ny + min_y, nx + min_x]
                    if self.local_map[ny, nx] == 0: # current place is not occupied by obstacles
                        #if self.global_record[ny + min_y, nx + min_x] >= 50:  # current place has been explored
                        #    explored_free_spaces.append(global_coord)
                        #else:
                        free_spaces.append(global_coord)
                        queue.append((nx, ny, dist + 1))

        if free_spaces:
            # The higher the explore_score is, the lower possibility it gets chosen.
            weights = [1 / (explore_score[space] + 1) for space in free_spaces]
            chosen_space = random.choices(free_spaces, weights=weights, k=1)[0]
            self.get_logger().info(f"Chosen random free space:{chosen_space}")
        #elif explored_free_spaces:
        #    chosen_explored_space = random.choice(explored_free_spaces)
        #    self.get_logger().info(f"No free space available: {chosen_explored_space}")
        return chosen_space, parents


    def build_path(self, parents, start_pos, target_pos):
        path = []
        min_x = self.min_x
        min_y = self.min_y
        if target_pos:
            step = (target_pos[0] - min_x, target_pos[1] - min_y)
            while step != (start_pos[0] - min_x, start_pos[1] - min_y):
                path.append((step[0] + min_x, step[1] + min_y))
                step = parents[step]
            path.append((start_pos[0], start_pos[1]))  # Add the start point
            path.reverse()  # Reverse the path to start from the original starting point
        self.get_logger().info(f"-----build_path  path = {path}")
        return path

    # Create a minimum local map that involves all relavant obstacles.
    def create_minimal_grid_map(self):
        self.get_logger().info(f"-----build_path  obstacle_pos = {self.obstacle_pos}")
        self.get_logger().info("No relevant obstacles. Need to redraw the map.")
        self.min_x = max(self.robotic_pose[0] - 5, 0)
        self.min_y = max(self.robotic_pose[1] - 5, 0)
        self.max_x = min(self.robotic_pose[0] + 5, MAP_SIZE)
        self.max_y = min(self.robotic_pose[1] + 5, MAP_SIZE)

        # Create the map
        height = self.max_y - self.min_y + 1
        width = self.max_x - self.min_x + 1
        self.local_map = np.zeros((int(self.max_y - self.min_y + 1), int(self.max_x - self.min_x + 1)), dtype=int)
        
        # add obstacles
        for x, y in self.obstacle_pos:
            grid_x = x - self.min_x
            grid_y = y - self.min_y
            if 0 <= grid_x < width and 0 <= grid_y < height:
                self.local_map[grid_y, grid_x] = 1
        self.local_map[self.robotic_pose[1] - self.min_y, self.robotic_pose[0] - self.min_x] = 2
        
        rows, cols = self.local_map.shape
        start_x, start_y = self.min_x, self.min_y
        end_x, end_y = start_x + cols, start_y + rows
        selected_area = self.global_record[start_y:end_y, start_x:end_x]
        average_value = np.mean(selected_area)

        self.get_logger().info(f"-----build_path created local_map = {self.local_map}, selected_area = {selected_area}, average_explore score = {average_value}")
        return average_value

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