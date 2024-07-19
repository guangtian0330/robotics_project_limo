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
        self.explore_route = []
        self.global_map = np.zeros((MAP_SIZE, MAP_SIZE), dtype=bool)
        self.local_map = None
        self.draw_map_pict()
        
    def publish_path(self, path):
        # Create a Path message
        path_msg = Path()
        path_msg.header.frame_id = "path"
        path_msg.header.stamp = self.get_clock().now().to_msg()

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
        theta = euler_angles[2]
        self.get_logger().info(f"map_callback grid map process index={occupied_indices}")
        self.get_logger().info(f"map_callback y_wall_indices={y_wall_indices}, x_wall_indices={x_wall_indices}")
        self.get_logger().info(f"map_callback width = {self.width}, height = {self.height}, origin_x={origin_x}, origin_y={origin_y}, theta={theta}")
        self.update_map(x_wall_indices, y_wall_indices, origin_x, origin_y, theta)

        self.update_map_picture()
        self.update_path_picture()

    def map_process_callback(self, process_msg):
        index, direction = process_msg.data
        col, row = self.path[index]
        theta = direction * np.pi / 2
        self.get_logger().info(f"map_process_callback  col = {col}, row = {row}, theta = {theta}")

        dx = self.relative_obstacle_pos[:, 0] - col
        dy = -(self.relative_obstacle_pos[:, 1] - row)
        obs_angles = np.arctan2(dy, dx) - theta
        obs_angles = (obs_angles + np.pi) % (2 * np.pi) - np.pi
        valid_indices = (-np.pi / 2 <= obs_angles) & (obs_angles <= np.pi / 2) # Filter all obstacles that are in the front
        filtered_obstacle_pos = self.relative_obstacle_pos[valid_indices]
        removed_obstacles = self.relative_obstacle_pos[~valid_indices]         # Filter all the obstacles that are in the back
        self.get_logger().info(f"self.relative_obstacle_pos = {self.relative_obstacle_pos}")
        self.get_logger().info(f"filtered_obstacle_pos = {filtered_obstacle_pos}")
        self.get_logger().info(f"removed_obstacles should be marked as explored : {removed_obstacles}")

        # Mark the explored obstacles as True in the global map.
        for obs_x, obs_y in removed_obstacles:
            if 0 <= obs_y < self.global_map.shape[0] and 0 <= obs_x < self.global_map.shape[1]:
                self.global_map[obs_y, obs_x] = True
        self.relative_obstacle_pos = filtered_obstacle_pos
        self.explore_route.append(self.path[index]) # Add the explored point to the route.

    def draw_map_pict(self):
        self.map_image = np.ones((MAP_SIZE_PICT, MAP_SIZE_PICT, 3), dtype=np.uint8) * 200
        color = (100, 100, 100)
        for x in range(0, self.map_image.shape[1], GRPD_SIZE_PICT):
            cv2.line(self.map_image, (x, 0), (x, self.map_image.shape[0]), color, 1)
        for y in range(0, self.map_image.shape[0], GRPD_SIZE_PICT):
            cv2.line(self.map_image, (0, y), (self.map_image.shape[1], y), color, 1)
        cv2.imshow('Gridworld: Limo with Obstacles', self.map_image)

    # Convert the wall indices and postion of the robotics to a grid map.
    def update_map(self, x_wall_indices, y_wall_indices, pose_x, pose_y, rotation):
        grid_x_indices = x_wall_indices // self.cell_size_x
        grid_y_indices = y_wall_indices // self.cell_size_y
        combined_indices = np.vstack((grid_x_indices, grid_y_indices)).T
        self.obstacle_pos = np.unique(combined_indices, axis=0)

        # Remove all repeated obstacles.
        #current_obstacles_set = set(map(tuple, self.all_obstacle_pos))
        #new_obstacles_set = set(map(tuple, new_obstacles))
        #self.new_obstacle_pos = np.array(list(new_obstacles_set - current_obstacles_set))
        
        self.robotic_pose = [int(pose_x // self.cell_size_x), int(pose_y // self.cell_size_y), rotation]
        self.target_pose, parents, min_x, min_y = self.find_farthest_obstacle(self.robotic_pose)
        if self.target_pose is None:
            self.path.reverse()
            self.publish_path([self.path])     # If there's no way out, then reverse.
            filtered_explore_route = [point for point in self.explore_route if point not in self.path]
            self.explore_route = filtered_explore_route # If reversed, remove the last route.
            return
        self.path = self.build_path(parents, self.robotic_pose, self.target_pose, min_x, min_y )
        self.publish_path(self.path)
        self.get_logger().info(f"Grided coordinates for all obstacles = {self.obstacle_pos}")
        self.get_logger().info(f"Grided coordinates for robotic_pose = {self.robotic_pose}")
        self.get_logger().info(f"Grided coordinates for target_pose = {self.target_pose}")

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
        #add_goal(map_image, goal_position)

    def find_farthest_obstacle(self, start):
        (min_x, min_y)= self.create_minimal_grid_map()
        rows, cols = self.local_map.shape
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
        visited = set()
        queue = deque([(int(start[0]) - min_x, int(start[1]) - min_y, 0)])  # (x, y, distance)
        parents = {}
        max_distance = 0
        farthest_free = None
        self.get_logger().info(f"self.local_map = {self.local_map}, rows={rows}, cols={cols}, start = {start}")
        while queue:
            x, y, dist = queue.popleft()
            for dx, dy in directions:
                nx, ny = x + dx, y + dy
                self.get_logger().info(f"x={x},y={y}, nx={nx},ny={ny}")
                if 0 <= nx < cols and 0 <= ny < rows and (nx, ny) not in visited:
                    self.get_logger().info(f"exploring self.local_map[{ny},{nx}]={self.local_map[ny, nx]}, dist={dist},"
                                           f"and in global map[{y + min_y},{x + min_x}]={self.global_map[y + min_y][x + min_x]}")
                    visited.add((nx, ny))
                    parents[(nx, ny)] = (x, y)
                    if not self.global_map[y + min_y][x + min_x] and self.local_map[ny, nx] != 0:  # Encounter an obstacle
                        self.get_logger().info(f"Encounter an obstacle that has not been visited.")
                        if dist + 1 > max_distance:
                            max_distance = dist + 1
                            farthest_free = (x + min_x, y + min_y)
                    else:
                        queue.append((nx, ny, dist + 1))

        return farthest_free, parents, min_x, min_y

    def build_path(self, parents, start_pos, target_pos, min_x, min_y):
        path = []
        if target_pos:
            step = (target_pos[0] - min_x, target_pos[1] - min_y)
            while step != (start_pos[0] - min_x, start_pos[1] - min_y):
                path.append((step[0] + min_x, step[1] + min_y))
                step = parents[step]
            path.append((start_pos[0], start_pos[1]))  # Add the start point
            path.reverse()  # Reverse the path to start from the original starting point
        self.get_logger().info(f"-----build_path  path = {path}")
        return path

    def transform_coordinates(self, local_x, local_y):
        global_x = local_x + self.min_x
        global_y = local_y + self.min_y
        return global_x, global_y

    def create_minimal_grid_map(self):
        self.get_logger().info(f"-----build_path  obstacle_pos = {self.obstacle_pos}")
        robot_x, robot_y, robot_theta = self.robotic_pose
        self.relative_obstacle_pos = []
        for obs in self.obstacle_pos:
            obs_x, obs_y = obs
            dx = obs_x - robot_x
            dy = -(obs_y - robot_y)
            distance = np.sqrt(dx**2 + dy**2)
            if distance > 8:
                continue
            obs_angle = np.arctan2(dy, dx) - robot_theta
            obs_angle = (obs_angle + np.pi) % (2 * np.pi) - np.pi
            if -np.pi / 2 <= obs_angle <= np.pi / 2:
                self.relative_obstacle_pos.append([obs_x, obs_y])
        self.relative_obstacle_pos = np.array(self.relative_obstacle_pos)

        # Calculate boundaries
        min_x = min(np.min(self.relative_obstacle_pos[:, 0]), self.robotic_pose[0])
        max_x = max(np.max(self.relative_obstacle_pos[:, 0]), self.robotic_pose[0])
        min_y = min(np.min(self.relative_obstacle_pos[:, 1]), self.robotic_pose[1])
        max_y = max(np.max(self.relative_obstacle_pos[:, 1]), self.robotic_pose[1])

        # Create the map
        width = int(max_x - min_x + 1)
        height = int(max_y - min_y + 1)
        self.local_map = np.zeros((height, width), dtype=int)
        
        # add obstacles
        for x, y in self.relative_obstacle_pos:
            grid_x = x - min_x
            grid_y = y - min_y
            self.local_map[grid_y, grid_x] = 1

        return (min_x, min_y)

"""
def mouse_click(event, x, y, flags, param):
    global robot_position, goal_position
    if event == cv2.EVENT_LBUTTONDOWN:
        col, row = x // GRID_SIZE, y // GRID_SIZE
        if (row, col) not in obstacle_positions:
            goal_position = (row, col)
            update_map(param)
    if event == cv2.EVENT_RBUTTONDOWN:
        col, row = x // GRID_SIZE, y // GRID_SIZE
        if (row, col) not in obstacle_positions:

            robot_position = (row, col)
            goal_position = (row, col)
            update_map(param)
"""

# Main function
def main(args=None):
    rclpy.init(args=args)  # Initialize ROS 2
    ros_node = PathPublisher()  # Create the ROS 2 node
    while rclpy.ok():
        rclpy.spin_once(ros_node, timeout_sec=0)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    ros_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()