import rclpy
from rclpy.node import Node
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class PathPublisher(Node):
    map_index = 0
    def __init__(self):
        super().__init__('path_publisher_node')
        self.cv_image = None
        self.path_subscription = self.create_subscription(
            Image,
            '/slam_map_image',
            self.map_pose_callback,
            10)
        self.path_subscription = self.create_subscription(
            Image,
            '/save_map',
            self.save_map,
            10)
        self.bridge = CvBridge()
        self.save_path = self.save_path = '/home/agilex/slam_logs/'
    
    def map_pose_callback(self, msg):
        # Create a Path message
        self.get_logger().info(f'map_pose_callback, grid_msg received.')
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow('Showing Map', self.cv_image)

    def save_map(self, msg):
        file_name = self.save_path + str(self.map_index)+'.png'
        self.get_logger().info(f"save_map  path = {self.save_path} ")
        if self.save_map is not None:
            cv2.imwrite(file_name, self.save_map)
        self.map_index += 1

# Main function
def main(args=None):
    rclpy.init(args=args)  # Initialize ROS 2
    ros_node = PathPublisher()  # Create the ROS 2 node
    cv2.namedWindow('Showing Map')

    while rclpy.ok():
        rclpy.spin_once(ros_node, timeout_sec=0)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    ros_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
