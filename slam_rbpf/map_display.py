import rclpy
from rclpy.node import Node
from nav_msgs.msg import Path
from std_msgs.msg import UInt8MultiArray
import cv2
from nav_msgs.msg import OccupancyGrid
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class PathPublisher(Node):
    def __init__(self):
        super().__init__('path_publisher_node')
        self.path_subscription = self.create_subscription(
            Image,
            '/slam_map_image',
            self.map_pose_callback,
            10)
        self.bridge = CvBridge()
    
    def map_pose_callback(self, msg):
        # Create a Path message
        self.get_logger().info(f'map_pose_callback, grid_msg received.')
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imshow('Showing Map', cv_image)

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
