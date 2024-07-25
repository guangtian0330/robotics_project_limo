import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

class VelocityPublisher(Node):
    def __init__(self):
        super().__init__('velocity_publisher')
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            Twist,
            'cmd_vel_input',
            self.listener_callback,
            10
        )
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.twist = Twist()

    def listener_callback(self, twist_msg):
        self.twist.linear.x = twist_msg.linear.x
        self.twist.angular.z = twist_msg.angular.z

    def timer_callback(self):
        self.publisher_.publish(self.twist)
        self.get_logger().info('Publishing: "%s"' % self.twist)

def main(args=None):
    rclpy.init(args=args)
    velocity_publisher = VelocityPublisher()
    rclpy.spin(velocity_publisher)
    rclpy.shutdown()

if __name__ == '__main__':
    main()