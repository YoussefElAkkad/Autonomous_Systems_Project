import math
from typing import NamedTuple

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
from ackermann_msgs.msg import AckermannDriveStamped
from geometry_msgs.msg import Point, Pose, Quaternion, Twist, TwistWithCovariance, Vector3
from nav_msgs.msg import Odometry
from scipy.spatial.transform import Rotation

import numpy as np

class AckermannState(NamedTuple):
    position: np.array
    orientation: Rotation
    left_wheel_speed: float
    right_wheel_speed: float
    steering_angle: float
    time: rclpy.time.Time


class OdomPublisher(Node):
    """Listens to odometry information and publishes Odometry message at regular rate"""

    def __init__(self):
        super().__init__('odom_publisher')

        # Parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('axle_length', None),
                ('wheelbase_length', None),
                ('wheel_radius', None),
                ('center_of_mass_offset', 0.0),
                ('damping_factor', 1)
            ]
        )
        self.get_logger().info("Initialized OdomPublisher Node")

        self.axle_length = self.get_parameter('axle_length').value
        self.wheelbase_length = self.get_parameter('wheelbase_length').value
        self.wheel_radius = self.get_parameter('wheel_radius').value
        self.center_of_mass_offset = self.get_parameter('center_of_mass_offset').value
        self.damping_factor = self.get_parameter('damping_factor').value

        # Publishers
        queue_size = 10
        self.publisher = self.create_publisher(Odometry, 'odom', queue_size)

        # Subscribers
        self.create_subscription(
            AckermannDriveStamped,
            'feedback',
            self.feedback_callback,
            10
        )

        self.state = AckermannState(
            position=np.array([0,0,0]),
            orientation=Rotation([0, 0, 0, 1]), # identity
            left_wheel_speed=0.0,
            right_wheel_speed=0.0,
            steering_angle=0.0,
            time=self.get_clock().now()
        )

    def state_update(self, state: AckermannState, feedback: AckermannDriveStamped) -> AckermannState:
        """Calculate the next state based off the current state"""
        average_wheel_speed = (state.left_wheel_speed + state.right_wheel_speed) / 2
        linear_speed = average_wheel_speed * self.wheel_radius
        turn_radius = self.turn_radius(state.steering_angle)
        angular_speed = linear_speed / turn_radius # This is zero if turn_radius is infinite

        feedback_time = self.get_clock().now()
        time_delta = (feedback_time - state.time).nanoseconds * 1e-9

        heading_delta = angular_speed * time_delta # This is zero if angular_speed is zero

        orientation_delta = Rotation.from_euler('xyz', [0, 0, heading_delta])
        if math.isfinite(turn_radius):
            lateral_delta = turn_radius * (1 - math.cos(heading_delta))
            forward_delta = turn_radius * math.sin(heading_delta)
            relative_delta = np.array([forward_delta, lateral_delta, 0])
            position_delta = state.orientation.apply(relative_delta)
        else:
            position_delta = time_delta * self.linear_velocity(state.orientation, linear_speed)

        return AckermannState(
            position=state.position + self.damping_factor * position_delta,
            orientation=orientation_delta * state.orientation,
            left_wheel_speed=feedback.drive.speed,
            right_wheel_speed=feedback.drive.speed,
            steering_angle=feedback.drive.steering_angle,
            time=feedback_time
        )

    def output(self, state: AckermannState) -> Odometry:
        """Build Odometry message from state"""
        quaternion = state.orientation.as_quat()
        linear_speed = self.wheel_radius * (state.left_wheel_speed + state.right_wheel_speed) / 2
        linear_velocity = self.linear_velocity(state.orientation, linear_speed)
        angular_speed = linear_speed / self.turn_radius(state.steering_angle)

        return Odometry(
            header=Header(
                stamp=self.get_clock().now().to_msg(),
                frame_id='odom'
            ),
            pose=Pose(
                position=Point(
                    x=state.position[0],
                    y=state.position[1],
              
