#!/usr/bin/env python3

import roslib
import rospy
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import Float64
import sys, select, termios, tty
import _thread
from numpy import clip

control_keys = {
    'up'    : '\x41',
    'down'  : '\x42',
    'right' : '\x43',
    'left'  : '\x44',
    'space' : '\x20',
    'tab'   : '\x09'}

key_bindings = {
    '\x41' : ( 1.0 , 0.0),
    '\x42' : (-1.0 , 0.0),
    '\x43' : ( 0.0 ,-1.0),
    '\x44' : ( 0.0 , 1.0),
    '\x20' : ( 0.0 , 0.0),
    '\x09' : ( 0.0 , 0.0)}


class TeleopAckermann:

    def __init__(self, args):
        max_speed = 0.26
        max_steering_angle = 0.78

        self.speed_range = [-float(max_speed), float(max_speed)]
        self.steering_angle_range = [-float(max_steering_angle), float(max_steering_angle)]
        for key in key_bindings:
            key_bindings[key] = (key_bindings[key][0] * 0.01,
                     key_bindings[key][1] * 0.03)

        self.speed = 0
        self.steering_angle = 0
        self.pub = rospy.Publisher('/ackermann_cmd', AckermannDrive, queue_size=10)
        rospy.Timer(rospy.Duration(1.0/10.0), self.pub_callback)
        self.print_state()
        self.key_loop()

    def pub_callback(self, event):
        ackermann_cmd_msg = AckermannDrive()
        ackermann_cmd_msg.speed = self.speed
        ackermann_cmd_msg.steering_angle = self.steering_angle
        self.pub.publish(ackermann_cmd_msg)

    def print_state(self):
        sys.stderr.write('\x1b[2J\x1b[H')
        rospy.loginfo('\x1b[1M\r*********************************************')
        rospy.loginfo('\x1b[1M\rUse arrows to change speed and steering angle')
        rospy.loginfo('\x1b[1M\rUse space to brake and tab to align wheels')
        rospy.loginfo('\x1b[1M\rPress <ctrl-c> or <q> to exit')
        rospy.loginfo('\x1b[1M\r*********************************************')

    def get_key(self):
        tty.setraw(sys.stdin.fileno())
        select.select([sys.stdin], [], [], 0)
        key = sys.stdin.read(1)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key


    def key_loop(self):
        self.settings = termios.tcgetattr(sys.stdin)
        while 1:
            key = self.get_key()
            if key in key_bindings.keys():
                if key == control_keys['space']:
                    self.speed = 0.0
                elif key == control_keys['tab']:
                    self.steering_angle = 0.0
                else:
                    self.speed = self.speed + key_bindings[key][0]
                    self.steering_angle = self.steering_angle + key_bindings[key][1]
                    self.speed = clip(self.speed, self.speed_range[0], self.speed_range[1])
                    self.steering_angle = clip(
                        self.steering_angle,
                        self.steering_angle_range[0],
                        self.steering_angle_range[1])
            elif key == '\x03' or key == '\x71':  # ctr-c or q
                break
            else:
                continue
            rospy.loginfo('\x1b[1M\r'
                      '\033[34;1mSpeed: \033[32;1m%0.2f m/s, '
                      '\033[34;1mSteer Angle: \033[32;1m%0.2f rad\033[0m',
                      self.speed, self.steering_angle)
        self.finalize()

    def finalize(self):
        self.settings = termios.tcgetattr(sys.stdin)
        ackermann_cmd_msg = AckermannDrive()
        ackermann_cmd_msg.speed = 0
        ackermann_cmd_msg.steering_angle = 0
        self.pub.publish(ackermann_cmd_msg)
        rospy.loginfo('Teleop node has stopped')
        sys.exit()

if __name__ == '__main__':
    rospy.init_node('teleop_ackermann')
    keyop = TeleopAckermann(sys.argv[1:len(sys.argv)])