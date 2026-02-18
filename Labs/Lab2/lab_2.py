import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
import numpy as np
import math
import pygame
import pickle
import time

# Set up the sound
pygame.mixer.init()
sound = pygame.mixer.Sound('./car_crash.mp3')

# Set up the logging
logging_names_list = ['time_stamp','theta1_f', 'theta2_f', 'theta3_f', 'theta1_b', 'theta2_b', 'theta3_b', 'end_effector_position_f', 'end_effector_position_b']

class ForwardKinematics(Node):

    def __init__(self):
        super().__init__("forward_kinematics")
        self.joint_subscription = self.create_subscription(JointState, "joint_states", self.listener_callback, 10)
        self.joint_subscription  # prevent unused variable warning

        self.position_publisher = self.create_publisher(Float64MultiArray, "leg_front_l_end_effector_position", 10)
        self.marker_publisher = self.create_publisher(Marker, "marker", 10)

        self.joint_positions = None
        timer_period = 0.02  # publish FK information and marker at 50Hz
        self.timer = self.create_timer(timer_period, self.timer_callback)

        self.kp_publisher = self.create_publisher(Float64MultiArray, "/forward_kp_controller/commands", 10)
        self.kd_publisher = self.create_publisher(Float64MultiArray, "/forward_kd_controller/commands", 10)

        # Open file for writing
        self.filename = "lab_2_data.pkl"
        self.data_dictionary = {}
        for name in logging_names_list:
            self.data_dictionary[name] = []
        self.start_time = time.time()

        # Periodically set gains to 0 so legs go limp
        self.create_timer(0.1, self.publish_zero_gains)

    def log_data(self, time_stamp, theta1_f, theta2_f, theta3_f, theta1_b, theta2_b, theta3_b, end_effector_position_f, end_effector_position_b):
        self.data_dictionary['time_stamp'].append(time_stamp)
        self.data_dictionary['theta1_f'].append(theta1_f)
        self.data_dictionary['theta2_f'].append(theta2_f)
        self.data_dictionary['theta3_f'].append(theta3_f)
        self.data_dictionary['theta1_b'].append(theta1_b)
        self.data_dictionary['theta2_b'].append(theta2_b)
        self.data_dictionary['theta3_b'].append(theta3_b)
        self.data_dictionary['end_effector_position_f'].append(end_effector_position_f)
        self.data_dictionary['end_effector_position_b'].append(end_effector_position_b)
        with open(self.filename, 'wb') as file_handle:
            pickle.dump(self.data_dictionary, file_handle)

    def publish_zero_gains(self):
        self.kp_publisher.publish(Float64MultiArray(data=[0.0] * 12))
        self.kd_publisher.publish(Float64MultiArray(data=[0.0] * 12))

    def listener_callback(self, msg):
        # Extract the positions of the joints related to leg_front_l
        joints_of_interest = ["leg_front_l_1", "leg_front_l_2", "leg_front_l_3","leg_back_l_1", "leg_back_l_2", "leg_back_l_3"]
        self.joint_positions = [msg.position[msg.name.index(joint)] for joint in joints_of_interest]

    

    def rotation_x(self, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        ## TODO: Implement the rotation matrix about the x-axis
        return np.array(
            [
                [1, 0, 0, 0],
                [0, c, -s, 0],
                [0, s, c, 0],
                [0, 0, 0, 1],
            ]
        )

    def rotation_y(self, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        ## TODO: Implement the rotation matrix about the y-axis
        return np.array(
            [
                [c, 0, s, 0],
                [0, 1, 0, 0],
                [-s, 0, c, 0],
                [0, 0, 0, 1],
            ]
        )

    def rotation_z(self, angle):
        ## TODO: Implement the rotation matrix about the z-axis
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array(
            [
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )

    def translation(self, x, y, z):
        ## TODO: Implement the translation matrix
        return np.array(
            [
                [1, 0, 0, x],
                [0, 1, 0, y],
                [0, 0, 1, z],
                [0, 0, 0, 1],
            ]
        )

    # FK for forward left leg
    def forward_kinematics_f(self, theta1, theta2, theta3):

        T_0_1 = self.translation(0.08, 0.0, 0.0) @ self.rotation_y(theta1)

        # --- T_1_2 (leg_front_l_1 -> leg_front_l_2) ---
        T_1_2 = self.translation(0, 0.04, 0.0) @ self.rotation_x(theta2)

        # --- T_2_3 (leg_front_l_2 -> leg_front_l_3) ---
        T_2_3 = self.translation(-0.07, 0, -0.06) @ self.rotation_y(theta3)

        # --- T_3_ee (leg_front_l_3 -> end effector) ---
        T_3_ee = self.translation(0.072, 0.053, 0.0)

        # --- Chain them: T_0_ee ---
        T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee

        # --- End-effector position (xyz in base_link) ---
        end_effector_position = T_0_ee[0:3, 3]

        return end_effector_position



    # FK for back left leg
    def forward_kinematics_b(self, theta1, theta2, theta3):

        T_0_1 = self.translation(-0.08, 0.0, 0.0) @ self.rotation_y(theta1)
        T_1_2 = self.translation(0, 0.04, 0.0) @ self.rotation_x(theta2)
        T_2_3 = self.translation(-0.07, 0, -0.06) @ self.rotation_y(theta3)
        T_3_ee = self.translation(0.072, 0.053, 0.0)

        # --- Chain them: T_0_ee ---
        T_0_ee = T_0_1 @ T_1_2 @ T_2_3 @ T_3_ee

        # --- End-effector position (xyz in base_link) ---
        end_effector_position = T_0_ee[0:3, 3]

        return end_effector_position


    def timer_callback(self):
        """Timer callback for publishing end-effector marker and position."""
        if self.joint_positions is None:
            return
        theta1_f_raw = self.joint_positions[0]
        theta2_f_raw = self.joint_positions[1]
        theta3_f_raw = self.joint_positions[2]
        theta1_b_raw = self.joint_positions[3]
        theta2_b_raw = self.joint_positions[4]
        theta3_b_raw = self.joint_positions[5]

        self.get_logger().info(
            f"RAW front: [{theta1_f_raw:.3f}, {theta2_f_raw:.3f}, {theta3_f_raw:.3f}] "
            f"back: [{theta1_b_raw:.3f}, {theta2_b_raw:.3f}, {theta3_b_raw:.3f}]"
        )

        if self.joint_positions is not None:
            # Joint angles
            theta1_f = self.joint_positions[0] - 0.016 
            theta2_f = -self.joint_positions[1] + 0.011
            theta3_f = self.joint_positions[2] + 0.708
            theta1_b = self.joint_positions[3] + 0.169
            theta2_b = -self.joint_positions[4] + 0.006
            theta3_b = self.joint_positions[5] + 0.554
            end_effector_position_f = self.forward_kinematics_f(theta1_f, theta2_f, theta3_f)
            end_effector_position_b = self.forward_kinematics_b(theta1_b, theta2_b, theta3_b)
            
            # --- Collision check: distance between front-left and back-left feet ---
            diff = end_effector_position_f - end_effector_position_b
            dist = float(np.linalg.norm(diff))

            THRESH = 0.06  # 6 cm
            if not hasattr(self, "last_crash_time"):
                self.last_crash_time = 0.0

            now = time.time()
            if dist < THRESH and (now - self.last_crash_time) > 0.01:  # 0.8s cooldown between crashes
                sound.play()
                self.last_crash_time = now
                self.get_logger().warn(f"CRASH! dist={dist:.3f} m")

            time_stamp = time.time() - self.start_time
            self.log_data(time_stamp, theta1_f, theta2_f, theta3_f, theta1_b, theta2_b, theta3_b, end_effector_position_f, end_effector_position_b)
            
            marker = Marker()
            marker.header.frame_id = "/base_link"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.type = marker.SPHERE
            marker.id = 0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.pose.position.x = end_effector_position_f[0]
            marker.pose.position.y = end_effector_position_f[1]
            marker.pose.position.z = end_effector_position_f[2]
            self.marker_publisher.publish(marker)

            position = Float64MultiArray()
            position.data = end_effector_position_f
            self.position_publisher.publish(position)
            self.get_logger().info(
                f"End-Effector Position: x={end_effector_position_f[0]:.2f}, y={end_effector_position_f[1]:.2f}, z={end_effector_position_f[2]:.2f}"
            )


def main(args=None):
    rclpy.init(args=args)
    forward_kinematics = ForwardKinematics()
    rclpy.spin(forward_kinematics)


if __name__ == "__main__":
    main()
