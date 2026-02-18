#!/usr/bin/env python3
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

# ----------------------------
# Sound + Logging setup
# ----------------------------
pygame.mixer.init()
sound = pygame.mixer.Sound("./car_crash.mp3")

logging_names_list = [
    "time_stamp",
    "theta1_f", "theta2_f", "theta3_f",
    "theta1_b", "theta2_b", "theta3_b",
    "end_effector_position_f", "end_effector_position_b",
]


class ForwardKinematics(Node):
    def __init__(self):
        super().__init__("forward_kinematics")

        self.joint_subscription = self.create_subscription(
            JointState, "joint_states", self.listener_callback, 10
        )

        self.position_publisher = self.create_publisher(
            Float64MultiArray, "leg_front_l_end_effector_position", 10
        )
        self.marker_publisher = self.create_publisher(Marker, "marker", 10)

        self.kp_publisher = self.create_publisher(Float64MultiArray, "/forward_kp_controller/commands", 10)
        self.kd_publisher = self.create_publisher(Float64MultiArray, "/forward_kd_controller/commands", 10)

        self.joint_positions = None
        self.timer = self.create_timer(0.02, self.timer_callback)  # 50 Hz
        self.create_timer(0.1, self.publish_zero_gains)

        # Logging to pickle
        self.filename = "lab_2_data.pkl"
        self.data_dictionary = {name: [] for name in logging_names_list}
        self.start_time = time.time()

        # ---------------------------------------------------------
        # STEP 2 & 3: SIGNS (+1 / -1) AND OFFSETS (radians)
        # ---------------------------------------------------------
        # HOW TO USE:
        # 1) First set offsets to 0, run, and move ONE joint at a time.
        # 2) If angle decreases when you rotate it in your chosen + direction, set sign = -1.
        # 3) Put leg into your "zero pose", read raw value theta_raw_zero, then set:
        #       offset = - sign * theta_raw_zero
        #
        # Start with these defaults; you MUST update them based on your robot.
        self.sign_f = np.array([+1.0, +1.0, +1.0])   # front left: [theta1, theta2, theta3]
        self.sign_b = np.array([+1.0, +1.0, +1.0])   # back left:  [theta1, theta2, theta3]
        self.offset_f = np.array([0.0, 0.0, 0.0])    # radians
        self.offset_b = np.array([0.0, 0.0, 0.0])    # radians

        # ---------------------------------------------------------
        # Your STEP 1–3 geometry (you gave frame 0,1,2,ee)
        # We interpret your numbers as cm and convert to meters.
        # frame0: (0,0,0)
        # frame1: (8,0,0)
        # frame2: (8,3,-8.5)
        # ee:     (17,3,-8.5)
        # ---------------------------------------------------------
        self.p01 = np.array([0.08, 0.0, 0.0])         # (8,0,0) cm -> m
        self.p12 = np.array([0.0, 0.03, -0.085])      # (0,3,-8.5) cm -> m
        self.p2e = np.array([0.09, 0.0, 0.0])         # (9,0,0) cm -> m

        # NOTE: You said "we don't need frame 3". That means your FK uses only theta1 & theta2.
        # theta3 is still read/logged but not used in FK.

    # ----------------------------
    # Logging / utility
    # ----------------------------
    def log_data(self, time_stamp, theta1_f, theta2_f, theta3_f, theta1_b, theta2_b, theta3_b,
                 end_effector_position_f, end_effector_position_b):
        self.data_dictionary["time_stamp"].append(time_stamp)
        self.data_dictionary["theta1_f"].append(theta1_f)
        self.data_dictionary["theta2_f"].append(theta2_f)
        self.data_dictionary["theta3_f"].append(theta3_f)
        self.data_dictionary["theta1_b"].append(theta1_b)
        self.data_dictionary["theta2_b"].append(theta2_b)
        self.data_dictionary["theta3_b"].append(theta3_b)
        self.data_dictionary["end_effector_position_f"].append(end_effector_position_f)
        self.data_dictionary["end_effector_position_b"].append(end_effector_position_b)
        with open(self.filename, "wb") as fh:
            pickle.dump(self.data_dictionary, fh)

    def publish_zero_gains(self):
        self.kp_publisher.publish(Float64MultiArray(data=[0.0] * 12))
        self.kd_publisher.publish(Float64MultiArray(data=[0.0] * 12))

    def listener_callback(self, msg: JointState):
        joints_of_interest = [
            "leg_front_l_1", "leg_front_l_2", "leg_front_l_3",
            "leg_back_l_1",  "leg_back_l_2",  "leg_back_l_3",
        ]
        try:
            self.joint_positions = [msg.position[msg.name.index(j)] for j in joints_of_interest]
        except ValueError:
            # Some joint name missing in the message; ignore this tick.
            return

    # ----------------------------
    # STEP 4: Homogeneous transforms
    # ----------------------------
    def rotation_x(self, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array(
            [[1, 0,  0, 0],
             [0, c, -s, 0],
             [0, s,  c, 0],
             [0, 0,  0, 1]], dtype=float
        )

    def rotation_y(self, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array(
            [[ c, 0, s, 0],
             [ 0, 1, 0, 0],
             [-s, 0, c, 0],
             [ 0, 0, 0, 1]], dtype=float
        )

    def rotation_z(self, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        return np.array(
            [[c, -s, 0, 0],
             [s,  c, 0, 0],
             [0,  0, 1, 0],
             [0,  0, 0, 1]], dtype=float
        )

    def translation(self, x, y, z):
        return np.array(
            [[1, 0, 0, x],
             [0, 1, 0, y],
             [0, 0, 1, z],
             [0, 0, 0, 1]], dtype=float
        )

    # ----------------------------
    # STEP 5: FK (Front Left) using frames 0,1,2,ee only
    # ----------------------------
    def forward_kinematics_f(self, theta1, theta2, theta3_unused):
        # Choose axes based on your frame design.
        # Common/simple default:
        # joint1 about z, joint2 about y
        T_0_1 = self.translation(*self.p01) @ self.rotation_z(theta1)
        T_1_2 = self.translation(*self.p12) @ self.rotation_y(theta2)
        T_2_ee = self.translation(*self.p2e)

        T_0_ee = T_0_1 @ T_1_2 @ T_2_ee
        return T_0_ee[:3, 3]

    # Back-left FK (same structure; adjust p01 if your back hip offset differs!)
    def forward_kinematics_b(self, theta1, theta2, theta3_unused):
        # IMPORTANT:
        # If your back-left hip location differs from front-left, you MUST change p01 for the back leg.
        # For now we use the same geometry as front-left as a placeholder.
        T_0_1 = self.translation(*self.p01) @ self.rotation_z(theta1)
        T_1_2 = self.translation(*self.p12) @ self.rotation_y(theta2)
        T_2_ee = self.translation(*self.p2e)

        T_0_ee = T_0_1 @ T_1_2 @ T_2_ee
        return T_0_ee[:3, 3]

    # ----------------------------
    # Main loop
    # ----------------------------
    def timer_callback(self):
        if self.joint_positions is None:
            return

        # RAW angles from /joint_states
        raw_f = np.array(self.joint_positions[0:3], dtype=float)
        raw_b = np.array(self.joint_positions[3:6], dtype=float)

        # STEP 2 & 3: apply sign + offset
        used_f = self.sign_f * raw_f + self.offset_f
        used_b = self.sign_b * raw_b + self.offset_b

        theta1_f, theta2_f, theta3_f = used_f.tolist()
        theta1_b, theta2_b, theta3_b = used_b.tolist()

        # FK
        end_effector_position_f = self.forward_kinematics_f(theta1_f, theta2_f, theta3_f)
        end_effector_position_b = self.forward_kinematics_b(theta1_b, theta2_b, theta3_b)

        # Log data
        time_stamp = time.time() - self.start_time
        self.log_data(
            time_stamp,
            theta1_f, theta2_f, theta3_f,
            theta1_b, theta2_b, theta3_b,
            end_effector_position_f, end_effector_position_b
        )

        # Publish marker
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
        marker.pose.position.x = float(end_effector_position_f[0])
        marker.pose.position.y = float(end_effector_position_f[1])
        marker.pose.position.z = float(end_effector_position_f[2])
        self.marker_publisher.publish(marker)

        # Publish position
        position = Float64MultiArray()
        position.data = end_effector_position_f.tolist()
        self.position_publisher.publish(position)

        # Print RAW + USED + EE (this is exactly what the lab hints you to do)
        self.get_logger().info(
            "RAW front:[{:.3f},{:.3f},{:.3f}] USED front:[{:.3f},{:.3f},{:.3f}] | "
            "EE_f:[{:.3f},{:.3f},{:.3f}]".format(
                raw_f[0], raw_f[1], raw_f[2],
                theta1_f, theta2_f, theta3_f,
                end_effector_position_f[0], end_effector_position_f[1], end_effector_position_f[2]
            )
        )


def main(args=None):
    rclpy.init(args=args)
    node = ForwardKinematics()
    rclpy.spin(node)


if __name__ == "__main__":
    main()
