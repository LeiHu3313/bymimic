"""Fourier GR3Mini V2.1.1 articulation configuration."""

import math

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR
from whole_body_tracking.robots.fourier_actuator import FourierActuatorTNCfg


# Active joint ordering used by UFO's GR3Mini 23-DoF motion files. Isaac Lab's
# articulation storage order is resolved from the URDF and is different.
GR3MINI_V211_JOINT_NAMES = [
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_pitch_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_pitch_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
]

# Order exposed by the Isaac Lab articulation generated from the URDF.  Motion
# files consumed by the tracking task must use this order.
GR3MINI_V211_ISAAC_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_knee_pitch_joint",
    "right_knee_pitch_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_elbow_pitch_joint",
    "right_elbow_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

# The head links are rigidly attached in the training URDF, matching UFO's
# gr3mini_v211_23dof model rather than adding two unused articulation joints.
GR3MINI_V211_FIXED_JOINT_NAMES = ("head_yaw_joint", "head_pitch_joint")
GR3MINI_V211_POLICY_JOINT_NAMES = GR3MINI_V211_ISAAC_JOINT_NAMES

GR3MINI_V211_ACTION_SCALE = 0.25


GR3MINI_V211_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ASSET_DIR}/fourier/gr3mini_v211/urdf/gr3mini_v211.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0, damping=0
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            "waist_yaw_joint": 0.0,
            "left_shoulder_pitch_joint": 0.174533,
            "left_shoulder_roll_joint": 0.349066,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_pitch_joint": -0.349066,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.174533,
            "right_shoulder_roll_joint": -0.349066,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_pitch_joint": -0.349066,
            "right_wrist_yaw_joint": 0.0,
            ".*_hip_pitch_joint": -0.20,
            ".*_hip_roll_joint": 0.0,
            ".*_hip_yaw_joint": 0.0,
            ".*_knee_pitch_joint": 0.30,
            ".*_ankle_pitch_joint": -0.10,
            ".*_ankle_roll_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "pj3_97": FourierActuatorTNCfg(
            joint_names_expr=[
                ".*_hip_pitch_joint",
                ".*_knee_pitch_joint",
                "waist_yaw_joint",
            ],
            stiffness={
                ".*_hip_pitch_joint": 130.0,
                ".*_knee_pitch_joint": 130.0,
                "waist_yaw_joint": 100.0,
            },
            damping={
                ".*_hip_pitch_joint": 6.0,
                ".*_knee_pitch_joint": 5.0,
                "waist_yaw_joint": 5.0,
            },
            effort_limit=120.0,
            effort_limit_sim=1.0e9,
            velocity_limit_sim=125.0 / 60.0 * 2.0 * math.pi,
            armature=0.029064366,
            friction=0.02,
            X1=50.0 / 60.0 * 2.0 * math.pi,
            X2=125.0 / 60.0 * 2.0 * math.pi,
            X3=125.0 / 60.0 * 2.0 * math.pi,
            Y1=120.0,
            Y2=60.0,
            Y3=120.0,
            Fd=0.01,
        ),
        "pj3_75": FourierActuatorTNCfg(
            joint_names_expr=[
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_shoulder_pitch_joint",
            ],
            stiffness={
                ".*_hip_roll_joint": 100.0,
                ".*_hip_yaw_joint": 50.0,
                ".*_shoulder_pitch_joint": 50.0,
            },
            damping={
                ".*_hip_roll_joint": 5.0,
                ".*_hip_yaw_joint": 4.0,
                ".*_shoulder_pitch_joint": 4.0,
            },
            effort_limit=40.0,
            effort_limit_sim=1.0e9,
            velocity_limit_sim=140.0 / 60.0 * 2.0 * math.pi,
            armature=0.00884,
            friction=0.02,
            X1=60.0 / 60.0 * 2.0 * math.pi,
            X2=140.0 / 60.0 * 2.0 * math.pi,
            X3=140.0 / 60.0 * 2.0 * math.pi,
            Y1=40.0,
            Y2=20.0,
            Y3=40.0,
            Fd=0.01,
        ),
        "pj3_60": FourierActuatorTNCfg(
            joint_names_expr=[
                ".*_ankle_pitch_joint",
                ".*_ankle_roll_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_pitch_joint",
            ],
            stiffness={
                ".*_ankle_pitch_joint": 53.0,
                ".*_ankle_roll_joint": 26.0,
                ".*_shoulder_roll_joint": 30.0,
                ".*_shoulder_yaw_joint": 30.0,
                ".*_elbow_pitch_joint": 30.0,
            },
            damping={
                ".*_ankle_pitch_joint": 2.65,
                ".*_ankle_roll_joint": 1.30,
                ".*_shoulder_roll_joint": 1.0,
                ".*_shoulder_yaw_joint": 1.0,
                ".*_elbow_pitch_joint": 1.0,
            },
            effort_limit={
                ".*_ankle_pitch_joint": 52.0,
                ".*_ankle_roll_joint": 25.0,
                ".*_shoulder_roll_joint": 28.5,
                ".*_shoulder_yaw_joint": 28.5,
                ".*_elbow_pitch_joint": 28.5,
            },
            effort_limit_sim=1.0e9,
            velocity_limit_sim=180.0 / 60.0 * 2.0 * math.pi,
            armature={
                ".*_ankle_pitch_joint": 0.01324486112,
                ".*_ankle_roll_joint": 0.00649747904,
                ".*_shoulder_roll_joint": 0.003123788,
                ".*_shoulder_yaw_joint": 0.003123788,
                ".*_elbow_pitch_joint": 0.003123788,
            },
            friction=0.02,
            X1=75.0 / 60.0 * 2.0 * math.pi,
            X2=180.0 / 60.0 * 2.0 * math.pi,
            X3=180.0 / 60.0 * 2.0 * math.pi,
            Y1=28.0,
            Y2=10.0,
            Y3=28.0,
            Fd=0.01,
        ),
        "hj2_52": FourierActuatorTNCfg(
            joint_names_expr=[".*_wrist_yaw_joint"],
            stiffness=15.0,
            damping=2.0,
            effort_limit=14.0,
            effort_limit_sim=1.0e9,
            velocity_limit_sim=50.0 / 60.0 * 2.0 * math.pi,
            armature=0.0489648,
            friction=0.02,
            X1=0.0,
            X2=50.0 / 60.0 * 2.0 * math.pi,
            X3=50.0 / 60.0 * 2.0 * math.pi,
            Y1=15.0,
            Y2=10.0,
            Y3=15.0,
            Fd=0.01,
        ),
    },
)
