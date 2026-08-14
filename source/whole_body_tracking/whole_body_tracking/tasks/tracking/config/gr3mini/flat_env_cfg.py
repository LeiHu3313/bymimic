from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from whole_body_tracking.robots.gr3mini import (
    GR3MINI_V211_ACTION_SCALE,
    GR3MINI_V211_CFG,
    GR3MINI_V211_POLICY_JOINT_NAMES,
)
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


# Feet and hands are the only links allowed to make contact during tracking.
# Every listed link contributes one violation when its contact-force norm exceeds
# the 1 N threshold configured by TrackingEnvCfg.rewards.undesired_contacts.
GR3MINI_V211_UNDESIRED_CONTACT_BODY_NAMES = [
    "base_link",
    "waist_yaw_link",
    "left_thigh_pitch_link",
    "left_thigh_roll_link",
    "left_thigh_yaw_link",
    "left_shank_pitch_link",
    "left_foot_pitch_link",
    "right_thigh_pitch_link",
    "right_thigh_roll_link",
    "right_thigh_yaw_link",
    "right_shank_pitch_link",
    "right_foot_pitch_link",
    "left_upper_arm_pitch_link",
    "left_upper_arm_roll_link",
    "left_upper_arm_yaw_link",
    "left_lower_arm_pitch_link",
    "right_upper_arm_pitch_link",
    "right_upper_arm_roll_link",
    "right_upper_arm_yaw_link",
    "right_lower_arm_pitch_link",
]


@configclass
class GR3MiniV211FlatEnvCfg(TrackingEnvCfg):
    """Flat-ground motion tracking configuration for GR3Mini V2.1.1."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = GR3MINI_V211_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.contact_forces.debug_vis = False

        # The policy controls all 23 articulation joints in a stable order. The
        # two head connections are fixed joints in the training URDF.
        self.actions.joint_pos.joint_names = GR3MINI_V211_POLICY_JOINT_NAMES
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = GR3MINI_V211_ACTION_SCALE

        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = [
            "base_link",
            "left_shank_pitch_link",
            "left_foot_roll_link",
            "right_shank_pitch_link",
            "right_foot_roll_link",
            "waist_yaw_link",
            "left_upper_arm_yaw_link",
            "left_hand_yaw_link",
            "right_upper_arm_yaw_link",
            "right_hand_yaw_link",
        ]

        # torso/head links are merged into this body through fixed joints.
        self.events.base_com.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names="waist_yaw_link"
        )

        # self.rewards.undesired_contacts.params["sensor_cfg"] = SceneEntityCfg(
        #     "contact_forces",
        #     body_names=GR3MINI_V211_UNDESIRED_CONTACT_BODY_NAMES,
        # )
        self.terminations.anchor_pos.params["threshold"] = 0.35
        self.terminations.anchor_ori.params["threshold"] = 0.7
        self.terminations.ee_body_pos.params["threshold"] = 0.30
        self.terminations.ee_body_pos.params["body_names"] = [
            "left_foot_roll_link",
            "right_foot_roll_link",
            "left_hand_yaw_link",
            "right_hand_yaw_link",
        ]
