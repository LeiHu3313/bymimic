"""Convert the 23-DoF GR3Mini LAFAN motions to tracking-ready NPZ files.

The input files contain only root pose and 23 controlled joint positions.  This
script uses the 23-DoF URDF with both head joints fixed, resamples to 50 Hz by default, performs
forward kinematics from the checked-in URDF, and adds the joint/body velocities
required by :class:`MotionLoader`.

By default the script converts the 40 motion names shared with the G1 LAFAN
training set::

    python scripts/utils/convert_gr3mini_lafan.py \
      --input-dir /path/to/robot_state_npz_23dof \
      --output-dir /path/to/gr3mini_lafan_g1_40
"""

from __future__ import annotations

import argparse
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_MANIFEST = SCRIPT_DIR / "gr3mini_lafan_g1_40.txt"
DEFAULT_URDF = (
    PROJECT_ROOT
    / "source"
    / "whole_body_tracking"
    / "whole_body_tracking"
    / "assets"
    / "fourier"
    / "gr3mini_v211"
    / "urdf"
    / "gr3mini_v211.urdf"
)

HEAD_JOINT_NAMES = ("head_yaw_joint", "head_pitch_joint")

# Joint order stored by the UFO 23-DoF retargeting pipeline.
SOURCE_JOINT_NAMES = (
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
)

# Isaac Lab articulation storage order for the checked-in GR3Mini URDF.
ISAAC_JOINT_NAMES = (
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
)

# Isaac Lab body storage order after the URDF importer merges fixed joints.
ISAAC_BODY_NAMES = (
    "base_link",
    "left_thigh_pitch_link",
    "right_thigh_pitch_link",
    "waist_yaw_link",
    "left_thigh_roll_link",
    "right_thigh_roll_link",
    "left_upper_arm_pitch_link",
    "right_upper_arm_pitch_link",
    "left_thigh_yaw_link",
    "right_thigh_yaw_link",
    "left_upper_arm_roll_link",
    "right_upper_arm_roll_link",
    "left_shank_pitch_link",
    "right_shank_pitch_link",
    "left_upper_arm_yaw_link",
    "right_upper_arm_yaw_link",
    "left_foot_pitch_link",
    "right_foot_pitch_link",
    "left_lower_arm_pitch_link",
    "right_lower_arm_pitch_link",
    "left_foot_roll_link",
    "right_foot_roll_link",
    "left_hand_yaw_link",
    "right_hand_yaw_link",
)


@dataclass(frozen=True)
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    origin_xyz: np.ndarray
    origin_rot: np.ndarray
    axis: np.ndarray


@dataclass(frozen=True)
class UrdfModel:
    root_link: str
    joints: list[UrdfJoint]
    body_com_offsets: dict[str, np.ndarray]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing the UFO 23-DoF NPZ files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for tracking-ready NPZ files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST,
        help="One motion stem per line.",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help="GR3Mini V2.1.1 URDF used for FK.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=50.0,
        help="Output sampling rate (default: 50 Hz).",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace existing output files."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the manifest and source schemas without writing output files.",
    )
    return parser.parse_args()


def _read_manifest(path: Path) -> list[str]:
    names = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name and not name.startswith("#")]
    if len(names) != 40 or len(set(names)) != 40:
        raise ValueError(
            f"Expected 40 unique G1-aligned motions in {path}, found {len(names)}."
        )
    return names


def _vector(text: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if text is None:
        return np.asarray(default, dtype=np.float64)
    value = np.fromstring(text, sep=" ", dtype=np.float64)
    if value.shape != (3,):
        raise ValueError(f"Expected a 3-vector, got {text!r}.")
    return value


def _load_urdf(path: Path) -> UrdfModel:
    root = ET.parse(path).getroot()
    link_elements = {
        element.attrib["name"]: element for element in root.findall("link")
    }
    links = set(link_elements)
    joints: list[UrdfJoint] = []
    child_links: set[str] = set()
    for element in root.findall("joint"):
        parent = element.find("parent").attrib["link"]
        child = element.find("child").attrib["link"]
        origin = element.find("origin")
        axis = element.find("axis")
        origin_xyz = _vector(
            None if origin is None else origin.get("xyz"), (0.0, 0.0, 0.0)
        )
        origin_rpy = _vector(
            None if origin is None else origin.get("rpy"), (0.0, 0.0, 0.0)
        )
        axis_xyz = _vector(None if axis is None else axis.get("xyz"), (1.0, 0.0, 0.0))
        axis_norm = np.linalg.norm(axis_xyz)
        if axis_norm == 0.0:
            raise ValueError(f"Joint {element.attrib['name']} has a zero axis.")
        joints.append(
            UrdfJoint(
                name=element.attrib["name"],
                joint_type=element.attrib["type"],
                parent=parent,
                child=child,
                origin_xyz=origin_xyz,
                origin_rot=Rotation.from_euler("xyz", origin_rpy).as_matrix(),
                axis=axis_xyz / axis_norm,
            )
        )
        child_links.add(child)

    root_links = links - child_links
    if len(root_links) != 1:
        raise ValueError(f"Expected one URDF root link, found {sorted(root_links)}.")
    root_link = next(iter(root_links))

    ordered: list[UrdfJoint] = []
    available = {root_link}
    remaining = joints.copy()
    while remaining:
        ready = [joint for joint in remaining if joint.parent in available]
        if not ready:
            raise ValueError("URDF joint graph is disconnected or cyclic.")
        for joint in ready:
            ordered.append(joint)
            available.add(joint.child)
            remaining.remove(joint)

    fixed_children: dict[str, list[UrdfJoint]] = {}
    for joint in ordered:
        if joint.joint_type == "fixed":
            fixed_children.setdefault(joint.parent, []).append(joint)

    def combined_com(link_name: str) -> np.ndarray:
        weighted_positions: list[np.ndarray] = []
        masses: list[float] = []

        def visit_fixed(link: str, position: np.ndarray, rotation: np.ndarray) -> None:
            inertial = link_elements[link].find("inertial")
            if inertial is not None:
                mass = float(inertial.find("mass").attrib["value"])
                origin = inertial.find("origin")
                local_com = _vector(
                    None if origin is None else origin.get("xyz"), (0.0, 0.0, 0.0)
                )
                weighted_positions.append(position + rotation @ local_com)
                masses.append(mass)
            for fixed_joint in fixed_children.get(link, []):
                child_position = position + rotation @ fixed_joint.origin_xyz
                child_rotation = rotation @ fixed_joint.origin_rot
                visit_fixed(fixed_joint.child, child_position, child_rotation)

        visit_fixed(link_name, np.zeros(3), np.eye(3))
        total_mass = sum(masses)
        if total_mass <= 0.0:
            return np.zeros(3, dtype=np.float64)
        return (
            sum(
                mass * position
                for mass, position in zip(masses, weighted_positions, strict=True)
            )
            / total_mass
        )

    body_com_offsets = {name: combined_com(name) for name in ISAAC_BODY_NAMES}
    return UrdfModel(
        root_link=root_link, joints=ordered, body_com_offsets=body_com_offsets
    )


def _validate_source(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        required = {"root_pos", "root_quat", "dof_pos", "fps", "joint_names"}
        missing = required - set(data.files)
        if missing:
            raise ValueError(f"{path}: missing fields {sorted(missing)}.")
        result = {key: np.asarray(data[key]).copy() for key in data.files}

    frame_count = result["root_pos"].shape[0]
    if result["root_pos"].shape != (frame_count, 3):
        raise ValueError(f"{path}: invalid root_pos shape {result['root_pos'].shape}.")
    if result["root_quat"].shape != (frame_count, 4):
        raise ValueError(
            f"{path}: invalid root_quat shape {result['root_quat'].shape}."
        )
    if result["dof_pos"].shape != (frame_count, len(SOURCE_JOINT_NAMES)):
        raise ValueError(
            f"{path}: expected 23-DoF data, got {result['dof_pos'].shape}."
        )
    joint_names = tuple(str(name) for name in result["joint_names"].tolist())
    if joint_names != SOURCE_JOINT_NAMES:
        raise ValueError(f"{path}: unexpected joint order: {joint_names}.")
    fps = float(np.asarray(result["fps"]).reshape(-1)[0])
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{path}: invalid fps {fps}.")
    if frame_count < 2:
        raise ValueError(f"{path}: at least two frames are required.")
    return result


def _resample(
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    dof_pos: np.ndarray,
    input_fps: float,
    output_fps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if output_fps <= 0.0:
        raise ValueError(f"Output fps must be positive, got {output_fps}.")
    duration = (len(root_pos) - 1) / input_fps
    input_times = np.arange(len(root_pos), dtype=np.float64) / input_fps
    output_times = np.arange(0.0, duration + 1.0e-8, 1.0 / output_fps, dtype=np.float64)
    position_out = np.stack(
        [np.interp(output_times, input_times, root_pos[:, i]) for i in range(3)], axis=1
    )
    dof_out = np.stack(
        [
            np.interp(output_times, input_times, dof_pos[:, i])
            for i in range(dof_pos.shape[1])
        ],
        axis=1,
    )
    quaternion_out = Slerp(input_times, Rotation.from_quat(root_quat_xyzw))(
        output_times
    ).as_quat()
    return (
        position_out.astype(np.float32),
        quaternion_out.astype(np.float32),
        dof_out.astype(np.float32),
    )


def _expand_joints(
    source_dof_pos: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    values_by_name = {
        name: source_dof_pos[:, index] for index, name in enumerate(SOURCE_JOINT_NAMES)
    }
    for name in HEAD_JOINT_NAMES:
        values_by_name[name] = np.zeros(len(source_dof_pos), dtype=np.float32)
    joint_pos = np.stack([values_by_name[name] for name in ISAAC_JOINT_NAMES], axis=1)
    return values_by_name, joint_pos.astype(np.float32)


def _forward_kinematics(
    model: UrdfModel,
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    joint_values: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = len(root_pos)
    link_pos = {model.root_link: root_pos.astype(np.float64)}
    link_rot = {model.root_link: Rotation.from_quat(root_quat_xyzw).as_matrix()}

    for joint in model.joints:
        parent_pos = link_pos[joint.parent]
        parent_rot = link_rot[joint.parent]
        child_pos = parent_pos + np.einsum("fij,j->fi", parent_rot, joint.origin_xyz)
        child_rot = np.einsum("fij,jk->fik", parent_rot, joint.origin_rot)
        if joint.joint_type in {"revolute", "continuous"}:
            if joint.name not in joint_values:
                raise ValueError(
                    f"No motion value was provided for URDF joint {joint.name}."
                )
            rotation = Rotation.from_rotvec(
                joint_values[joint.name][:, None] * joint.axis
            ).as_matrix()
            child_rot = np.einsum("fij,fjk->fik", child_rot, rotation)
        elif joint.joint_type != "fixed":
            raise ValueError(
                f"Unsupported URDF joint type {joint.joint_type!r} for {joint.name}."
            )
        link_pos[joint.child] = child_pos
        link_rot[joint.child] = child_rot

    missing = set(ISAAC_BODY_NAMES) - set(link_pos)
    if missing:
        raise ValueError(f"URDF is missing expected Isaac bodies: {sorted(missing)}.")
    body_pos = np.stack([link_pos[name] for name in ISAAC_BODY_NAMES], axis=1).astype(
        np.float32
    )
    body_rot = np.stack([link_rot[name] for name in ISAAC_BODY_NAMES], axis=1)
    body_quat_xyzw = (
        Rotation.from_matrix(body_rot.reshape(-1, 3, 3))
        .as_quat()
        .reshape(frame_count, -1, 4)
    )
    body_quat_wxyz = body_quat_xyzw[..., [3, 0, 1, 2]].astype(np.float32)
    return body_pos, body_quat_wxyz


def _finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    return np.asarray(np.gradient(values, dt, axis=0), dtype=np.float32)


def _angular_velocity(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
    frame_count, body_count, _ = quat_wxyz.shape
    result = np.zeros((frame_count, body_count, 3), dtype=np.float32)
    for body_index in range(body_count):
        rotations = Rotation.from_quat(quat_wxyz[:, body_index, [1, 2, 3, 0]])
        central = rotations[2:] * rotations[:-2].inv()
        result[1:-1, body_index] = (central.as_rotvec() / (2.0 * dt)).astype(np.float32)
        result[0, body_index] = (rotations[1] * rotations[0].inv()).as_rotvec().astype(
            np.float32
        ) / dt
        result[-1, body_index] = (
            rotations[-1] * rotations[-2].inv()
        ).as_rotvec().astype(np.float32) / dt
    return result


def _forward_kinematic_velocities(
    model: UrdfModel,
    root_pos: np.ndarray,
    root_quat_xyzw: np.ndarray,
    joint_values: dict[str, np.ndarray],
    joint_velocities: dict[str, np.ndarray],
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]][:, None, :]
    link_pos = {model.root_link: root_pos.astype(np.float64)}
    link_rot = {model.root_link: Rotation.from_quat(root_quat_xyzw).as_matrix()}
    link_lin_vel = {
        model.root_link: _finite_difference(root_pos, dt).astype(np.float64)
    }
    link_ang_vel = {
        model.root_link: _angular_velocity(root_quat_wxyz, dt)[:, 0].astype(np.float64)
    }

    for joint in model.joints:
        parent_pos = link_pos[joint.parent]
        parent_rot = link_rot[joint.parent]
        parent_lin_vel = link_lin_vel[joint.parent]
        parent_ang_vel = link_ang_vel[joint.parent]
        child_pos = parent_pos + np.einsum("fij,j->fi", parent_rot, joint.origin_xyz)
        child_rot = np.einsum("fij,jk->fik", parent_rot, joint.origin_rot)
        child_lin_vel = parent_lin_vel + np.cross(
            parent_ang_vel, child_pos - parent_pos
        )
        child_ang_vel = parent_ang_vel.copy()
        if joint.joint_type in {"revolute", "continuous"}:
            rotation = Rotation.from_rotvec(
                joint_values[joint.name][:, None] * joint.axis
            ).as_matrix()
            world_axis = np.einsum("fij,j->fi", child_rot, joint.axis)
            child_rot = np.einsum("fij,fjk->fik", child_rot, rotation)
            child_ang_vel += world_axis * joint_velocities[joint.name][:, None]
        link_pos[joint.child] = child_pos
        link_rot[joint.child] = child_rot
        link_lin_vel[joint.child] = child_lin_vel
        link_ang_vel[joint.child] = child_ang_vel

    body_lin_vel = []
    for name in ISAAC_BODY_NAMES:
        com_offset_w = np.einsum(
            "fij,j->fi", link_rot[name], model.body_com_offsets[name]
        )
        body_lin_vel.append(
            link_lin_vel[name] + np.cross(link_ang_vel[name], com_offset_w)
        )
    body_lin_vel = np.stack(body_lin_vel, axis=1).astype(np.float32)
    body_ang_vel = np.stack(
        [link_ang_vel[name] for name in ISAAC_BODY_NAMES], axis=1
    ).astype(np.float32)
    return body_lin_vel, body_ang_vel


def _convert_motion(
    source: dict[str, np.ndarray], output_fps: float, urdf: UrdfModel
) -> dict:
    input_fps = float(np.asarray(source["fps"]).reshape(-1)[0])
    root_pos, root_quat_xyzw, source_dof_pos = _resample(
        source["root_pos"],
        source["root_quat"],
        source["dof_pos"],
        input_fps,
        output_fps,
    )
    joint_values, joint_pos = _expand_joints(source_dof_pos)
    joint_vel = _finite_difference(joint_pos, 1.0 / output_fps)
    joint_velocities = {
        name: joint_vel[:, index] for index, name in enumerate(ISAAC_JOINT_NAMES)
    }
    body_pos_w, body_quat_w = _forward_kinematics(
        urdf, root_pos, root_quat_xyzw, joint_values
    )
    dt = 1.0 / output_fps
    body_lin_vel_w, body_ang_vel_w = _forward_kinematic_velocities(
        urdf,
        root_pos,
        root_quat_xyzw,
        joint_values,
        joint_velocities,
        dt,
    )
    return {
        "fps": np.asarray([output_fps], dtype=np.float32),
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "joint_names": np.asarray(ISAAC_JOINT_NAMES),
        "body_names": np.asarray(ISAAC_BODY_NAMES),
        "fixed_joint_names": np.asarray(HEAD_JOINT_NAMES),
    }


def main() -> None:
    args = _parse_args()
    names = _read_manifest(args.manifest)
    missing = [name for name in names if not (args.input_dir / f"{name}.npz").is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} manifest motions under {args.input_dir}: {missing}"
        )

    sources = [
        (name, _validate_source(args.input_dir / f"{name}.npz")) for name in names
    ]
    total_frames = sum(source["root_pos"].shape[0] for _, source in sources)
    print(f"[INFO] Validated {len(sources)} motions ({total_frames} input frames).")
    if args.validate_only:
        return

    urdf = _load_urdf(args.urdf)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for index, (name, source) in enumerate(sources, start=1):
        output_path = args.output_dir / f"{name}.npz"
        if output_path.exists() and not args.overwrite:
            print(
                f"[SKIP {index:02d}/40] {output_path} already exists (pass --overwrite to replace it)."
            )
            continue
        converted = _convert_motion(source, args.output_fps, urdf)
        np.savez_compressed(output_path, **converted)
        print(
            f"[SAVE {index:02d}/40] {output_path} ({converted['joint_pos'].shape[0]} frames)"
        )


if __name__ == "__main__":
    main()
