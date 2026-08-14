"""Convert a GR3Mini retargeting PKL into a tracking-ready 23-DoF NPZ.

The training GR3Mini URDF fixes head yaw and pitch.  Source PKLs may still
contain these two joints, so this converter selects the active 23 joints by
name and delegates forward kinematics to ``convert_gr3mini_lafan.py``.

Example:

    python scripts/utils/convert_gr3mini_pkl.py \
      --input-file Extended_3_stageii_from_g1_gr3mini_v211.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from convert_gr3mini_lafan import (
    DEFAULT_URDF,
    SOURCE_JOINT_NAMES,
    _convert_motion,
    _load_urdf,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-file", type=Path, required=True, help="GR3Mini retargeting PKL file."
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Output NPZ path. Defaults to data/gr3mini/custom/<input-stem>.npz.",
    )
    parser.add_argument(
        "--output-fps",
        type=float,
        default=50.0,
        help="Output sampling rate (default: 50 Hz).",
    )
    parser.add_argument(
        "--urdf",
        type=Path,
        default=DEFAULT_URDF,
        help="GR3Mini 23-DoF URDF used for FK.",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing output file."
    )
    return parser.parse_args()


def _load_source_motion(path: Path) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with path.open("rb") as file:
        # PKL is executable Python serialization: only use files from a trusted source.
        raw = pickle.load(file)
    if not isinstance(raw, dict):
        raise TypeError(f"{path}: expected a dict, got {type(raw).__name__}.")

    required_fields = {
        "fps",
        "root_pos",
        "root_rot",
        "dof_pos",
        "dof_names",
        "root_quat_format",
    }
    missing = required_fields - raw.keys()
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)}.")

    root_pos = np.asarray(raw["root_pos"], dtype=np.float32)
    root_quat = np.asarray(raw["root_rot"], dtype=np.float32)
    dof_pos = np.asarray(raw["dof_pos"], dtype=np.float32)
    dof_names = tuple(str(name) for name in raw["dof_names"])
    fps = float(raw["fps"])
    quat_order = str(raw["root_quat_format"]).lower()

    frame_count = root_pos.shape[0]
    if root_pos.shape != (frame_count, 3):
        raise ValueError(f"{path}: invalid root_pos shape {root_pos.shape}.")
    if root_quat.shape != (frame_count, 4):
        raise ValueError(f"{path}: invalid root_rot shape {root_quat.shape}.")
    if dof_pos.shape != (frame_count, len(dof_names)):
        raise ValueError(
            f"{path}: dof_pos shape {dof_pos.shape} does not match dof_names ({len(dof_names)})."
        )
    if len(set(dof_names)) != len(dof_names):
        raise ValueError(f"{path}: dof_names contains duplicates.")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"{path}: invalid fps {fps}.")
    if quat_order == "wxyz":
        root_quat = root_quat[:, [1, 2, 3, 0]]
    elif quat_order != "xyzw":
        raise ValueError(f"{path}: unsupported root quaternion order {quat_order!r}.")
    if (
        not np.isfinite(root_pos).all()
        or not np.isfinite(root_quat).all()
        or not np.isfinite(dof_pos).all()
    ):
        raise ValueError(f"{path}: source motion contains non-finite values.")

    missing_joints = set(SOURCE_JOINT_NAMES) - set(dof_names)
    if missing_joints:
        raise ValueError(
            f"{path}: source is missing active joints {sorted(missing_joints)}."
        )
    source_joint_indices = [dof_names.index(name) for name in SOURCE_JOINT_NAMES]
    source = {
        "fps": np.asarray([fps], dtype=np.float32),
        "root_pos": root_pos,
        "root_quat": root_quat,
        "dof_pos": dof_pos[:, source_joint_indices],
        "joint_names": np.asarray(SOURCE_JOINT_NAMES),
    }
    metadata = {
        "source_dof_names": np.asarray(dof_names),
        "source_root_quat_format": np.asarray(quat_order),
        "source_format": np.asarray(str(raw.get("format", "unknown"))),
        "source_name": np.asarray(str(raw.get("name", path.stem))),
    }
    return source, metadata


def main() -> None:
    args = _parse_args()
    if not args.input_file.is_file():
        raise FileNotFoundError(f"Input PKL does not exist: {args.input_file}")
    output_file = args.output_file
    if output_file is None:
        output_file = Path("data/gr3mini/custom") / f"{args.input_file.stem}.npz"
    if output_file.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {output_file}. Pass --overwrite to replace it."
        )

    source, metadata = _load_source_motion(args.input_file)
    converted = _convert_motion(source, args.output_fps, _load_urdf(args.urdf))
    converted.update(metadata)
    converted["source_file"] = np.asarray(str(args.input_file.resolve()))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **converted)
    print(
        f"[INFO] Converted {args.input_file} -> {output_file} "
        f"({converted['joint_pos'].shape[0]} frames at {float(converted['fps'][0]):g} Hz, 23 DoF)."
    )


if __name__ == "__main__":
    main()
