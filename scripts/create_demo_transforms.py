#!/usr/bin/env python3
"""Create smooth novel-view demos for a calibrated multi-camera transform set.

Three trajectories are available.  A standard ``sweep`` starts exactly at a
selected test camera, follows the calibrated training-camera arc at the same
height tier, reverses at both ends, and returns to the starting camera.  Its
``half-ellipse`` interpolation variant instead makes one stable open pass
between the two rail endpoints and aims every view at the consensus point of
the training-camera optical rays.  ``ellipse`` is centered exactly at the
selected test-camera position.  Its plane can follow the neighboring
training-camera chord and world-up directions or the exact test-camera image
plane while remaining in supported space.  ``staged-ellipse`` first advances
time at the fixed test view, freezes
time while moving from the center onto a full ellipse and back, then resumes
time at the fixed test view.

Sweep poses can use piecewise linear/quaternion-SLERP interpolation, smoother
natural-cubic position and sign-aligned quaternion curves, or an analytic
arc-length-parameterized half-ellipse with fixed-target look-at rotations.
Ellipse rotations continuously look toward the optical-axis target of the test
camera.  The stored matrices remain OpenGL/Blender camera-to-world matrices, as
expected by this project's transform loader.

The input format is the one used by this project's multi-camera datasets:
frame paths end in ``_<frame number>`` and calibrated focal/principal-point
values are available per frame or at the JSON top level.

For sweep and ellipse trajectories, ``--frames-per-timestamp`` can hold each
real test timestamp for several output views while the novel camera keeps
moving along its spatial path.
"""

import argparse
import copy
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np


INTRINSIC_KEYS = ("fl_x", "fl_y", "cx", "cy")
ELLIPSE_HORIZONTAL_RADIUS = 0.90
ELLIPSE_VERTICAL_RADIUS = 0.50
SWEEP_CRUISE_EASE_FRACTION = 0.20


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a closed sweep, periodic local ellipse, or staged "
            "time/ellipse demo within supported training-camera space."
        )
    )
    parser.add_argument(
        "dataset",
        type=Path,
        help="Dataset directory containing transforms_train.json and transforms_test.json.",
    )
    parser.add_argument(
        "--trajectory",
        choices=("sweep", "ellipse", "staged-ellipse"),
        default="sweep",
        help="Trajectory type (default: sweep).",
    )
    parser.add_argument(
        "--train-file",
        default="transforms_train.json",
        help="Training transform JSON, relative to the dataset unless absolute.",
    )
    parser.add_argument(
        "--test-file",
        default="transforms_test.json",
        help="Test transform JSON, relative to the dataset unless absolute.",
    )
    parser.add_argument(
        "--output",
        default="",
        help=(
            "Output JSON, relative to the dataset unless absolute (default: "
            "transforms_demo.json for sweep, transforms_demo_ellipse.json for "
            "ellipse, or transforms_demo_staged_ellipse.json for staged-ellipse)."
        ),
    )
    parser.add_argument(
        "--test-camera",
        default="",
        help="Test camera prefix such as cam0026 (default: camera of the first test frame).",
    )
    parser.add_argument(
        "--start-test-frame",
        type=int,
        default=0,
        help=(
            "Index within the selected test camera's frames used as the sweep "
            "start or ellipse reference pose."
        ),
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=0,
        help=(
            "Sweep output view count (default: number of selected test-camera "
            "frames). Sweep and ellipse counts are instead derived from "
            "--frames-per-timestamp when that option is set."
        ),
    )
    parser.add_argument(
        "--frames-per-timestamp",
        type=int,
        default=0,
        help=(
            "Sweep/ellipse views emitted at each selected test timestamp. For "
            "ellipse the default is 4, producing 240 views for the default 60 "
            "timestamps; for sweep the default is disabled."
        ),
    )
    parser.add_argument(
        "--sweep-timestamp-start-frame",
        type=int,
        default=None,
        help=(
            "First numeric source-frame timestamp used by a sweep. When omitted, "
            "the first selected test-camera frame is used."
        ),
    )
    parser.add_argument(
        "--sweep-timestamp-end-frame",
        type=int,
        default=None,
        help=(
            "Last numeric source-frame timestamp used by a sweep, inclusive. "
            "When omitted, the last selected test-camera frame is used."
        ),
    )
    parser.add_argument(
        "--ellipse-timestamp-start-frame",
        type=int,
        default=0,
        help="First source-frame timestamp used by the ellipse (default: 0).",
    )
    parser.add_argument(
        "--ellipse-timestamp-end-frame",
        type=int,
        default=59,
        help="Last source-frame timestamp used by the ellipse, inclusive (default: 59).",
    )
    parser.add_argument(
        "--ellipse-scale",
        type=float,
        default=1.0,
        help=(
            "Scale in (0, 1] applied to the conservative local ellipse "
            "(default: 1)."
        ),
    )
    parser.add_argument(
        "--ellipse-plane",
        choices=("rail", "camera"),
        default="rail",
        help=(
            "Orient the ellipse from the neighboring training-camera rail or "
            "exactly in the test camera's image plane (default: rail)."
        ),
    )
    parser.add_argument(
        "--ellipse-horizontal-radius",
        type=float,
        default=0.0,
        help=(
            "Override the ellipse horizontal radius before --ellipse-scale; "
            "must be supplied with --ellipse-vertical-radius."
        ),
    )
    parser.add_argument(
        "--ellipse-vertical-radius",
        type=float,
        default=0.0,
        help=(
            "Override the ellipse vertical radius before --ellipse-scale; "
            "must be supplied with --ellipse-horizontal-radius."
        ),
    )
    parser.add_argument(
        "--ellipse-bounds",
        choices=("aabb", "radial"),
        default="aabb",
        help=(
            "Validate ellipse centers against the training-camera coordinate "
            "box or their maximum distance from the mean training-camera center "
            "(default: aabb)."
        ),
    )
    parser.add_argument(
        "--freeze-frame",
        type=int,
        default=70,
        help=(
            "Numeric source frame at which staged-ellipse freezes scene time "
            "while the camera moves (default: 70)."
        ),
    )
    parser.add_argument(
        "--transition-frames",
        type=int,
        default=30,
        help=(
            "New views in each eased center/ellipse connector for "
            "staged-ellipse (default: 30)."
        ),
    )
    parser.add_argument(
        "--orbit-frames",
        type=int,
        default=160,
        help=(
            "New views used for one complete staged-ellipse orbit, excluding "
            "the initial ellipse point and including the returned point "
            "(default: 160)."
        ),
    )
    parser.add_argument(
        "--motion-scale",
        type=float,
        default=0.35,
        help=(
            "Fraction of the supported rail used on both sides of the test camera, "
            "in (0, 1] (default: 0.35 for a smooth 61-frame demo). "
            "Use 1 to sweep the full training-camera range."
        ),
    )
    parser.add_argument(
        "--rail-interpolation",
        choices=("linear", "smooth", "half-ellipse"),
        default="linear",
        help=(
            "Interpolate sweep positions/rotations piecewise linearly or with "
            "C2-continuous natural cubic curves, or use one analytic half-ellipse "
            "with a fixed training-ray look target (default: linear)."
        ),
    )
    parser.add_argument(
        "--sweep-bounds",
        choices=("aabb", "radial"),
        default="aabb",
        help=(
            "Validate sweep centers against the training-camera coordinate box "
            "or maximum distance from their mean center (default: aabb)."
        ),
    )
    parser.add_argument(
        "--sweep-pacing",
        choices=("minimum-jerk", "smooth-cruise"),
        default="minimum-jerk",
        help=(
            "Use minimum-jerk easing across each full sweep leg or smoother, "
            "lower-peak-speed cruise motion with C2 easing near turnarounds "
            "(default: minimum-jerk)."
        ),
    )
    parser.add_argument(
        "--station-angle-deg",
        type=float,
        default=3.0,
        help="Maximum azimuth separation used to group cameras at one rig station.",
    )
    parser.add_argument(
        "--time-mode",
        choices=("sweep", "fixed"),
        default="sweep",
        help=(
            "sweep spans the selected test camera's time range; fixed renders all "
            "novel views at one scene time."
        ),
    )
    parser.add_argument(
        "--fixed-time",
        type=float,
        default=None,
        help="Scene time for --time-mode fixed (default: selected start frame's time).",
    )
    parser.add_argument(
        "--intrinsics",
        choices=("fixed", "interpolate"),
        default="fixed",
        help="Keep the test camera intrinsics fixed, or interpolate calibrated rail intrinsics.",
    )
    parser.add_argument(
        "--image-extension",
        default=".jpg",
        help="Extension appended while validating extensionless frame file_path values.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser.parse_args()


def _relative_to_dataset(dataset, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else dataset / path


def _load_json(path):
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data.get("frames"), list) or not data["frames"]:
        raise ValueError(f"{path} has no non-empty 'frames' list")
    return data


def _camera_key(frame):
    stem = Path(str(frame["file_path"])).name
    prefix, separator, suffix = stem.rpartition("_")
    if separator and prefix and suffix.isdigit():
        return prefix
    return stem


def _frame_number(frame):
    stem = Path(str(frame["file_path"])).name
    _, separator, suffix = stem.rpartition("_")
    return int(suffix) if separator and suffix.isdigit() else None


def _select_frame_number_range(frames, first, last):
    numbered_frames = []
    for frame in frames:
        number = _frame_number(frame)
        if number is None:
            raise ValueError(
                f"Could not read a numeric frame suffix from {frame.get('file_path')}"
            )
        if first <= number <= last:
            numbered_frames.append((number, frame))

    numbered_frames.sort(key=lambda item: item[0])
    actual = [number for number, _ in numbered_frames]
    expected = list(range(first, last + 1))
    if actual != expected:
        raise ValueError(
            f"Requested timestamp frames {first}-{last}, but found {actual}"
        )
    return [frame for _, frame in numbered_frames]


def _group_frames_by_camera(frames):
    grouped = {}
    for frame in frames:
        grouped.setdefault(_camera_key(frame), []).append(frame)
    return grouped


def _pose(frame):
    matrix = np.asarray(frame["transform_matrix"], dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"Frame {frame.get('file_path')} does not contain a 4x4 transform_matrix")
    return matrix


def _intrinsics(frame, top_level):
    values = []
    for key in INTRINSIC_KEYS:
        if key in frame:
            values.append(float(frame[key]))
        elif key in top_level:
            values.append(float(top_level[key]))
        else:
            raise ValueError(f"Missing camera intrinsic '{key}'")
    return np.asarray(values, dtype=np.float64)


def _project_to_rotation(matrix):
    u, _, vt = np.linalg.svd(matrix)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0.0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def _rotation_to_quaternion(matrix):
    """Return a normalized [w, x, y, z] quaternion."""
    r = _project_to_rotation(matrix)
    trace = float(np.trace(r))
    if trace > 0.0:
        root = math.sqrt(trace + 1.0)
        w = 0.5 * root
        scale = 0.5 / root
        x = (r[2, 1] - r[1, 2]) * scale
        y = (r[0, 2] - r[2, 0]) * scale
        z = (r[1, 0] - r[0, 1]) * scale
    else:
        diagonal = np.diag(r)
        index = int(np.argmax(diagonal))
        if index == 0:
            root = math.sqrt(max(0.0, 1.0 + r[0, 0] - r[1, 1] - r[2, 2]))
            x = 0.5 * root
            scale = 0.5 / root
            w = (r[2, 1] - r[1, 2]) * scale
            y = (r[0, 1] + r[1, 0]) * scale
            z = (r[0, 2] + r[2, 0]) * scale
        elif index == 1:
            root = math.sqrt(max(0.0, 1.0 - r[0, 0] + r[1, 1] - r[2, 2]))
            y = 0.5 * root
            scale = 0.5 / root
            w = (r[0, 2] - r[2, 0]) * scale
            x = (r[0, 1] + r[1, 0]) * scale
            z = (r[1, 2] + r[2, 1]) * scale
        else:
            root = math.sqrt(max(0.0, 1.0 - r[0, 0] - r[1, 1] + r[2, 2]))
            z = 0.5 * root
            scale = 0.5 / root
            w = (r[1, 0] - r[0, 1]) * scale
            x = (r[0, 2] + r[2, 0]) * scale
            y = (r[1, 2] + r[2, 1]) * scale
    quaternion = np.asarray([w, x, y, z], dtype=np.float64)
    return quaternion / np.linalg.norm(quaternion)


def _quaternion_to_rotation(quaternion):
    w, x, y, z = quaternion / np.linalg.norm(quaternion)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _slerp(first, second, weight):
    q0 = first / np.linalg.norm(first)
    q1 = second / np.linalg.norm(second)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = float(np.clip(dot, -1.0, 1.0))
    if dot > 0.9995:
        result = q0 + weight * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    return (
        math.sin((1.0 - weight) * theta) / sin_theta * q0
        + math.sin(weight * theta) / sin_theta * q1
    )


def _natural_cubic_second_derivatives(values, cumulative):
    """Second derivatives for a vector-valued natural cubic spline."""
    values = np.asarray(values, dtype=np.float64)
    count = len(values)
    if count != len(cumulative) or count < 3:
        raise ValueError("Natural cubic interpolation requires at least three samples")

    intervals = np.diff(cumulative)
    system = np.zeros((count, count), dtype=np.float64)
    right_hand_side = np.zeros_like(values)
    system[0, 0] = 1.0
    system[-1, -1] = 1.0
    for index in range(1, count - 1):
        previous_interval = intervals[index - 1]
        next_interval = intervals[index]
        system[index, index - 1] = previous_interval
        system[index, index] = 2.0 * (previous_interval + next_interval)
        system[index, index + 1] = next_interval
        previous_slope = (values[index] - values[index - 1]) / previous_interval
        next_slope = (values[index + 1] - values[index]) / next_interval
        right_hand_side[index] = 6.0 * (next_slope - previous_slope)
    return np.linalg.solve(system, right_hand_side)


def _sample_natural_cubic(values, second_derivatives, cumulative, left, weight):
    """Evaluate a natural cubic spline inside rail segment ``left``."""
    right = left + 1
    segment_length = float(cumulative[right] - cumulative[left])
    first_weight = 1.0 - weight
    second_weight = weight
    return (
        first_weight * values[left]
        + second_weight * values[right]
        + (first_weight**3 - first_weight)
        * (segment_length * segment_length / 6.0)
        * second_derivatives[left]
        + (second_weight**3 - second_weight)
        * (segment_length * segment_length / 6.0)
        * second_derivatives[right]
    )


def _prepare_smooth_rail(rail, cumulative):
    """Precompute C2 natural-cubic position and quaternion curves."""
    positions = np.asarray([record["center"] for record in rail], dtype=np.float64)
    quaternions = np.asarray(
        [record["quaternion"] for record in rail],
        dtype=np.float64,
    )
    for index in range(1, len(quaternions)):
        if np.dot(quaternions[index - 1], quaternions[index]) < 0.0:
            quaternions[index] *= -1.0

    return {
        "positions": positions,
        "position_second_derivatives": _natural_cubic_second_derivatives(
            positions,
            cumulative,
        ),
        "quaternions": quaternions,
        "quaternion_second_derivatives": _natural_cubic_second_derivatives(
            quaternions,
            cumulative,
        ),
    }


def _prepare_half_ellipse_rail(
    rail,
    start_index,
    look_target,
    reference_up,
):
    """Fit one analytic half-ellipse through both rail ends and the test view."""
    first_center = rail[0]["center"]
    last_center = rail[-1]["center"]
    start_center = rail[start_index]["center"]
    ellipse_center = 0.5 * (first_center + last_center)
    cosine_axis = 0.5 * (first_center - last_center)
    start_offset = start_center - ellipse_center
    start_cosine = float(
        np.clip(
            np.dot(start_offset, cosine_axis) / np.dot(cosine_axis, cosine_axis),
            -1.0,
            1.0,
        )
    )
    start_theta = math.acos(start_cosine)
    start_sine = math.sin(start_theta)
    if abs(start_sine) <= 1.0e-8:
        raise ValueError("The test camera is too close to a half-ellipse endpoint")

    sine_axis = (start_offset - start_cosine * cosine_axis) / start_sine
    plane_normal = _normalized(
        np.cross(cosine_axis, sine_axis),
        "half-ellipse plane normal",
    )
    axis_matrix = np.column_stack((cosine_axis, sine_axis))
    axis_lengths = np.linalg.svd(axis_matrix, compute_uv=False)
    if axis_lengths[-1] <= 1.0e-8:
        raise ValueError("The fitted half-ellipse is degenerate")

    theta_samples = np.unique(
        np.concatenate(
            (
                np.linspace(0.0, math.pi, 16385, dtype=np.float64),
                [start_theta],
            )
        )
    )
    sampled_centers = (
        ellipse_center
        + np.cos(theta_samples)[:, None] * cosine_axis
        + np.sin(theta_samples)[:, None] * sine_axis
    )
    arc_lengths = np.concatenate(
        (
            [0.0],
            np.cumsum(np.linalg.norm(np.diff(sampled_centers, axis=0), axis=1)),
        )
    )
    start_sample_index = int(np.where(theta_samples == start_theta)[0][0])

    return {
        "center": ellipse_center,
        "cosine_axis": cosine_axis,
        "sine_axis": sine_axis,
        "plane_normal": plane_normal,
        "axis_lengths": axis_lengths,
        "axis_angle_deg": math.degrees(
            math.acos(
                float(
                    np.clip(
                        np.dot(cosine_axis, sine_axis)
                        / (np.linalg.norm(cosine_axis) * np.linalg.norm(sine_axis)),
                        -1.0,
                        1.0,
                    )
                )
            )
        ),
        "coordinate_min": 0.0,
        "coordinate_max": float(arc_lengths[-1]),
        "start_coordinate": float(arc_lengths[start_sample_index]),
        "start_theta": start_theta,
        "start_center": start_center.copy(),
        "first_center": first_center.copy(),
        "last_center": last_center.copy(),
        "theta_samples": theta_samples,
        "arc_lengths": arc_lengths,
        "look_target": np.asarray(look_target, dtype=np.float64),
        "reference_up": _normalized(reference_up, "half-ellipse reference up"),
        "endpoint_cameras": (rail[0]["camera_key"], rail[-1]["camera_key"]),
    }


def _half_ellipse_center(half_ellipse, coordinate):
    theta = float(
        np.interp(
            float(coordinate),
            half_ellipse["arc_lengths"],
            half_ellipse["theta_samples"],
        )
    )
    return (
        half_ellipse["center"]
        + math.cos(theta) * half_ellipse["cosine_axis"]
        + math.sin(theta) * half_ellipse["sine_axis"]
    )


def _sample_half_ellipse(half_ellipse, coordinate):
    center = _half_ellipse_center(half_ellipse, coordinate)
    return _look_at_matrix(
        center,
        half_ellipse["look_target"],
        half_ellipse["reference_up"],
    )


def _estimate_world_up(records):
    ups = [record["matrix"][:3, 1].copy() for record in records]
    reference = ups[0]
    aligned = [up if np.dot(up, reference) >= 0.0 else -up for up in ups]
    world_up = np.mean(aligned, axis=0)
    norm = np.linalg.norm(world_up)
    if norm < 1.0e-8:
        raise ValueError("Could not infer a stable world-up direction from camera poses")
    return world_up / norm


def _estimate_look_target(records):
    lhs = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    for record in records:
        matrix = record["matrix"]
        center = matrix[:3, 3]
        forward = -matrix[:3, 2]
        forward /= np.linalg.norm(forward)
        projector = np.eye(3) - np.outer(forward, forward)
        lhs += projector
        rhs += projector @ center
    target, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)
    return target


def _make_record(camera_key, frame, top_level, is_start=False):
    matrix = _pose(frame).copy()
    return {
        "camera_key": camera_key,
        "frame": frame,
        "matrix": matrix,
        "center": matrix[:3, 3].copy(),
        "quaternion": _rotation_to_quaternion(matrix[:3, :3]),
        "intrinsics": _intrinsics(frame, top_level),
        "is_start": is_start,
    }


def _representative_records(transform_json):
    records = []
    for camera_key, frames in _group_frames_by_camera(transform_json["frames"]).items():
        records.append(_make_record(camera_key, frames[0], transform_json))
    return records


def _build_camera_rail(train_json, start_record, station_angle_deg):
    train_records = _representative_records(train_json)
    geometry_records = train_records + [start_record]
    world_up = _estimate_world_up(geometry_records)
    target = _estimate_look_target(geometry_records)

    start_radial = start_record["center"] - target
    start_radial -= world_up * np.dot(start_radial, world_up)
    start_radial /= np.linalg.norm(start_radial)
    tangent = np.cross(world_up, start_radial)
    tangent /= np.linalg.norm(tangent)

    for record in train_records:
        radial = record["center"] - target
        radial -= world_up * np.dot(radial, world_up)
        radial /= np.linalg.norm(radial)
        record["angle"] = math.atan2(float(np.dot(radial, tangent)), float(np.dot(radial, start_radial)))
        record["elevation"] = float(np.dot(record["center"] - target, world_up))
    start_record["angle"] = 0.0
    start_record["elevation"] = float(np.dot(start_record["center"] - target, world_up))

    threshold = math.radians(station_angle_deg)
    ordered = sorted(train_records, key=lambda record: record["angle"])
    station_groups = []
    for record in ordered:
        if not station_groups or record["angle"] - station_groups[-1][-1]["angle"] > threshold:
            station_groups.append([record])
        else:
            station_groups[-1].append(record)

    reference_group = min(
        station_groups,
        key=lambda group: min(abs(record["angle"]) for record in group),
    )
    elevation_tolerance = 1.0e-6
    below_start = sum(
        record["elevation"] < start_record["elevation"] - elevation_tolerance
        for record in reference_group
    )
    above_start = sum(
        record["elevation"] > start_record["elevation"] + elevation_tolerance
        for record in reference_group
    )
    comparable_tiers = below_start + above_start
    tier_fraction = below_start / comparable_tiers if comparable_tiers else 0.5

    rail = []
    inserted_start = False
    for group in station_groups:
        if min(abs(record["angle"]) for record in group) <= threshold:
            rail.append(start_record)
            inserted_start = True
            continue
        elevation_order = sorted(group, key=lambda record: record["elevation"])
        tier_index = int(math.floor(tier_fraction * (len(elevation_order) - 1) + 0.5))
        rail.append(elevation_order[tier_index])
    if not inserted_start:
        rail.append(start_record)

    rail.sort(key=lambda record: record["angle"])
    if len(rail) < 3:
        raise ValueError("Could not infer at least three camera stations for a demo rail")
    start_indices = [index for index, record in enumerate(rail) if record["is_start"]]
    if len(start_indices) != 1:
        raise ValueError("Could not insert the test camera exactly once into the camera rail")
    start_index = start_indices[0]
    if start_index in (0, len(rail) - 1):
        raise ValueError("The selected test camera is not inside the supported training-camera arc")

    segment_lengths = [
        float(np.linalg.norm(rail[index + 1]["center"] - rail[index]["center"]))
        for index in range(len(rail) - 1)
    ]
    if min(segment_lengths) <= 1.0e-8:
        raise ValueError("Inferred camera rail contains duplicate adjacent positions")
    cumulative = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    return rail, cumulative, start_index, target, world_up


def _normalized(vector, description):
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-10:
        raise ValueError(f"Could not infer a stable {description}")
    return np.asarray(vector, dtype=np.float64) / norm


def _look_at_matrix(center, look_target, reference_up):
    """Create an OpenGL camera-to-world pose aimed at ``look_target``."""
    center = np.asarray(center, dtype=np.float64)
    forward = _normalized(look_target - center, "camera viewing direction")
    right = _normalized(np.cross(forward, reference_up), "camera-right direction")
    up = _normalized(np.cross(right, forward), "camera-up direction")

    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.column_stack((right, up, -forward))
    matrix[:3, 3] = center
    return matrix


def _ellipse_matrices(
    start_record,
    target,
    world_up,
    rail,
    start_index,
    num_frames,
    scale,
    plane_mode="rail",
    requested_horizontal_radius=None,
    requested_vertical_radius=None,
):
    """Build a periodic ellipse centered at the selected test-camera position.

    By default, a chord through neighboring training stations supplies the
    horizontal axis and world-up supplies the vertical axis.  ``camera`` plane
    mode instead uses the exact test-camera image plane, which can provide more
    usable motion when the calibrated rail is tilted near a coordinate bound.
    """
    if num_frames < 4:
        raise ValueError("An ellipse requires at least four output frames")

    start_matrix = start_record["matrix"]
    start_center = start_record["center"]
    start_rotation = _project_to_rotation(start_matrix[:3, :3])
    start_forward = _normalized(-start_rotation[:, 2], "test-camera forward direction")
    reference_up = _normalized(start_rotation[:, 1], "test-camera up direction")

    if plane_mode == "camera":
        horizontal = _normalized(start_rotation[:, 0], "test-camera right direction")
        vertical = start_rotation[:, 1] - horizontal * float(
            np.dot(start_rotation[:, 1], horizontal)
        )
        vertical = _normalized(vertical, "test-camera image-plane vertical direction")
    elif plane_mode == "rail":
        horizontal = rail[start_index + 1]["center"] - rail[start_index - 1]["center"]
        horizontal -= world_up * float(np.dot(horizontal, world_up))
        horizontal = _normalized(horizontal, "local training-camera chord")
        if np.dot(horizontal, start_rotation[:, 0]) < 0.0:
            horizontal = -horizontal
        vertical = world_up - horizontal * float(np.dot(world_up, horizontal))
        vertical = _normalized(vertical, "ellipse vertical direction")
    else:
        raise ValueError(f"Unsupported ellipse plane mode: {plane_mode}")
    plane_normal = _normalized(np.cross(horizontal, vertical), "ellipse plane normal")
    if np.dot(plane_normal, start_forward) < 0.0:
        plane_normal = -plane_normal

    base_horizontal_radius = (
        ELLIPSE_HORIZONTAL_RADIUS
        if requested_horizontal_radius is None
        else float(requested_horizontal_radius)
    )
    base_vertical_radius = (
        ELLIPSE_VERTICAL_RADIUS
        if requested_vertical_radius is None
        else float(requested_vertical_radius)
    )
    horizontal_radius = base_horizontal_radius * scale
    vertical_radius = base_vertical_radius * scale
    ellipse_center = start_center.copy()

    # Keep the test camera's original optical-axis depth while looking at one
    # fixed point so that orientation changes continuously around the orbit.
    optical_depth = float(np.dot(target - start_center, start_forward))
    if optical_depth <= 1.0e-8:
        raise ValueError("The inferred scene target is behind the selected test camera")
    optical_target = start_center + optical_depth * start_forward

    matrices = []
    thetas = 2.0 * math.pi * np.arange(num_frames, dtype=np.float64) / num_frames
    for theta in thetas:
        center = (
            ellipse_center
            + horizontal_radius * math.cos(float(theta)) * horizontal
            + vertical_radius * math.sin(float(theta)) * vertical
        )
        forward = _normalized(optical_target - center, "ellipse viewing direction")
        right = _normalized(np.cross(forward, reference_up), "ellipse camera-right direction")
        up = _normalized(np.cross(right, forward), "ellipse camera-up direction")

        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = np.column_stack((right, up, -forward))
        matrix[:3, 3] = center
        matrices.append(matrix)

    return matrices, {
        "center": ellipse_center,
        "horizontal": horizontal,
        "vertical": vertical,
        "plane_normal": plane_normal,
        "plane_angle_deg": math.degrees(
            math.acos(float(np.clip(np.dot(plane_normal, start_forward), -1.0, 1.0)))
        ),
        "look_target": optical_target,
        "horizontal_radius": horizontal_radius,
        "vertical_radius": vertical_radius,
        "plane_mode": plane_mode,
        "support_cameras": (
            rail[start_index - 1]["camera_key"],
            rail[start_index + 1]["camera_key"],
        ),
    }


def _staged_ellipse_matrices(
    start_record,
    target,
    world_up,
    rail,
    start_index,
    transition_frames,
    orbit_frames,
    scale,
):
    """Move center -> ellipse -> center with zero-speed eased joins.

    Returned views exclude the initial center pose.  The outward connector
    includes the ellipse start, the orbit excludes that initial sample but
    includes the same point after one revolution, and the inward connector
    includes the exact original center pose.
    """
    _, ellipse_info = _ellipse_matrices(
        start_record,
        target,
        world_up,
        rail,
        start_index,
        orbit_frames,
        scale,
    )
    center = start_record["center"]
    reference_up = _normalized(
        _project_to_rotation(start_record["matrix"][:3, :3])[:, 1],
        "test-camera up direction",
    )
    horizontal = ellipse_info["horizontal"]
    vertical = ellipse_info["vertical"]
    horizontal_radius = ellipse_info["horizontal_radius"]
    vertical_radius = ellipse_info["vertical_radius"]
    look_target = ellipse_info["look_target"]
    ellipse_start_center = center + horizontal_radius * horizontal
    ellipse_start_matrix = _look_at_matrix(
        ellipse_start_center,
        look_target,
        reference_up,
    )

    outward = []
    for index in range(1, transition_frames + 1):
        amount = _smootherstep(index / float(transition_frames))
        camera_center = center + amount * (ellipse_start_center - center)
        outward.append(_look_at_matrix(camera_center, look_target, reference_up))
    outward[-1] = ellipse_start_matrix.copy()

    orbit = []
    for index in range(1, orbit_frames + 1):
        theta = 2.0 * math.pi * _smootherstep(index / float(orbit_frames))
        camera_center = (
            center
            + horizontal_radius * math.cos(theta) * horizontal
            + vertical_radius * math.sin(theta) * vertical
        )
        orbit.append(_look_at_matrix(camera_center, look_target, reference_up))
    orbit[-1] = ellipse_start_matrix.copy()

    inward = []
    for index in range(1, transition_frames + 1):
        amount = _smootherstep(index / float(transition_frames))
        camera_center = ellipse_start_center + amount * (center - ellipse_start_center)
        inward.append(_look_at_matrix(camera_center, look_target, reference_up))
    inward[-1] = start_record["matrix"].copy()

    ellipse_info["ellipse_start_center"] = ellipse_start_center
    ellipse_info["ellipse_start_matrix"] = ellipse_start_matrix
    return outward, orbit, inward, ellipse_info


def _allocate_intervals(total_intervals, weights):
    if total_intervals < len(weights):
        raise ValueError(f"At least {len(weights) + 1} output frames are required")
    weights = np.asarray(weights, dtype=np.float64)
    if np.any(weights <= 0.0):
        raise ValueError("All trajectory legs must have positive length")
    remaining = total_intervals - len(weights)
    raw_extra = weights / weights.sum() * remaining
    counts = np.ones(len(weights), dtype=np.int64) + np.floor(raw_extra).astype(np.int64)
    leftovers = total_intervals - int(counts.sum())
    remainders = raw_extra - np.floor(raw_extra)
    for index in np.argsort(-remainders)[:leftovers]:
        counts[index] += 1
    return counts


def _smootherstep(value):
    return value * value * value * (value * (value * 6.0 - 15.0) + 10.0)


def _integrated_smootherstep(value):
    """Integral of smootherstep from zero to ``value``."""
    return value**6 - 3.0 * value**5 + 2.5 * value**4


def _smooth_cruise_progress(value, ease_fraction=SWEEP_CRUISE_EASE_FRACTION):
    """C2-eased motion with a constant-speed middle section."""
    if not 0.0 < ease_fraction < 0.5:
        raise ValueError("Sweep cruise easing fraction must be in (0, 0.5)")
    cruise_speed = 1.0 / (1.0 - ease_fraction)
    if value < ease_fraction:
        local = value / ease_fraction
        return cruise_speed * ease_fraction * _integrated_smootherstep(local)
    if value > 1.0 - ease_fraction:
        local = (1.0 - value) / ease_fraction
        return 1.0 - cruise_speed * ease_fraction * _integrated_smootherstep(local)
    return cruise_speed * (value - 0.5 * ease_fraction)


def _paced_progress(fraction, pacing):
    if pacing == "minimum-jerk":
        return _smootherstep(fraction)
    if pacing == "smooth-cruise":
        return _smooth_cruise_progress(fraction)
    raise ValueError(f"Unsupported sweep pacing mode: {pacing}")


def _trajectory_coordinates(
    cumulative,
    start_index,
    num_frames,
    motion_scale,
    pacing="minimum-jerk",
):
    start = float(cumulative[start_index])
    low = start + motion_scale * (float(cumulative[0]) - start)
    high = start + motion_scale * (float(cumulative[-1]) - start)
    endpoints = ((start, low), (low, high), (high, start))
    weights = [abs(second - first) for first, second in endpoints]
    counts = _allocate_intervals(num_frames - 1, weights)

    coordinates = []
    reversal_indices = []
    for leg_index, ((first, second), count) in enumerate(zip(endpoints, counts)):
        begin = 0 if leg_index == 0 else 1
        for sample_index in range(begin, int(count) + 1):
            fraction = sample_index / float(count)
            progress = _paced_progress(fraction, pacing)
            coordinates.append(first + (second - first) * progress)
        reversal_indices.append(len(coordinates) - 1)
    if len(coordinates) != num_frames:
        raise AssertionError(f"Internal trajectory sample count mismatch: {len(coordinates)} != {num_frames}")
    return np.asarray(coordinates), counts.tolist(), reversal_indices


def _open_trajectory_coordinates(first, last, num_frames, pacing):
    if num_frames < 2:
        raise ValueError("An open trajectory requires at least two output frames")
    coordinates = []
    for index in range(num_frames):
        fraction = index / float(num_frames - 1)
        progress = _paced_progress(fraction, pacing)
        coordinates.append(first + (last - first) * progress)
    return np.asarray(coordinates, dtype=np.float64)


def _sample_rail(
    rail,
    cumulative,
    coordinate,
    interpolate_intrinsics,
    smooth_rail=None,
):
    distances = np.abs(cumulative - coordinate)
    exact_index = int(np.argmin(distances))
    if distances[exact_index] <= 1.0e-10:
        record = rail[exact_index]
        intrinsics = record["intrinsics"].copy() if interpolate_intrinsics else None
        return record["matrix"].copy(), intrinsics

    left = int(np.searchsorted(cumulative, coordinate, side="right") - 1)
    left = min(max(left, 0), len(rail) - 2)
    right = left + 1
    weight = float((coordinate - cumulative[left]) / (cumulative[right] - cumulative[left]))
    first = rail[left]
    second = rail[right]

    matrix = np.eye(4, dtype=np.float64)
    if smooth_rail is None:
        quaternion = _slerp(first["quaternion"], second["quaternion"], weight)
        center = (1.0 - weight) * first["center"] + weight * second["center"]
    else:
        center = _sample_natural_cubic(
            smooth_rail["positions"],
            smooth_rail["position_second_derivatives"],
            cumulative,
            left,
            weight,
        )
        quaternion = _sample_natural_cubic(
            smooth_rail["quaternions"],
            smooth_rail["quaternion_second_derivatives"],
            cumulative,
            left,
            weight,
        )
        quaternion /= np.linalg.norm(quaternion)
    matrix[:3, :3] = _quaternion_to_rotation(quaternion)
    matrix[:3, 3] = center
    if interpolate_intrinsics:
        intrinsics = (1.0 - weight) * first["intrinsics"] + weight * second["intrinsics"]
    else:
        intrinsics = None
    return matrix, intrinsics


def _nearest_frame(frames, target_time):
    return min(frames, key=lambda frame: abs(float(frame.get("time", 0.0)) - target_time))


def _output_times(test_camera_frames, num_frames, time_mode, fixed_time, start_frame):
    source_times = np.asarray(
        [float(frame.get("time", 0.0)) for frame in test_camera_frames],
        dtype=np.float64,
    )
    if time_mode == "fixed":
        value = float(start_frame.get("time", 0.0)) if fixed_time is None else float(fixed_time)
        return np.full(num_frames, value, dtype=np.float64)
    if fixed_time is not None:
        raise ValueError("--fixed-time is only valid with --time-mode fixed")
    return np.linspace(float(source_times.min()), float(source_times.max()), num_frames)


def _frame_image_path(dataset, frame, extension):
    image_path = Path(str(frame["file_path"]))
    if not image_path.suffix:
        image_path = image_path.with_suffix(extension)
    return image_path if image_path.is_absolute() else dataset / image_path


def _rotation_step_degrees(first, second):
    relative = _project_to_rotation(first).T @ _project_to_rotation(second)
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def _validate_output(
    output_json,
    train_json,
    start_frame,
    dataset,
    image_extension,
    trajectory,
    ellipse_info=None,
    bounds_mode="aabb",
    exact_test_pose=True,
):
    frames = output_json["frames"]
    periodic = trajectory == "ellipse"
    if trajectory in ("sweep", "staged-ellipse"):
        if exact_test_pose:
            if frames[0]["transform_matrix"] != start_frame["transform_matrix"]:
                raise AssertionError("The first generated pose is not an exact copy of the test pose")
            if frames[-1]["transform_matrix"] != start_frame["transform_matrix"]:
                raise AssertionError("The closed trajectory does not return exactly to the test pose")
        elif exact_test_pose is False:
            start_center = _pose(start_frame)[:3, 3]
            first_matrix = _pose(frames[0])
            last_matrix = _pose(frames[-1])
            if np.linalg.norm(first_matrix[:3, 3] - start_center) > 1.0e-8:
                raise AssertionError("The first generated center is not the test-camera center")
            if not np.allclose(first_matrix, last_matrix, atol=1.0e-12, rtol=0.0):
                raise AssertionError("The closed trajectory does not return to its first pose")
    for key in INTRINSIC_KEYS:
        if frames[0].get(key, output_json.get(key)) != start_frame.get(key, output_json.get(key)):
            raise AssertionError(f"The first generated intrinsic '{key}' differs from the test camera")

    matrices = np.asarray([frame["transform_matrix"] for frame in frames], dtype=np.float64)
    rotations = matrices[:, :3, :3]
    centers = matrices[:, :3, 3]
    for index, rotation in enumerate(rotations):
        orthogonality_error = np.max(np.abs(rotation.T @ rotation - np.eye(3)))
        determinant_error = abs(float(np.linalg.det(rotation)) - 1.0)
        if orthogonality_error > 1.0e-5 or determinant_error > 1.0e-5:
            raise AssertionError(
                f"Generated frame {index} has an invalid rotation "
                f"(orthogonality={orthogonality_error:.3g}, det_error={determinant_error:.3g})"
            )

    plane_error = 0.0
    ellipse_error = 0.0
    if periodic:
        if ellipse_info is None:
            raise AssertionError("Ellipse metadata is required for periodic validation")
        if np.allclose(matrices[0], matrices[-1], atol=1.0e-12, rtol=0.0):
            raise AssertionError("Periodic ellipse contains a duplicated endpoint")

        expected_center = _pose(start_frame)[:3, 3]
        mean_center_error = float(np.linalg.norm(centers.mean(axis=0) - expected_center))
        if mean_center_error > 1.0e-8:
            raise AssertionError(
                "The ellipse is not centered at the test camera "
                f"(center error={mean_center_error:.3g})"
            )

        offsets = centers - expected_center
        plane_error = float(np.max(np.abs(offsets @ ellipse_info["plane_normal"])))
        if plane_error > 1.0e-8:
            raise AssertionError(
                f"Ellipse centers leave their inferred plane (error={plane_error:.3g})"
            )
        horizontal_coordinates = (
            offsets @ ellipse_info["horizontal"] / ellipse_info["horizontal_radius"]
        )
        vertical_coordinates = (
            offsets @ ellipse_info["vertical"] / ellipse_info["vertical_radius"]
        )
        ellipse_error = float(
            np.max(
                np.abs(
                    horizontal_coordinates * horizontal_coordinates
                    + vertical_coordinates * vertical_coordinates
                    - 1.0
                )
            )
        )
        if ellipse_error > 1.0e-8:
            raise AssertionError(
                f"Generated centers do not lie on the requested ellipse (error={ellipse_error:.3g})"
            )

    train_centers = np.asarray(
        [_pose(frame)[:3, 3] for frame in train_json["frames"]],
        dtype=np.float64,
    )
    generated_radial_max = 0.0
    training_radial_max = 0.0
    if bounds_mode == "aabb":
        lower = train_centers.min(axis=0) - 1.0e-8
        upper = train_centers.max(axis=0) + 1.0e-8
        outside = np.where(np.any((centers < lower) | (centers > upper), axis=1))[0]
        if outside.size:
            raise AssertionError(
                f"Generated camera center {int(outside[0])} leaves the "
                "training-camera coordinate bounds"
            )
    elif bounds_mode == "radial":
        unique_train_centers = np.unique(train_centers, axis=0)
        training_center = unique_train_centers.mean(axis=0)
        training_radial_max = float(
            np.max(np.linalg.norm(unique_train_centers - training_center, axis=1))
        )
        generated_distances = np.linalg.norm(centers - training_center, axis=1)
        generated_radial_max = float(np.max(generated_distances))
        outside = np.where(generated_distances > training_radial_max + 1.0e-8)[0]
        if outside.size:
            raise AssertionError(
                f"Generated camera center {int(outside[0])} lies beyond the "
                f"training radial range ({generated_radial_max:.6g} > "
                f"{training_radial_max:.6g})"
            )
    else:
        raise ValueError(f"Unsupported training-camera bounds mode: {bounds_mode}")

    train_times = np.asarray(
        [float(frame.get("time", 0.0)) for frame in train_json["frames"]],
        dtype=np.float64,
    )
    output_times = np.asarray(
        [float(frame.get("time", 0.0)) for frame in frames],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(output_times)):
        raise AssertionError("Generated frame times must all be finite")
    time_tolerance = 1.0e-8
    if (
        output_times.min() < train_times.min() - time_tolerance
        or output_times.max() > train_times.max() + time_tolerance
    ):
        raise AssertionError(
            "Generated times leave the training range "
            f"[{train_times.min():.10g}, {train_times.max():.10g}]"
        )

    missing_images = []
    for frame in frames:
        path = _frame_image_path(dataset, frame, image_extension)
        if not path.is_file():
            missing_images.append(str(path))
            if len(missing_images) == 3:
                break
    if missing_images:
        raise FileNotFoundError("Generated frames reference missing images: " + ", ".join(missing_images))

    if periodic:
        position_steps = np.linalg.norm(np.roll(centers, -1, axis=0) - centers, axis=1)
        rotation_steps = [
            _rotation_step_degrees(rotations[index], rotations[(index + 1) % len(rotations)])
            for index in range(len(rotations))
        ]
    else:
        position_steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        rotation_steps = [
            _rotation_step_degrees(rotations[index], rotations[index + 1])
            for index in range(len(rotations) - 1)
        ]
    return {
        "center_min": centers.min(axis=0),
        "center_max": centers.max(axis=0),
        "max_position_step": float(position_steps.max(initial=0.0)),
        "max_rotation_step_deg": float(max(rotation_steps, default=0.0)),
        "seam_position_step": (
            float(np.linalg.norm(centers[0] - centers[-1])) if periodic else 0.0
        ),
        "seam_rotation_step_deg": (
            _rotation_step_degrees(rotations[-1], rotations[0]) if periodic else 0.0
        ),
        "plane_error": plane_error,
        "ellipse_error": ellipse_error,
        "bounds_mode": bounds_mode,
        "generated_radial_max": generated_radial_max,
        "training_radial_max": training_radial_max,
    }


def _validate_half_ellipse_output(output_json, coordinates, half_ellipse):
    matrices = np.asarray(
        [frame["transform_matrix"] for frame in output_json["frames"]],
        dtype=np.float64,
    )
    centers = matrices[:, :3, 3]
    expected_centers = np.asarray(
        [_half_ellipse_center(half_ellipse, coordinate) for coordinate in coordinates]
    )
    center_error = float(
        np.max(np.linalg.norm(centers - expected_centers, axis=1))
    )
    if center_error > 1.0e-8:
        raise AssertionError(
            f"Half-ellipse centers differ from the analytic path (error={center_error:.3g})"
        )
    endpoint_error = max(
        float(np.linalg.norm(centers[0] - half_ellipse["first_center"])),
        float(np.linalg.norm(centers[-1] - half_ellipse["last_center"])),
    )
    if endpoint_error > 1.0e-8:
        raise AssertionError(
            f"Half-ellipse endpoints miss their training cameras (error={endpoint_error:.3g})"
        )

    offsets = centers - half_ellipse["center"]
    plane_error = float(
        np.max(np.abs(offsets @ half_ellipse["plane_normal"]))
    )
    coefficients = np.linalg.lstsq(
        np.column_stack(
            (half_ellipse["cosine_axis"], half_ellipse["sine_axis"])
        ),
        offsets.T,
        rcond=None,
    )[0].T
    ellipse_error = float(
        np.max(np.abs(np.sum(coefficients * coefficients, axis=1) - 1.0))
    )
    if plane_error > 1.0e-8 or ellipse_error > 1.0e-8:
        raise AssertionError(
            "Generated sweep leaves its half-ellipse "
            f"(plane={plane_error:.3g}, ellipse={ellipse_error:.3g})"
        )

    desired_forwards = np.asarray(
        [
            _normalized(
                half_ellipse["look_target"] - center,
                "half-ellipse viewing direction",
            )
            for center in centers
        ]
    )
    actual_forwards = -matrices[:, :3, 2]
    look_error = float(
        np.max(np.linalg.norm(actual_forwards - desired_forwards, axis=1))
    )
    if look_error > 1.0e-8:
        raise AssertionError(
            f"A half-ellipse camera misses the fixed look target (error={look_error:.3g})"
        )

    up_alignment = matrices[:, :3, 1] @ half_ellipse["reference_up"]
    if np.any(up_alignment <= 0.0):
        raise AssertionError("A half-ellipse camera contains an up-vector flip")

    return {
        "center_error": center_error,
        "endpoint_error": endpoint_error,
        "plane_error": plane_error,
        "ellipse_error": ellipse_error,
        "look_error": look_error,
        "minimum_up_alignment": float(up_alignment.min()),
    }


def _validate_staged_ellipse_output(
    output_json,
    source_frames,
    freeze_index,
    ellipse_info,
    transition_frames,
    orbit_frames,
):
    frames = output_json["frames"]
    lead_count = freeze_index + 1
    tail_count = len(source_frames) - lead_count
    motion_count = 2 * transition_frames + orbit_frames
    expected_count = len(source_frames) + motion_count
    if len(frames) != expected_count:
        raise AssertionError(
            f"Staged trajectory has {len(frames)} frames; expected {expected_count}"
        )

    freeze_frame = source_frames[freeze_index]
    expected_sources = (
        source_frames[:lead_count]
        + [freeze_frame] * motion_count
        + source_frames[lead_count:]
    )
    actual_numbers = [_frame_number(frame) for frame in frames]
    expected_numbers = [_frame_number(frame) for frame in expected_sources]
    if actual_numbers != expected_numbers:
        raise AssertionError("Staged trajectory does not follow the requested source-frame schedule")
    actual_paths = [str(frame["file_path"]) for frame in frames]
    expected_paths = [str(frame["file_path"]) for frame in expected_sources]
    if actual_paths != expected_paths:
        raise AssertionError("Staged trajectory does not reuse the expected source images")
    actual_times = [float(frame.get("time", 0.0)) for frame in frames]
    expected_times = [float(frame.get("time", 0.0)) for frame in expected_sources]
    if actual_times != expected_times:
        raise AssertionError("Staged trajectory timestamps do not match the requested freeze schedule")

    matrices = np.asarray([frame["transform_matrix"] for frame in frames], dtype=np.float64)
    centers = matrices[:, :3, 3]
    test_center = np.asarray(ellipse_info["center"], dtype=np.float64)
    horizontal = np.asarray(ellipse_info["horizontal"], dtype=np.float64)
    vertical = np.asarray(ellipse_info["vertical"], dtype=np.float64)
    horizontal_radius = float(ellipse_info["horizontal_radius"])
    vertical_radius = float(ellipse_info["vertical_radius"])
    ellipse_start = np.asarray(ellipse_info["ellipse_start_center"], dtype=np.float64)

    moving_start = lead_count
    orbit_start = moving_start + transition_frames
    inward_start = orbit_start + orbit_frames
    tail_start = inward_start + transition_frames

    stationary_indices = list(range(lead_count)) + list(range(tail_start, len(frames)))
    stationary_error = float(
        np.max(
            np.linalg.norm(centers[stationary_indices] - test_center, axis=1),
            initial=0.0,
        )
    )
    if stationary_error > 1.0e-10:
        raise AssertionError(
            f"A staged temporal view leaves the test camera (error={stationary_error:.3g})"
        )

    expected_moving_centers = []
    for index in range(1, transition_frames + 1):
        amount = _smootherstep(index / float(transition_frames))
        expected_moving_centers.append(test_center + amount * (ellipse_start - test_center))
    for index in range(1, orbit_frames + 1):
        theta = 2.0 * math.pi * _smootherstep(index / float(orbit_frames))
        expected_moving_centers.append(
            test_center
            + horizontal_radius * math.cos(theta) * horizontal
            + vertical_radius * math.sin(theta) * vertical
        )
    expected_moving_centers[transition_frames + orbit_frames - 1] = ellipse_start.copy()
    for index in range(1, transition_frames + 1):
        amount = _smootherstep(index / float(transition_frames))
        expected_moving_centers.append(ellipse_start + amount * (test_center - ellipse_start))
    expected_moving_centers[-1] = test_center.copy()
    expected_moving_centers = np.asarray(expected_moving_centers)
    actual_moving_centers = centers[moving_start:tail_start]
    center_error = float(
        np.max(np.linalg.norm(actual_moving_centers - expected_moving_centers, axis=1))
    )
    if center_error > 1.0e-9:
        raise AssertionError(
            f"Staged motion differs from its requested path (error={center_error:.3g})"
        )

    orbit_centers = centers[orbit_start:inward_start]
    orbit_offsets = orbit_centers - test_center
    orbit_plane_error = float(
        np.max(np.abs(orbit_offsets @ ellipse_info["plane_normal"]))
    )
    horizontal_coordinates = orbit_offsets @ horizontal / horizontal_radius
    vertical_coordinates = orbit_offsets @ vertical / vertical_radius
    orbit_ellipse_error = float(
        np.max(
            np.abs(
                horizontal_coordinates * horizontal_coordinates
                + vertical_coordinates * vertical_coordinates
                - 1.0
            )
        )
    )
    if orbit_plane_error > 1.0e-8 or orbit_ellipse_error > 1.0e-8:
        raise AssertionError(
            "The staged orbit leaves its ellipse "
            f"(plane={orbit_plane_error:.3g}, ellipse={orbit_ellipse_error:.3g})"
        )

    moving_matrices = matrices[moving_start:tail_start]
    desired_forwards = np.asarray(
        [
            _normalized(ellipse_info["look_target"] - center, "staged viewing direction")
            for center in actual_moving_centers
        ]
    )
    actual_forwards = -moving_matrices[:, :3, 2]
    look_error = float(np.max(np.linalg.norm(actual_forwards - desired_forwards, axis=1)))
    if look_error > 1.0e-6:
        raise AssertionError(
            f"A staged camera does not face the fixed optical target (error={look_error:.3g})"
        )

    moving_with_center = centers[lead_count - 1 : tail_start]
    moving_steps = np.linalg.norm(np.diff(moving_with_center, axis=0), axis=1)
    if np.any(moving_steps <= 1.0e-12):
        duplicate = int(np.where(moving_steps <= 1.0e-12)[0][0] + lead_count - 1)
        raise AssertionError(
            f"Staged camera motion contains adjacent duplicate poses at frame {duplicate}"
        )

    return {
        "lead_count": lead_count,
        "tail_count": tail_count,
        "motion_count": motion_count,
        "moving_start": moving_start,
        "orbit_start": orbit_start,
        "inward_start": inward_start,
        "tail_start": tail_start,
        "stationary_error": stationary_error,
        "center_error": center_error,
        "orbit_plane_error": orbit_plane_error,
        "orbit_ellipse_error": orbit_ellipse_error,
        "look_error": look_error,
    }


def _atomic_write_json(path, contents):
    path.parent.mkdir(parents=True, exist_ok=True)
    output_mode = (path.stat().st_mode & 0o777) if path.exists() else 0o644
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(contents, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.chmod(temporary_name, output_mode)
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main():
    args = _parse_args()
    dataset = args.dataset.expanduser().resolve()
    train_path = _relative_to_dataset(dataset, args.train_file).resolve()
    test_path = _relative_to_dataset(dataset, args.test_file).resolve()
    default_outputs = {
        "sweep": "transforms_demo.json",
        "ellipse": "transforms_demo_ellipse.json",
        "staged-ellipse": "transforms_demo_staged_ellipse.json",
    }
    default_output = default_outputs[args.trajectory]
    output_path = _relative_to_dataset(dataset, args.output or default_output).resolve()

    if not 0.0 < args.motion_scale <= 1.0:
        raise ValueError("--motion-scale must be in (0, 1]")
    if not 0.0 < args.ellipse_scale <= 1.0:
        raise ValueError("--ellipse-scale must be in (0, 1]")
    if args.station_angle_deg <= 0.0:
        raise ValueError("--station-angle-deg must be positive")
    if args.num_frames < 0:
        raise ValueError("--num-frames cannot be negative")
    if args.frames_per_timestamp < 0:
        raise ValueError("--frames-per-timestamp cannot be negative")
    custom_sweep_timestamp_range = (
        args.sweep_timestamp_start_frame is not None
        or args.sweep_timestamp_end_frame is not None
    )
    for option_name, option_value in (
        ("--sweep-timestamp-start-frame", args.sweep_timestamp_start_frame),
        ("--sweep-timestamp-end-frame", args.sweep_timestamp_end_frame),
    ):
        if option_value is not None and option_value < 0:
            raise ValueError(f"{option_name} cannot be negative")
    if (
        args.sweep_timestamp_start_frame is not None
        and args.sweep_timestamp_end_frame is not None
        and args.sweep_timestamp_end_frame < args.sweep_timestamp_start_frame
    ):
        raise ValueError(
            "--sweep-timestamp-end-frame must be greater than or equal to "
            "--sweep-timestamp-start-frame"
        )
    if args.ellipse_horizontal_radius < 0.0 or args.ellipse_vertical_radius < 0.0:
        raise ValueError("Ellipse radius overrides cannot be negative")
    custom_ellipse_radii = bool(args.ellipse_horizontal_radius) or bool(
        args.ellipse_vertical_radius
    )
    if bool(args.ellipse_horizontal_radius) != bool(args.ellipse_vertical_radius):
        raise ValueError(
            "--ellipse-horizontal-radius and --ellipse-vertical-radius must be "
            "supplied together"
        )
    if args.trajectory == "ellipse":
        if custom_sweep_timestamp_range:
            raise ValueError(
                "Sweep timestamp range options require --trajectory sweep"
            )
        if args.rail_interpolation != "linear":
            raise ValueError("--rail-interpolation is only valid for sweep trajectories")
        if args.sweep_pacing != "minimum-jerk":
            raise ValueError("--sweep-pacing is only valid for sweep trajectories")
        if args.sweep_bounds != "aabb":
            raise ValueError("--sweep-bounds is only valid for sweep trajectories")
        if args.ellipse_timestamp_start_frame < 0:
            raise ValueError("--ellipse-timestamp-start-frame cannot be negative")
        if args.ellipse_timestamp_end_frame < args.ellipse_timestamp_start_frame:
            raise ValueError(
                "--ellipse-timestamp-end-frame must be greater than or equal to "
                "--ellipse-timestamp-start-frame"
            )
        if args.num_frames:
            raise ValueError(
                "Ellipse length is derived from timestamps; use "
                "--frames-per-timestamp instead of --num-frames"
            )
        if args.time_mode != "sweep" or args.fixed_time is not None:
            raise ValueError(
                "Ellipse timestamps come from the test sequence; "
                "--time-mode fixed and --fixed-time are not supported"
            )
        if args.intrinsics != "fixed":
            raise ValueError("Ellipse mode keeps the test intrinsics fixed")
    elif args.trajectory == "staged-ellipse":
        if custom_sweep_timestamp_range:
            raise ValueError(
                "Sweep timestamp range options require --trajectory sweep"
            )
        if args.rail_interpolation != "linear":
            raise ValueError("--rail-interpolation is only valid for sweep trajectories")
        if args.sweep_pacing != "minimum-jerk":
            raise ValueError("--sweep-pacing is only valid for sweep trajectories")
        if args.sweep_bounds != "aabb":
            raise ValueError("--sweep-bounds is only valid for sweep trajectories")
        if (
            args.ellipse_plane != "rail"
            or custom_ellipse_radii
            or args.ellipse_bounds != "aabb"
        ):
            raise ValueError(
                "Ellipse plane, radius, and bounds overrides are currently "
                "supported only with --trajectory ellipse"
            )
        if args.freeze_frame < 0:
            raise ValueError("--freeze-frame cannot be negative")
        if args.transition_frames < 1:
            raise ValueError("--transition-frames must be positive")
        if args.orbit_frames < 4:
            raise ValueError("--orbit-frames must be at least 4")
        if args.num_frames:
            raise ValueError(
                "Staged-ellipse length is derived from its source sequence and phase "
                "counts; --num-frames is not supported"
            )
        if args.frames_per_timestamp:
            raise ValueError(
                "--frames-per-timestamp is only valid with sweep or ellipse trajectories"
            )
        if args.time_mode != "sweep" or args.fixed_time is not None:
            raise ValueError(
                "Staged-ellipse controls its own timestamp freeze; --time-mode fixed "
                "and --fixed-time are not supported"
            )
        if args.intrinsics != "fixed":
            raise ValueError("Staged-ellipse mode keeps the test intrinsics fixed")
        if args.start_test_frame:
            raise ValueError(
                "Staged-ellipse selects its reference pose with --freeze-frame; "
                "--start-test-frame must remain 0"
            )
    else:
        if args.rail_interpolation == "half-ellipse":
            if args.motion_scale != 1.0:
                raise ValueError(
                    "Half-ellipse sweeps span both endpoints; --motion-scale must be 1"
                )
            if args.intrinsics != "fixed":
                raise ValueError("Half-ellipse sweeps keep the test intrinsics fixed")
            if args.sweep_bounds != "radial":
                raise ValueError(
                    "Half-ellipse sweeps require --sweep-bounds radial because the "
                    "analytic arc can slightly exceed a coordinate extremum"
                )
        if args.frames_per_timestamp or custom_sweep_timestamp_range:
            if args.num_frames:
                raise ValueError(
                    "Sweep length is derived from timestamps when "
                    "a timestamp range or --frames-per-timestamp is set; do not "
                    "also use --num-frames"
                )
            if args.time_mode != "sweep" or args.fixed_time is not None:
                raise ValueError(
                    "Selected sweep timestamps come from the test sequence; "
                    "--time-mode fixed and --fixed-time are not supported"
                )
        if (
            args.ellipse_plane != "rail"
            or custom_ellipse_radii
            or args.ellipse_bounds != "aabb"
        ):
            raise ValueError(
                "Ellipse plane, radius, and bounds overrides require "
                "--trajectory ellipse"
            )
    if output_path in (train_path, test_path):
        raise ValueError(
            "--output must be a separate demo file; refusing to replace a source transform JSON"
        )
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")

    train_json = _load_json(train_path)
    test_json = _load_json(test_path)
    test_groups = _group_frames_by_camera(test_json["frames"])
    selected_camera = args.test_camera or _camera_key(test_json["frames"][0])
    if selected_camera not in test_groups:
        available = ", ".join(sorted(test_groups))
        raise ValueError(f"Test camera '{selected_camera}' was not found; available: {available}")
    test_camera_frames = sorted(
        test_groups[selected_camera],
        key=lambda frame: float(frame.get("time", 0.0)),
    )
    sweep_timestamp_frames = test_camera_frames
    if args.trajectory == "sweep" and custom_sweep_timestamp_range:
        source_numbers = [_frame_number(frame) for frame in test_camera_frames]
        if any(number is None for number in source_numbers):
            raise ValueError("Sweep timestamp ranges require numeric source-frame suffixes")
        first_sweep_frame = (
            min(source_numbers)
            if args.sweep_timestamp_start_frame is None
            else args.sweep_timestamp_start_frame
        )
        last_sweep_frame = (
            max(source_numbers)
            if args.sweep_timestamp_end_frame is None
            else args.sweep_timestamp_end_frame
        )
        sweep_timestamp_frames = _select_frame_number_range(
            test_camera_frames,
            first_sweep_frame,
            last_sweep_frame,
        )
    staged_source_frames = None
    freeze_index = None
    if args.trajectory == "staged-ellipse":
        source_numbers = [_frame_number(frame) for frame in test_camera_frames]
        if any(number is None for number in source_numbers):
            raise ValueError("Staged-ellipse requires numeric source-frame suffixes")
        staged_source_frames = _select_frame_number_range(
            test_camera_frames,
            min(source_numbers),
            max(source_numbers),
        )
        staged_numbers = [_frame_number(frame) for frame in staged_source_frames]
        if args.freeze_frame not in staged_numbers:
            raise ValueError(
                f"--freeze-frame {args.freeze_frame} was not found; available source "
                f"frames are {staged_numbers[0]}-{staged_numbers[-1]}"
            )
        freeze_index = staged_numbers.index(args.freeze_frame)
        start_frame = staged_source_frames[freeze_index]
        reference_pose = _pose(start_frame)
        pose_error = max(
            float(np.max(np.abs(_pose(frame) - reference_pose)))
            for frame in staged_source_frames
        )
        if pose_error > 1.0e-10:
            raise ValueError(
                "Staged-ellipse requires one fixed test-camera pose across the "
                f"temporal source sequence (maximum difference={pose_error:.3g})"
            )
    else:
        if not 0 <= args.start_test_frame < len(test_camera_frames):
            raise IndexError(
                f"--start-test-frame must be in [0, {len(test_camera_frames) - 1}]"
            )
        start_frame = test_camera_frames[args.start_test_frame]
    start_record = _make_record(
        selected_camera,
        start_frame,
        test_json,
        is_start=True,
    )

    rail, cumulative, start_index, target, world_up = _build_camera_rail(
        train_json,
        start_record,
        args.station_angle_deg,
    )
    smooth_rail = (
        _prepare_smooth_rail(rail, cumulative)
        if args.trajectory == "sweep" and args.rail_interpolation == "smooth"
        else None
    )
    half_ellipse_info = None
    if args.trajectory == "sweep" and args.rail_interpolation == "half-ellipse":
        training_records = _representative_records(train_json)
        training_look_target = _estimate_look_target(training_records)
        test_reference_up = _project_to_rotation(start_record["matrix"][:3, :3])[:, 1]
        half_ellipse_info = _prepare_half_ellipse_rail(
            rail,
            start_index,
            training_look_target,
            test_reference_up,
        )
    fixed_intrinsics = start_record["intrinsics"]
    output_frames = []
    ellipse_info = None
    ellipse_timestamp_frames = None
    frames_per_timestamp = None
    sweep_source_frames = None
    staged_validation = None
    half_ellipse_validation = None

    if args.trajectory == "sweep":
        if args.frames_per_timestamp or custom_sweep_timestamp_range:
            frames_per_timestamp = args.frames_per_timestamp or 1
            sweep_source_frames = [
                source_frame
                for source_frame in sweep_timestamp_frames
                for _ in range(frames_per_timestamp)
            ]
            num_frames = len(sweep_source_frames)
            times = np.asarray(
                [float(frame.get("time", 0.0)) for frame in sweep_source_frames],
                dtype=np.float64,
            )
        else:
            num_frames = args.num_frames or len(test_camera_frames)
            times = _output_times(
                test_camera_frames,
                num_frames,
                args.time_mode,
                args.fixed_time,
                start_frame,
            )
        if half_ellipse_info is None:
            coordinates, leg_intervals, reversal_indices = _trajectory_coordinates(
                cumulative,
                start_index,
                num_frames,
                args.motion_scale,
                args.sweep_pacing,
            )
        else:
            coordinates = _open_trajectory_coordinates(
                half_ellipse_info["coordinate_min"],
                half_ellipse_info["coordinate_max"],
                num_frames,
                args.sweep_pacing,
            )
            leg_intervals = [num_frames - 1]
            reversal_indices = []

        for index, (coordinate, timestamp) in enumerate(zip(coordinates, times)):
            source_frame = (
                sweep_source_frames[index]
                if sweep_source_frames is not None
                else _nearest_frame(test_camera_frames, float(timestamp))
            )
            output_frame = copy.deepcopy(source_frame)
            if half_ellipse_info is None:
                matrix, sampled_intrinsics = _sample_rail(
                    rail,
                    cumulative,
                    float(coordinate),
                    args.intrinsics == "interpolate",
                    smooth_rail,
                )
            else:
                matrix = _sample_half_ellipse(half_ellipse_info, float(coordinate))
                sampled_intrinsics = None
            output_frame["transform_matrix"] = matrix.tolist()
            output_frame["time"] = float(timestamp)
            camera_intrinsics = (
                sampled_intrinsics if sampled_intrinsics is not None else fixed_intrinsics
            )
            for key, value in zip(INTRINSIC_KEYS, camera_intrinsics):
                output_frame[key] = float(value)
            output_frames.append(output_frame)
    elif args.trajectory == "ellipse":
        ellipse_timestamp_frames = _select_frame_number_range(
            test_camera_frames,
            args.ellipse_timestamp_start_frame,
            args.ellipse_timestamp_end_frame,
        )
        frames_per_timestamp = args.frames_per_timestamp or 4
        num_frames = len(ellipse_timestamp_frames) * frames_per_timestamp
        matrices, ellipse_info = _ellipse_matrices(
            start_record,
            target,
            world_up,
            rail,
            start_index,
            num_frames,
            args.ellipse_scale,
            args.ellipse_plane,
            (
                args.ellipse_horizontal_radius
                if custom_ellipse_radii
                else None
            ),
            args.ellipse_vertical_radius if custom_ellipse_radii else None,
        )
        source_frames = [
            source_frame
            for source_frame in ellipse_timestamp_frames
            for _ in range(frames_per_timestamp)
        ]
        for matrix, source_frame in zip(matrices, source_frames):
            output_frame = copy.deepcopy(source_frame)
            output_frame["transform_matrix"] = matrix.tolist()
            output_frame["time"] = float(source_frame.get("time", 0.0))
            for key, value in zip(INTRINSIC_KEYS, fixed_intrinsics):
                output_frame[key] = float(value)
            output_frames.append(output_frame)
    else:
        outward, orbit, inward, ellipse_info = _staged_ellipse_matrices(
            start_record,
            target,
            world_up,
            rail,
            start_index,
            args.transition_frames,
            args.orbit_frames,
            args.ellipse_scale,
        )
        lead_frames = staged_source_frames[: freeze_index + 1]
        tail_frames = staged_source_frames[freeze_index + 1 :]
        output_frames.extend(copy.deepcopy(lead_frames))

        for matrix in outward + orbit + inward:
            output_frame = copy.deepcopy(start_frame)
            output_frame["transform_matrix"] = matrix.tolist()
            output_frame["time"] = float(start_frame.get("time", 0.0))
            for key, value in zip(INTRINSIC_KEYS, fixed_intrinsics):
                output_frame[key] = float(value)
            output_frames.append(output_frame)

        output_frames.extend(copy.deepcopy(tail_frames))

    if args.trajectory == "sweep" and half_ellipse_info is None:
        # Preserve the selected test camera bit-for-bit at both sweep boundaries.
        output_frames[0]["transform_matrix"] = copy.deepcopy(
            start_frame["transform_matrix"]
        )
        output_frames[-1]["transform_matrix"] = copy.deepcopy(
            start_frame["transform_matrix"]
        )
        for boundary in (output_frames[0], output_frames[-1]):
            for key, value in zip(INTRINSIC_KEYS, fixed_intrinsics):
                boundary[key] = float(start_frame.get(key, value))

    output_json = copy.deepcopy(test_json)
    output_json["frames"] = output_frames
    bounds_mode = (
        args.sweep_bounds if args.trajectory == "sweep" else args.ellipse_bounds
    )
    stats = _validate_output(
        output_json,
        train_json,
        start_frame,
        dataset,
        args.image_extension,
        args.trajectory,
        ellipse_info,
        bounds_mode,
        None if half_ellipse_info is not None else True,
    )
    if half_ellipse_info is not None:
        half_ellipse_validation = _validate_half_ellipse_output(
            output_json,
            coordinates,
            half_ellipse_info,
        )
    if args.trajectory == "staged-ellipse":
        staged_validation = _validate_staged_ellipse_output(
            output_json,
            staged_source_frames,
            freeze_index,
            ellipse_info,
            args.transition_frames,
            args.orbit_frames,
        )
    _atomic_write_json(output_path, output_json)

    rail_names = " -> ".join(record["camera_key"] for record in rail)
    print(f"Wrote {len(output_frames)} demo views to {output_path}")
    if args.trajectory == "sweep":
        print(f"Rail ({len(rail)} stations): {rail_names}")
        if half_ellipse_info is None:
            print(
                f"Start camera: {selected_camera}; exact closed-loop frames: "
                f"0 and {len(output_frames) - 1}"
            )
            print(f"Look target: {np.array2string(target, precision=5)}")
            print(f"World up: {np.array2string(world_up, precision=5)}")
            print(
                "Leg intervals: "
                f"{leg_intervals}; reversal/end frame indices: {reversal_indices}"
            )
        else:
            first_camera, last_camera = half_ellipse_info["endpoint_cameras"]
            print(
                f"Open half-ellipse: {first_camera} at frame 0 -> "
                f"{last_camera} at frame {len(output_frames) - 1}; no reversals"
            )
            print(
                f"Semiaxes: {half_ellipse_info['axis_lengths'][0]:.5f}, "
                f"{half_ellipse_info['axis_lengths'][1]:.5f}; "
                f"axis angle: {half_ellipse_info['axis_angle_deg']:.3f} degrees"
            )
            print(
                "Training-ray look target: "
                f"{np.array2string(half_ellipse_info['look_target'], precision=5)}"
            )
            print(
                f"Test-camera center lies on the arc at "
                f"{math.degrees(half_ellipse_info['start_theta']):.3f} degrees"
            )
        print(
            f"Rail interpolation: {args.rail_interpolation}; "
            f"pacing: {args.sweep_pacing}"
        )
        if stats["bounds_mode"] == "radial":
            print(
                "Radial range from mean training-camera center: "
                f"generated {stats['generated_radial_max']:.5f} <= "
                f"training {stats['training_radial_max']:.5f}"
            )
        if sweep_source_frames is not None:
            first_source_number = _frame_number(sweep_timestamp_frames[0])
            last_source_number = _frame_number(sweep_timestamp_frames[-1])
            source_range = (
                f"source frames {first_source_number}-{last_source_number}"
                if first_source_number is not None and last_source_number is not None
                else "selected source frames"
            )
            print(
                f"Timestamps: {source_range} ({len(sweep_timestamp_frames)} values), "
                f"each held for {frames_per_timestamp} frames"
            )
    elif args.trajectory == "ellipse":
        support_cameras = " and ".join(ellipse_info["support_cameras"])
        print(
            f"Ellipse center: exact {selected_camera} camera center "
            f"{np.array2string(ellipse_info['center'], precision=5)}"
        )
        print(
            f"Radii: horizontal {ellipse_info['horizontal_radius']:.5f}, "
            f"vertical {ellipse_info['vertical_radius']:.5f}; "
            f"horizontal support: {support_cameras}"
        )
        print(
            "Ellipse plane-normal/look-direction angle: "
            f"{ellipse_info['plane_angle_deg']:.3f} degrees"
        )
        print(
            "Optical-axis look target: "
            f"{np.array2string(ellipse_info['look_target'], precision=5)}"
        )
        print(
            f"Timestamps: source frames {args.ellipse_timestamp_start_frame}-"
            f"{args.ellipse_timestamp_end_frame} ({len(ellipse_timestamp_frames)} values), "
            f"each held for {frames_per_timestamp} frames"
        )
        print(
            f"Periodic seam: frame {len(output_frames) - 1} -> frame 0; "
            f"{stats['seam_position_step']:.5f} world units, "
            f"{stats['seam_rotation_step_deg']:.3f} degrees"
        )
        if stats["bounds_mode"] == "radial":
            print(
                "Radial range from mean training-camera center: "
                f"generated {stats['generated_radial_max']:.5f} <= "
                f"training {stats['training_radial_max']:.5f}"
            )
    else:
        support_cameras = " and ".join(ellipse_info["support_cameras"])
        freeze_time = float(start_frame.get("time", 0.0))
        print(
            f"Ellipse center: exact {selected_camera} camera center "
            f"{np.array2string(ellipse_info['center'], precision=5)}"
        )
        print(
            f"Radii: horizontal {ellipse_info['horizontal_radius']:.5f}, "
            f"vertical {ellipse_info['vertical_radius']:.5f}; "
            f"horizontal support: {support_cameras}"
        )
        print(
            "Ellipse plane-normal/look-direction angle: "
            f"{ellipse_info['plane_angle_deg']:.3f} degrees"
        )
        print(
            f"Time freezes at source frame {args.freeze_frame} "
            f"(t={freeze_time:.10g}) during all {staged_validation['motion_count']} "
            "moving views"
        )
        print(
            "Phase frame indices: "
            f"time-to-freeze 0-{staged_validation['moving_start'] - 1}; "
            f"center-to-ellipse {staged_validation['moving_start']}-"
            f"{staged_validation['orbit_start'] - 1}; "
            f"orbit {staged_validation['orbit_start']}-"
            f"{staged_validation['inward_start'] - 1}; "
            f"ellipse-to-center {staged_validation['inward_start']}-"
            f"{staged_validation['tail_start'] - 1}; "
            f"resumed time {staged_validation['tail_start']}-{len(output_frames) - 1}"
        )
    print(
        "Generated center bounds: "
        f"{np.array2string(stats['center_min'], precision=5)} to "
        f"{np.array2string(stats['center_max'], precision=5)}"
    )
    print(
        "Maximum per-frame motion: "
        f"{stats['max_position_step']:.5f} world units, "
        f"{stats['max_rotation_step_deg']:.3f} degrees"
    )
    print(
        "Note: file_path values reuse real test images only because the renderer "
        "requires them; GT, error maps, and PSNR are not valid for novel poses."
    )


if __name__ == "__main__":
    main()
