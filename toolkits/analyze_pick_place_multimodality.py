# ----------------------------------------------------------------------------
# Copyright (c) 2021-2026 DexForce Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------

"""Analyze PickPlace joint-action branches without opening simulator videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_kinematics as pk
import torch
from pyarrow import parquet


def _matrix_to_quaternion(matrix: np.ndarray) -> list[float]:
    """Convert one rotation matrix to a scalar-first quaternion."""
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        return [
            0.25 * scale,
            float((matrix[2, 1] - matrix[1, 2]) / scale),
            float((matrix[0, 2] - matrix[2, 0]) / scale),
            float((matrix[1, 0] - matrix[0, 1]) / scale),
        ]
    diagonal = np.diag(matrix)
    index = int(np.argmax(diagonal))
    next_index = (index + 1) % 3
    last_index = (index + 2) % 3
    scale = 2.0 * np.sqrt(
        max(
            1.0e-12, 1.0 + diagonal[index] - diagonal[next_index] - diagonal[last_index]
        )
    )
    quaternion = np.zeros(4, dtype=np.float64)
    quaternion[index + 1] = 0.25 * scale
    quaternion[0] = (
        matrix[last_index, next_index] - matrix[next_index, last_index]
    ) / scale
    quaternion[next_index + 1] = (
        matrix[next_index, index] + matrix[index, next_index]
    ) / scale
    quaternion[last_index + 1] = (
        matrix[last_index, index] + matrix[index, last_index]
    ) / scale
    return quaternion.tolist()


def _pose_from_metadata(value: Any) -> np.ndarray:
    """Select the first environment row from a serialized pose."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (4, 4):
        return array
    return array.reshape(-1, 4, 4)[0]


def _episode_rows(
    root: Path,
) -> tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load sidecar metadata and flat actions/states from one dataset."""
    sidecar = root / "meta" / "embodichain_episodes.jsonl"
    episodes = [
        json.loads(line) for line in sidecar.read_text().splitlines() if line.strip()
    ]
    data_files = sorted((root / "data").glob("**/*.parquet"))
    if len(data_files) != 1:
        raise ValueError(
            f"Expected one parquet shard under {root}, found {len(data_files)}."
        )
    table = parquet.read_table(
        data_files[0],
        columns=["action", "observation.state", "episode_index", "frame_index"],
    )
    actions = np.asarray(table["action"].to_pylist(), dtype=np.float64)
    states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float64)
    episode_index = np.asarray(table["episode_index"].to_pylist(), dtype=np.int64)
    frame_index = np.asarray(table["frame_index"].to_pylist(), dtype=np.int64)
    return episodes, actions, states, episode_index, frame_index


def analyze_dataset(root: str | Path, *, urdf_path: str | Path) -> dict[str, Any]:
    """Return deterministic branch and first-50-step similarity statistics.

    Args:
        root: LeRobot dataset directory with episode metadata and one data shard.
        urdf_path: Franka URDF containing the ``fr3_hand_tcp`` endpoint.

    Returns:
        Episode poses, grasp branches, and comparisons for matching reset poses.
    """
    root = Path(root).expanduser().resolve()
    episodes, actions, states, episode_index, frame_index = _episode_rows(root)
    chain = pk.build_serial_chain_from_urdf(Path(urdf_path).read_text(), "fr3_hand_tcp")
    reports: list[dict[str, Any]] = []
    for episode in episodes:
        index = int(episode["episode_index"])
        rows = np.flatnonzero(episode_index == index)
        rows = rows[np.argsort(frame_index[rows])]
        episode_actions = actions[rows]
        episode_states = states[rows]
        first50 = episode_actions[:50]
        finger = episode_actions[:, -2:].mean(axis=1)
        close_candidates = np.flatnonzero(finger <= 0.02)
        close_index = int(close_candidates[0]) if close_candidates.size else -1
        close_index = min(max(close_index, 0), len(episode_actions) - 1)
        arm_qpos = torch.as_tensor(
            episode_actions[close_index : close_index + 1, :7], dtype=torch.float32
        )
        tcp = chain.forward_kinematics(arm_qpos).get_matrix()[0].detach().cpu().numpy()
        source = _pose_from_metadata(episode["segments"][0]["metadata"]["source_pose"])
        target = _pose_from_metadata(episode["segments"][0]["metadata"]["target_pose"])
        reports.append(
            {
                "episode_index": index,
                "seed": episode.get("seed"),
                "spatial_bin_id": episode["segments"][0]["metadata"].get(
                    "spatial_bin_id"
                ),
                "source_xy": source[:2, 3].tolist(),
                "target_xy": target[:2, 3].tolist(),
                "source_yaw": float(np.arctan2(source[1, 0], source[0, 0])),
                "target_yaw": float(np.arctan2(target[1, 0], target[0, 0])),
                "frames": len(rows),
                "first50_action_state_abs_mae": float(
                    np.abs(first50 - episode_states[:50]).mean()
                ),
                "close_index": close_index,
                "close_action_finger_mean": float(finger[close_index]),
                "close_tcp_position": tcp[:3, 3].tolist(),
                "close_tcp_quaternion_wxyz": _matrix_to_quaternion(tcp[:3, :3]),
                "first50_actions": first50.tolist(),
            }
        )

    pairwise: list[dict[str, Any]] = []
    for left_index, left in enumerate(reports):
        for right in reports[left_index + 1 :]:
            same_initial = bool(
                np.allclose(left["source_xy"], right["source_xy"], atol=1.0e-6)
                and np.allclose(left["target_xy"], right["target_xy"], atol=1.0e-6)
                and abs(left["source_yaw"] - right["source_yaw"]) <= 1.0e-6
                and abs(left["target_yaw"] - right["target_yaw"]) <= 1.0e-6
            )
            same_position = bool(
                np.allclose(left["source_xy"], right["source_xy"], atol=1.0e-6)
                and np.allclose(left["target_xy"], right["target_xy"], atol=1.0e-6)
            )
            left_actions = np.asarray(left["first50_actions"], dtype=np.float64)
            right_actions = np.asarray(right["first50_actions"], dtype=np.float64)
            count = min(len(left_actions), len(right_actions))
            difference = left_actions[:count] - right_actions[:count]
            pairwise.append(
                {
                    "left_episode": left["episode_index"],
                    "right_episode": right["episode_index"],
                    "same_initial_pose": same_initial,
                    "same_positions": same_position,
                    "first50_joint_rms": float(np.sqrt(np.mean(difference**2))),
                    "close_tcp_position_distance": float(
                        np.linalg.norm(
                            np.asarray(left["close_tcp_position"])
                            - np.asarray(right["close_tcp_position"])
                        )
                    ),
                    "close_index_delta": abs(
                        left["close_index"] - right["close_index"]
                    ),
                    "source_yaw_delta": abs(left["source_yaw"] - right["source_yaw"]),
                    "target_yaw_delta": abs(left["target_yaw"] - right["target_yaw"]),
                }
            )
    same_pose_pairs = [item for item in pairwise if item["same_initial_pose"]]
    same_position_pairs = [item for item in pairwise if item["same_positions"]]
    same_pose_approx_pairs = [
        item
        for item in same_position_pairs
        if item["source_yaw_delta"] <= np.deg2rad(1.0)
        and item["target_yaw_delta"] <= np.deg2rad(1.0)
    ]
    return {
        "dataset_root": str(root),
        "episodes": len(reports),
        "action_dim": int(actions.shape[1]),
        "state_dim": int(states.shape[1]),
        "first50_steps": 50,
        "episodes_report": reports,
        "pairwise": pairwise,
        "same_initial_pose_pairs": same_pose_pairs,
        "same_pose_approx_pairs_1deg": same_pose_approx_pairs,
        "same_position_pairs": same_position_pairs,
        "multimodality": {
            "same_pose_pair_count": len(same_pose_pairs),
            "same_pose_approx_pair_count_1deg": len(same_pose_approx_pairs),
            "same_position_pair_count": len(same_position_pairs),
            "same_pose_first50_joint_rms_max": max(
                (item["first50_joint_rms"] for item in same_pose_pairs), default=0.0
            ),
            "same_pose_close_tcp_distance_max": max(
                (item["close_tcp_position_distance"] for item in same_pose_pairs),
                default=0.0,
            ),
            "same_position_first50_joint_rms_max": max(
                (item["first50_joint_rms"] for item in same_position_pairs), default=0.0
            ),
            "same_position_close_tcp_distance_max": max(
                (item["close_tcp_position_distance"] for item in same_position_pairs),
                default=0.0,
            ),
            "same_pose_approx_first50_joint_rms_max_1deg": max(
                (item["first50_joint_rms"] for item in same_pose_approx_pairs),
                default=0.0,
            ),
            "same_position_first50_joint_rms_mean": (
                float(
                    np.mean([item["first50_joint_rms"] for item in same_position_pairs])
                )
                if same_position_pairs
                else 0.0
            ),
            "same_position_source_yaw_delta_mean": (
                float(
                    np.mean([item["source_yaw_delta"] for item in same_position_pairs])
                )
                if same_position_pairs
                else 0.0
            ),
            "interpretation": (
                "same-position branches differ materially"
                if any(item["first50_joint_rms"] > 0.02 for item in same_position_pairs)
                else "no material same-pose branch divergence detected"
            ),
        },
    }


def main() -> None:
    """Write branch-comparison reports for the datasets selected by the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_roots", nargs="+", type=Path)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = {
        "datasets": [
            analyze_dataset(root, urdf_path=args.urdf) for root in args.dataset_roots
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
