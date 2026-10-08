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

"""Configuration and registration tests for the visual pick-place task."""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from embodichain.lab.gym.utils._component_composition import _resolve_gym_components
from embodichain.lab.gym.utils.gym_utils import config_to_cfg, get_manager_modules
from embodichain.lab.gym.utils.registration import (
    REGISTERED_ENVS,
    discover_task_packages,
    execute_init_hooks,
)
from embodichain.utils.utility import load_config
from embodichain_tasks.manipulation.pick_place import (
    CUBE_HALF_HEIGHT,
    GOAL_MARKER_HALF_HEIGHT,
    GRASP_HOLD_STEPS,
    HAND_RELEASE_QPOS_THRESHOLD,
    PLACEMENT_ANGULAR_SPEED_THRESHOLD,
    PLACEMENT_LINEAR_SPEED_THRESHOLD,
    PLACEMENT_SETTLE_CONTROL_STEPS,
    PLACEMENT_STABLE_CONTROL_STEPS,
    PickPlaceEnv,
)

_ROOT = Path(__file__).parents[3]
_CONFIG = (
    _ROOT
    / "embodichain_tasks/configs/tasks/manipulation/pick_place/task.franka_smoke_train.yaml"
)
_RUNTIME_CONFIG = (
    _ROOT / "embodichain_tasks/configs/tasks/manipulation/pick_place/"
    "task.franka_smoke_train_runtime.yaml"
)


def test_pick_place_is_discovered_and_has_one_spatial_segment() -> None:
    """The task module is imported and the smoke deployment parses."""
    discover_task_packages()
    execute_init_hooks()
    assert "PickPlace-v1" in REGISTERED_ENVS

    payload = load_config(_CONFIG)
    resolved = _resolve_gym_components(payload, base_dir=_CONFIG.parent)
    cfg = config_to_cfg(
        payload,
        manager_modules=get_manager_modules(),
        source_path=_CONFIG,
    )
    assert resolved.config["max_episodes"] == 16
    assert resolved.config["seed"] == 4101
    assert cfg.extensions["spatial_grid_size"] == 5
    assert cfg.extensions["spatial_cube_x_range"] == [-0.43, -0.41]
    assert len(cfg.sensor) == 1
    assert cfg.sensor[0].uid == "cam_high"
    assert "randomize_cube" in resolved.config["env"]["events"]


def test_pick_place_runtime_forces_original_grasp_variant() -> None:
    """Runtime smoke deployment exposes the stable symmetric grasp choice."""
    payload = load_config(_RUNTIME_CONFIG)
    cfg = config_to_cfg(
        payload,
        manager_modules=get_manager_modules(),
        source_path=_RUNTIME_CONFIG,
    )
    assert cfg.extensions["deterministic_grasp_variant"] == "original"


def test_pick_place_target_center_clears_marker_top_surface() -> None:
    """The target cube center sits on, rather than inside, the marker."""
    marker_center_z = 0.005
    assert marker_center_z + GOAL_MARKER_HALF_HEIGHT + CUBE_HALF_HEIGHT == 0.035


def test_pick_place_expert_replays_joint_targets_through_env_step() -> None:
    """The expert action iterator contains only joint targets and hold frames."""
    signature = inspect.signature(PickPlaceEnv._iter_segment_actions)
    assert tuple(signature.parameters) == ("self", "trajectory")
    trajectory = torch.arange(18, dtype=torch.float32).reshape(1, 2, 9)
    actions = list(PickPlaceEnv._iter_segment_actions(object(), trajectory))
    assert len(actions) == 2 + PLACEMENT_SETTLE_CONTROL_STEPS
    assert torch.equal(actions[0], trajectory[:, 0])
    assert torch.equal(actions[1], trajectory[:, 1])
    assert all(torch.equal(action, trajectory[:, -1]) for action in actions[2:])


def test_pick_place_grasp_hold_adds_contact_stabilization_frames() -> None:
    """The expert inserts the configured closed-hand settling interval."""
    trajectory = torch.zeros(1, 90, 9)
    trajectory[:, :, -2:] = torch.linspace(0.04, 0.0, 90).reshape(1, 90, 1)
    held = PickPlaceEnv._insert_grasp_hold(object(), trajectory)

    assert GRASP_HOLD_STEPS == 60
    assert held.shape == (1, trajectory.shape[1] + GRASP_HOLD_STEPS, 9)


def test_pick_place_step_info_uses_task_success_contract() -> None:
    """Closed-loop step info mirrors the task validator and exposes distance."""
    cube_pose = torch.eye(4).unsqueeze(0)
    cube_pose[:, :3, 3] = torch.tensor([[0.0, 0.0, 0.035]])
    goal_pose = torch.eye(4).unsqueeze(0)
    goal_pose[:, :3, 3] = torch.tensor([[0.0, 0.0, 0.005]])
    fake_env = SimpleNamespace(
        _cube=SimpleNamespace(get_local_pose=lambda **_: cube_pose),
        _goal=SimpleNamespace(get_local_pose=lambda **_: goal_pose),
        device=torch.device("cpu"),
        is_task_success=lambda: torch.tensor([True]),
        _placement_released=lambda: torch.tensor([True]),
        _placement_stable_steps=torch.tensor([PLACEMENT_STABLE_CONTROL_STEPS]),
        _cube_linear_speed=torch.zeros(1),
        _cube_angular_speed=torch.zeros(1),
    )

    success, fail, metrics = PickPlaceEnv.compute_task_state(fake_env)

    assert success.tolist() == [True]
    assert fail.tolist() == [False]
    assert torch.allclose(metrics["distance_to_goal"], torch.zeros(1))
    assert metrics["placement_released"].tolist() == [True]
    assert metrics["placement_stable_steps"].tolist() == [
        PLACEMENT_STABLE_CONTROL_STEPS
    ]


def _pick_place_stability_env(num_envs: int = 2) -> PickPlaceEnv:
    """Build a CPU task shell for placement-state conformance checks."""
    env = object.__new__(PickPlaceEnv)
    env._num_envs = num_envs
    env.sim = SimpleNamespace(device=torch.device("cpu"))
    env.cfg = SimpleNamespace(events=None)
    env._elapsed_steps = torch.zeros(num_envs, dtype=torch.int32)
    env._placement_stable_steps = torch.zeros(num_envs, dtype=torch.long)
    env._last_placement_stability_step = torch.full((num_envs,), -1, dtype=torch.long)
    env._cube_linear_speed = torch.zeros(num_envs)
    env._cube_angular_speed = torch.zeros(num_envs)

    cube_pose = torch.eye(4).repeat(num_envs, 1, 1)
    cube_pose[:, 2, 3] = CUBE_HALF_HEIGHT + GOAL_MARKER_HALF_HEIGHT
    goal_pose = torch.eye(4).repeat(num_envs, 1, 1)
    goal_pose[:, 2, 3] = GOAL_MARKER_HALF_HEIGHT
    env._cube = SimpleNamespace(
        get_local_pose=lambda **_: cube_pose,
        body_data=SimpleNamespace(
            lin_vel=torch.zeros(num_envs, 3),
            ang_vel=torch.zeros(num_envs, 3),
        ),
    )
    env._goal = SimpleNamespace(get_local_pose=lambda **_: goal_pose)
    qpos = torch.zeros(num_envs, 9)
    qpos[:, -2:] = 0.04
    env.robot = SimpleNamespace(get_joint_ids=lambda **_: [7, 8], get_qpos=lambda: qpos)
    env._test_qpos = qpos
    return env


def test_pick_place_success_requires_release_and_ten_distinct_stable_steps() -> None:
    """Repeated info reads do not advance release stability evidence."""
    env = _pick_place_stability_env()

    for _ in range(12):
        PickPlaceEnv._update_sim_state(env)
    assert env._placement_stable_steps.tolist() == [1, 1]
    assert PickPlaceEnv.is_task_success(env).tolist() == [False, False]
    PickPlaceEnv.compute_task_state(env)
    PickPlaceEnv.compute_task_state(env)
    assert env._placement_stable_steps.tolist() == [1, 1]

    for elapsed in range(1, PLACEMENT_STABLE_CONTROL_STEPS):
        env._elapsed_steps[:] = elapsed
        PickPlaceEnv._update_sim_state(env)
    assert env._placement_stable_steps.tolist() == [PLACEMENT_STABLE_CONTROL_STEPS] * 2
    assert PickPlaceEnv.is_task_success(env).tolist() == [True, True]


def test_pick_place_closed_hand_or_motion_resets_stability_window() -> None:
    """A cube cannot pass while held or while moving above the speed limits."""
    env = _pick_place_stability_env(num_envs=1)
    env._placement_stable_steps[:] = PLACEMENT_STABLE_CONTROL_STEPS - 1
    env._test_qpos[:, -2:] = HAND_RELEASE_QPOS_THRESHOLD - 0.001
    PickPlaceEnv._update_sim_state(env)
    assert env._placement_stable_steps.tolist() == [0]
    assert PickPlaceEnv.is_task_success(env).tolist() == [False]

    env._test_qpos[:, -2:] = HAND_RELEASE_QPOS_THRESHOLD
    env._cube.body_data.lin_vel[:, 0] = PLACEMENT_LINEAR_SPEED_THRESHOLD + 0.01
    env._elapsed_steps[:] = 1
    PickPlaceEnv._update_sim_state(env)
    assert env._placement_stable_steps.tolist() == [0]

    env._cube.body_data.lin_vel.zero_()
    env._cube.body_data.ang_vel[:, 2] = PLACEMENT_ANGULAR_SPEED_THRESHOLD + 0.01
    env._elapsed_steps[:] = 2
    PickPlaceEnv._update_sim_state(env)
    assert env._placement_stable_steps.tolist() == [0]


def test_pick_place_partial_reset_clears_only_selected_stability_rows() -> None:
    """A partial reset cannot inherit another episode's settling evidence."""
    env = _pick_place_stability_env()
    env._placement_stable_steps[:] = torch.tensor([4, 9])
    env._last_placement_stability_step[:] = torch.tensor([3, 8])
    PickPlaceEnv._reset_placement_stability(env, torch.tensor([1]))
    assert env._placement_stable_steps.tolist() == [4, 0]
    assert env._last_placement_stability_step.tolist() == [3, -1]


def test_pick_place_expert_correction_respects_window_and_threshold() -> None:
    """Correction labels are expert joint targets only inside the window."""
    fake_env = object.__new__(PickPlaceEnv)
    fake_env._num_envs = 1
    fake_env.sim = SimpleNamespace(device=torch.device("cpu"))
    fake_env.robot = SimpleNamespace(dof=2)
    fake_env._correction_steps = torch.zeros(1, dtype=torch.long)
    expert = torch.tensor([[[1.0, 1.0]]])
    fake_env._correction_trajectory = expert
    fake_env._correction_plan_success = torch.tensor([True])
    fake_env._plan_pick_place = lambda: (
        torch.tensor([True]),
        expert,
        torch.eye(4).unsqueeze(0),
        torch.eye(4).unsqueeze(0),
    )

    corrected, flag = PickPlaceEnv.get_expert_correction_action(
        fake_env,
        torch.zeros(1, 2),
        max_steps=2,
        deviation_threshold=0.5,
        force_steps=1,
    )
    assert flag.tolist() == [True]
    assert torch.equal(corrected, expert[:, 0])

    corrected, flag = PickPlaceEnv.get_expert_correction_action(
        fake_env,
        expert[:, 0],
        max_steps=2,
        deviation_threshold=0.5,
        force_steps=1,
    )
    assert flag.tolist() == [False]
    assert torch.equal(corrected, expert[:, 0])


def test_pick_place_plan_seed_is_pose_deterministic() -> None:
    """Identical scene poses use the same grasp/IK RNG stream."""
    source = torch.eye(4).unsqueeze(0)
    target = torch.eye(4).unsqueeze(0)
    same = PickPlaceEnv._deterministic_plan_seed(source, target)
    repeat = PickPlaceEnv._deterministic_plan_seed(source.clone(), target.clone())
    target[:, 0, 3] = 0.01
    different = PickPlaceEnv._deterministic_plan_seed(source, target)
    assert same == repeat
    assert same != different


_PANDA_GRASP_CALIBRATION = [
    [0.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
]


def _calibration_demo_env(calibration=None):
    """Use a CPU planner boundary to inspect emitted demo invocations."""
    from embodichain.lab.sim.atomic_actions import (
        ActionBinding,
        Affordance,
        ObjectSemantics,
    )

    invocations = []
    pose = torch.eye(4).unsqueeze(0)
    held = SimpleNamespace(object_to_eef=pose)

    def compile_invocations(requests, context):
        invocations.extend(requests)
        return SimpleNamespace(
            plan_success=torch.tensor([True]),
            trajectory=SimpleNamespace(positions=torch.zeros(1, 90, 9)),
            projected_context=SimpleNamespace(get_held_object=lambda _: held),
        )

    env = object.__new__(PickPlaceEnv)
    env._num_envs = 1
    env.sim = SimpleNamespace(device=torch.device("cpu"), physics_dt=0.01)
    env.sim_cfg = SimpleNamespace(physics_dt=0.01)
    env.cfg = SimpleNamespace(sim_steps_per_control=4)
    env._cube = SimpleNamespace(get_local_pose=lambda **_: pose.clone())
    env._goal = SimpleNamespace(get_local_pose=lambda **_: pose.clone())
    env._cube_semantics = ObjectSemantics(
        affordance=Affordance(), geometry={}, label="cube", entity_id="cube"
    )
    env._spatial_bin_ids = torch.zeros(1, dtype=torch.long)
    env._action_engine = SimpleNamespace(
        bind_control_parts=lambda *_: ActionBinding("calibration-test"),
        initial_context=lambda **_: None,
        compile=compile_invocations,
    )
    if calibration is not None:
        env.grasp_frame_to_eef = calibration
    return env, invocations


@pytest.mark.parametrize("calibration", [None, _PANDA_GRASP_CALIBRATION])
def test_pick_place_demo_passes_and_records_grasp_calibration(calibration) -> None:
    """Optional task calibration reaches the planner and recorded metadata."""
    from embodichain.lab.sim.atomic_actions import PickUpOptions

    env, invocations = _calibration_demo_env(calibration)
    segment = env.create_demo_segments()[0]
    expected = (
        PickUpOptions().grasp_frame_to_eef
        if calibration is None
        else torch.tensor(calibration, dtype=torch.float32)
    )
    pickup = invocations[0]
    assert pickup.skill_id == "pick_up"
    assert torch.equal(pickup.skill_options.grasp_frame_to_eef, expected)
    assert segment.metadata["grasp_frame_to_eef"] == expected.tolist()
    if calibration is not None:
        # Panda opens along TCP y; canonical antipodal grasps open along x.
        rotation = pickup.skill_options.grasp_frame_to_eef[:3, :3]
        assert torch.equal(
            rotation @ torch.tensor([0.0, 1.0, 0.0]), torch.tensor([1.0, 0.0, 0.0])
        )
        assert torch.equal(
            rotation @ torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, 0.0])
        )


def test_pick_place_demo_rejects_invalid_grasp_calibration() -> None:
    """Malformed task transforms fail before reaching the planner boundary."""
    env, invocations = _calibration_demo_env([[1.0, 0.0], [0.0, 1.0]])
    with pytest.raises(ValueError):
        env.create_demo_segments()
    assert invocations == []
