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

"""Single-cycle visual pick-and-place expert for spatial generalization."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from functools import partial
from typing import TYPE_CHECKING, Any

import torch

from embodichain.lab.gym.envs import DemoSegment, EmbodiedEnv, EmbodiedEnvCfg
from embodichain.lab.gym.utils.registration import register_env
from embodichain.utils import logger

if TYPE_CHECKING:
    from embodichain.lab.sim.atomic_actions import (
        AtomicActionEngine,
        ObjectSemantics,
        PickUpOptions,
    )
    from embodichain.lab.sim.objects import RigidObject

__all__ = ["PickPlaceEnv"]

CUBE_UID = "cube"
GOAL_UID = "goal_marker"
CONTROL_PART = "arm"
HAND_CONTROL_PART = "hand"
CUBE_HALF_HEIGHT = 0.025
GOAL_MARKER_HALF_HEIGHT = 0.005
HAND_OPEN_QPOS = 0.04
HAND_CLOSE_QPOS = 0.0
HAND_RELEASE_QPOS_THRESHOLD = 0.035
PLACEMENT_LINEAR_SPEED_THRESHOLD = 0.02
PLACEMENT_ANGULAR_SPEED_THRESHOLD = 0.5
PLACEMENT_STABLE_CONTROL_STEPS = 10
PLACEMENT_SETTLE_CONTROL_STEPS = 10
PICK_SAMPLE_INTERVAL = 90
PLACE_SAMPLE_INTERVAL = 90
HAND_INTERP_STEPS = 10
# Give the real gripper enough control intervals to settle its contact before
# the lift segment starts.  The planner already closes the fingers gradually;
# this hold is the additional contact-stabilization interval.
GRASP_HOLD_STEPS = 60


@register_env("PickPlace-v1", max_episode_steps=600)
class PickPlaceEnv(EmbodiedEnv):
    """Pick one cube and place it on the reset-sampled visual target.

    The optional ``grasp_frame_to_eef`` extension is a 4x4 SE(3) transform
    from the sampled grasp frame to the robot TCP. Omitting it preserves the
    pickup primitive's identity calibration. Each demo records the effective
    matrix in its segment metadata.
    """

    def __init__(self, cfg: EmbodiedEnvCfg | None = None, **kwargs: Any) -> None:
        super().__init__(cfg, **kwargs)
        self._spatial_episode_index = 0
        self._spatial_bin_ids = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._correction_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._placement_stable_steps = torch.zeros(
            self.num_envs, dtype=torch.long, device=self.device
        )
        self._last_placement_stability_step = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._cube_linear_speed = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self._cube_angular_speed = torch.zeros_like(self._cube_linear_speed)
        self._correction_trajectory: torch.Tensor | None = None
        self._correction_plan_success: torch.Tensor | None = None
        cube = self.sim.get_rigid_object(CUBE_UID)
        goal = self.sim.get_rigid_object(GOAL_UID)
        if cube is None or goal is None:
            raise RuntimeError(
                "PickPlace-v1 requires rigid objects 'cube' and 'goal_marker'."
            )
        self._cube: RigidObject = cube
        self._goal: RigidObject = goal
        self._initialize_atomic_actions()

    def _initialize_episode(self, env_ids=None, **kwargs: Any) -> None:
        """Reset the scene and assign a balanced object/goal spatial bin."""
        super()._initialize_episode(env_ids=env_ids, **kwargs)
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._reset_placement_stability(ids)
        grid_size = int(getattr(self, "spatial_grid_size", 5))
        total_bins = grid_size * grid_size
        bin_ids = (
            torch.arange(len(ids), device=self.device) + self._spatial_episode_index
        )
        bin_ids = bin_ids.remainder(total_bins)
        self._spatial_episode_index += len(ids)
        self._spatial_bin_ids[ids] = bin_ids

        cube_pose = self._cube.get_local_pose(to_matrix=True)[ids].clone()
        goal_pose = self._goal.get_local_pose(to_matrix=True)[ids].clone()
        cube_x = self._spatial_range("cube_x", (-0.48, -0.36))
        cube_y = self._spatial_range("cube_y", (-0.14, -0.02))
        goal_x = self._spatial_range("goal_x", (-0.45, -0.35))
        goal_y = self._spatial_range("goal_y", (0.42, 0.54))
        cube_pose[:, 0, 3] = cube_x[bin_ids // grid_size]
        cube_pose[:, 1, 3] = cube_y[bin_ids % grid_size]
        goal_pose[:, 0, 3] = goal_x[bin_ids // grid_size]
        goal_pose[:, 1, 3] = goal_y[bin_ids % grid_size]
        self._cube.set_local_pose(cube_pose, env_ids=ids)
        self._goal.set_local_pose(goal_pose, env_ids=ids)
        self._cube.clear_dynamics(ids)
        self._correction_steps[ids] = 0
        if env_ids is None or self._correction_trajectory is None:
            self._correction_trajectory = None
            self._correction_plan_success = None

    def _reset_placement_stability(self, env_ids: torch.Tensor) -> None:
        """Clear placement evidence for rows reset by the simulator."""
        if not hasattr(self, "_placement_stable_steps"):
            return
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            return
        self._placement_stable_steps[ids] = 0
        self._last_placement_stability_step[ids] = -1
        self._cube_linear_speed[ids] = 0.0
        self._cube_angular_speed[ids] = 0.0

    def _placement_in_goal_tolerance(self) -> torch.Tensor:
        """Return the existing axis-wise cube-to-goal position check."""
        cube_pos = self._cube.get_local_pose(to_matrix=True)[:, :3, 3]
        goal_pos = self._goal.get_local_pose(to_matrix=True)[:, :3, 3].clone()
        goal_pos[:, 2] += GOAL_MARKER_HALF_HEIGHT + CUBE_HALF_HEIGHT
        tolerance = torch.tensor([0.055, 0.055, 0.025], device=self.device)
        return torch.all(torch.abs(cube_pos - goal_pos) <= tolerance, dim=1)

    def _placement_released(self) -> torch.Tensor:
        """Return whether both Panda fingers have reached the open position."""
        hand_joint_ids = self.robot.get_joint_ids(name=HAND_CONTROL_PART)
        finger_qpos = self.robot.get_qpos()[:, hand_joint_ids]
        return torch.all(finger_qpos >= HAND_RELEASE_QPOS_THRESHOLD, dim=1)

    def _update_sim_state(self, **kwargs: Any) -> None:
        """Advance placement stability once after each completed physics step."""
        super()._update_sim_state(**kwargs)
        # EmbodiedEnv can dispatch this override while its constructor is
        # building the base scene, before PickPlaceEnv has allocated counters.
        if not hasattr(self, "_placement_stable_steps"):
            return
        linear_velocity = self._cube.body_data.lin_vel
        angular_velocity = self._cube.body_data.ang_vel
        self._cube_linear_speed = torch.linalg.vector_norm(linear_velocity, dim=-1)
        self._cube_angular_speed = torch.linalg.vector_norm(angular_velocity, dim=-1)

        elapsed_steps = self._elapsed_steps.to(device=self.device, dtype=torch.long)
        new_step = elapsed_steps != self._last_placement_stability_step
        if not bool(new_step.any().item()):
            return

        released = self._placement_released()
        in_tolerance = self._placement_in_goal_tolerance()
        stable = (self._cube_linear_speed <= PLACEMENT_LINEAR_SPEED_THRESHOLD) & (
            self._cube_angular_speed <= PLACEMENT_ANGULAR_SPEED_THRESHOLD
        )
        accepted = in_tolerance & released & stable
        self._placement_stable_steps = torch.where(
            new_step,
            torch.where(
                accepted,
                self._placement_stable_steps + 1,
                torch.zeros_like(self._placement_stable_steps),
            ),
            self._placement_stable_steps,
        )
        self._last_placement_stability_step = torch.where(
            new_step, elapsed_steps, self._last_placement_stability_step
        )

    def get_expert_correction_action(
        self,
        policy_action: torch.Tensor,
        *,
        max_steps: int = 50,
        deviation_threshold: float = 0.05,
        force_steps: int = 10,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a contact-valid expert correction for a policy action.

        The helper is intentionally side-effect free with respect to the
        simulator: it plans from the current scene and returns the first joint
        target.  A caller may pass the returned action through the ordinary
        :meth:`step` path and persist ``intervene_flag`` alongside the frame.

        Args:
            policy_action: Joint-target action with shape ``(num_envs, 9)``.
            max_steps: Maximum correction window per reset episode.
            deviation_threshold: L2 joint-target error that triggers takeover.
            force_steps: Initial steps always labelled by the expert.

        Returns:
            ``(action, intervene_flag)`` with one action and flag per row.
        """
        if not isinstance(policy_action, torch.Tensor):
            raise TypeError("policy_action must be a torch.Tensor.")
        if policy_action.shape != (self.num_envs, self.robot.dof):
            raise ValueError(
                "policy_action must have shape "
                f"({self.num_envs}, {self.robot.dof}), got {tuple(policy_action.shape)}."
            )
        if type(max_steps) is not int or max_steps < 0:
            raise ValueError("max_steps must be a non-negative integer.")
        if type(force_steps) is not int or force_steps < 0:
            raise ValueError("force_steps must be a non-negative integer.")
        if deviation_threshold < 0.0:
            raise ValueError("deviation_threshold must be non-negative.")
        if self._correction_trajectory is None:
            plan_success, trajectory, _, _ = self._plan_pick_place()
            self._correction_plan_success = plan_success.detach().clone()
            self._correction_trajectory = trajectory.detach().clone()
        plan_success = self._correction_plan_success
        trajectory = self._correction_trajectory
        if trajectory is None or plan_success is None or trajectory.shape[1] == 0:
            return policy_action, torch.zeros(
                self.num_envs, dtype=torch.bool, device=self.device
            )
        trajectory_step = self._correction_steps.clamp_max(trajectory.shape[1] - 1)
        expert_action = trajectory[
            torch.arange(self.num_envs, device=self.device), trajectory_step
        ].to(policy_action.device)
        step_ids = self._correction_steps.to(policy_action.device)
        eligible = step_ids < max_steps
        deviation = torch.linalg.vector_norm(
            policy_action.to(expert_action.device) - expert_action,
            dim=1,
        )
        intervene = eligible & (
            (step_ids < force_steps) | (deviation >= float(deviation_threshold))
        )
        intervene &= plan_success.to(intervene.device)
        corrected = torch.where(intervene[:, None], expert_action, policy_action)
        self._correction_steps += 1
        return corrected, intervene

    def _spatial_range(self, name: str, default: tuple[float, float]) -> torch.Tensor:
        """Return one configured spatial-grid axis as a device tensor."""
        values = getattr(self, f"spatial_{name}_range", default)
        if len(values) != 2:
            raise ValueError(f"spatial_{name}_range must contain two values.")
        return torch.linspace(
            float(values[0]),
            float(values[1]),
            int(getattr(self, "spatial_grid_size", 5)),
            device=self.device,
        )

    def _initialize_atomic_actions(self) -> None:
        from embodichain.lab.sim.atomic_actions import (
            AntipodalAffordance,
            AtomicActionEngine,
            ControlPartCommandProfile,
            ObjectSemantics,
        )
        from embodichain.lab.sim.planners import (
            MotionGenCfg,
            MotionGenerator,
            ToppraPlannerCfg,
        )
        from embodichain.toolkits.graspkit import ParallelJawGripperModelCfg
        from embodichain.toolkits.graspkit.pg_grasp import (
            AntipodalGraspPoseGenerator,
            AntipodalGraspPoseGeneratorCfg,
            GraspAnnotationCfg,
            ParallelJawGraspCollisionCfg,
        )

        hand_dof = len(self.robot.get_joint_ids(name=HAND_CONTROL_PART))
        hand_open_qpos = torch.full(
            (hand_dof,), HAND_OPEN_QPOS, dtype=torch.float32, device=self.device
        )
        hand_close_qpos = torch.full(
            (hand_dof,), HAND_CLOSE_QPOS, dtype=torch.float32, device=self.device
        )
        motion_generator = MotionGenerator(
            cfg=MotionGenCfg(planner_cfg=ToppraPlannerCfg(robot_uid=self.robot.uid))
        )
        grasp_generator = AntipodalGraspPoseGenerator(
            ParallelJawGripperModelCfg(
                model_id="franka_panda_hand",
                min_opening_width=0.001,
                max_opening_width=0.08,
                finger_length=0.06,
                finger_width=0.02,
                finger_thickness=0.01,
                palm_depth=0.06,
            ),
            algorithm_cfg=AntipodalGraspPoseGeneratorCfg(
                sample_count=1000,
                approach_direction_samples=4,
                max_candidates=30,
            ),
            collision_cfg=ParallelJawGraspCollisionCfg(
                opening_margin=0.002,
                point_sample_density=0.012,
                filter_ground_collision=False,
            ),
            annotation_cfg=GraspAnnotationCfg(force_refresh=False),
        )
        self._action_engine: AtomicActionEngine = AtomicActionEngine(
            motion_generator,
            control_profiles={
                HAND_CONTROL_PART: ControlPartCommandProfile.joint_positions(
                    open=hand_open_qpos,
                    grasp=hand_close_qpos,
                )
            },
            grasp_pose_generators={HAND_CONTROL_PART: grasp_generator},
        )
        mesh_vertices = self._cube.get_vertices(scale=True)[0]
        mesh_triangles = self._cube.get_triangles()[0]
        self._cube_semantics: ObjectSemantics = ObjectSemantics(
            affordance=AntipodalAffordance(
                mesh_vertices=mesh_vertices,
                mesh_triangles=mesh_triangles,
            ),
            geometry={
                "mesh_vertices": mesh_vertices,
                "mesh_triangles": mesh_triangles,
            },
            label=CUBE_UID,
            entity_id=CUBE_UID,
        )

    def create_demo_segments(self, **kwargs: Any) -> tuple[DemoSegment]:
        """Plan one Pick→Place segment from the current randomized scene."""
        del kwargs
        (
            plan_success,
            trajectory,
            source_pose,
            target_pose,
        ) = self._plan_pick_place()
        return (
            DemoSegment(
                actions=self._iter_segment_actions(trajectory),
                name="pick_and_place",
                target_uid=CUBE_UID,
                instruction="Pick up the cube and place it on the marked target.",
                metadata={
                    "segment_index": 0,
                    "segment_count": 1,
                    "planning_success": plan_success.detach().cpu().tolist(),
                    "source_pose": source_pose.detach().cpu().tolist(),
                    "target_pose": target_pose.detach().cpu().tolist(),
                    "atomic_actions": ["pick_up", "place"],
                    "spatial_bin_id": self._spatial_bin_ids.detach().cpu().tolist(),
                    "planned_action_steps": int(trajectory.shape[1])
                    + PLACEMENT_SETTLE_CONTROL_STEPS,
                    "completion_criterion": "released_and_stable_10_steps_v1",
                    "release_qpos_threshold": HAND_RELEASE_QPOS_THRESHOLD,
                    "linear_speed_threshold": PLACEMENT_LINEAR_SPEED_THRESHOLD,
                    "angular_speed_threshold": PLACEMENT_ANGULAR_SPEED_THRESHOLD,
                    "stable_control_steps": PLACEMENT_STABLE_CONTROL_STEPS,
                    "grasp_frame_to_eef": self._pick_up_options()
                    .grasp_frame_to_eef.detach()
                    .cpu()
                    .tolist(),
                },
                validator=partial(
                    self._validate_pick_place, plan_success.detach().clone()
                ),
            ),
        )

    def _plan_pick_place(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Plan one deterministic grasp branch for the current scene."""
        source_pose = self._cube.get_local_pose(to_matrix=True).to(
            device=self.device, dtype=torch.float32
        )
        marker_pose = self._goal.get_local_pose(to_matrix=True).to(
            device=self.device, dtype=torch.float32
        )
        if not bool(getattr(self, "deterministic_grasp", False)):
            return self._plan_pick_place_impl(source_pose, marker_pose)
        seed = getattr(self, "deterministic_grasp_seed", None)
        if seed is None:
            seed = self._deterministic_plan_seed(source_pose, marker_pose)
        else:
            seed = int(seed)
        cuda_devices = []
        if self.device.type == "cuda":
            cuda_devices = [
                (
                    torch.cuda.current_device()
                    if self.device.index is None
                    else self.device.index
                )
            ]
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            if cuda_devices:
                torch.cuda.manual_seed_all(seed)
            return self._plan_pick_place_impl(source_pose, marker_pose)

    @staticmethod
    def _deterministic_plan_seed(
        source_pose: torch.Tensor, marker_pose: torch.Tensor
    ) -> int:
        """Derive a stable RNG seed from the scene poses."""
        payload = torch.cat((source_pose[:1], marker_pose[:1]), dim=0)
        digest = hashlib.blake2b(
            payload.detach().cpu().numpy().round(6).tobytes(),
            digest_size=8,
            person=b"PickPlacePlan",
        )
        return int.from_bytes(digest.digest(), "little") % (2**63 - 1)

    def _pick_up_options(self) -> PickUpOptions:
        """Apply the optional task calibration to the existing pickup options."""
        from embodichain.lab.sim.atomic_actions import PickUpOptions

        options: dict[str, Any] = {
            "pre_grasp_distance": 0.12,
            "lift_height": 0.15,
            "hand_interp_steps": HAND_INTERP_STEPS,
            "grasp_settle_steps": 15,
            "grasp_variant": getattr(self, "deterministic_grasp_variant", "closest"),
        }
        calibration = getattr(self, "grasp_frame_to_eef", None)
        if calibration is not None:
            options["grasp_frame_to_eef"] = torch.as_tensor(
                calibration, dtype=torch.float32, device=self.device
            )
        return PickUpOptions(**options)

    def _plan_pick_place_impl(
        self,
        source_pose: torch.Tensor,
        marker_pose: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        from embodichain.lab.sim.atomic_actions import (
            ActionInvocation,
            EntityState,
            GraspGoal,
            MotionPolicy,
            PlaceGoal,
            PlaceOptions,
            SceneSnapshot,
        )

        target_pose = marker_pose.clone()
        # Place the cube on the marker's top surface.  Using only the cube
        # half-height leaves the cube intersecting the marker by its half
        # thickness; the resulting contact impulse is especially unstable
        # when the cube and marker have exactly zero yaw.
        target_pose[:, 2, 3] += GOAL_MARKER_HALF_HEIGHT + CUBE_HALF_HEIGHT

        endpoints = {"primary": {"motion": CONTROL_PART, "grasp": HAND_CONTROL_PART}}
        pick_binding = self._action_engine.bind_control_parts("pick_up", endpoints)
        place_binding = self._action_engine.bind_control_parts("place", endpoints)
        pick_compiled = self._action_engine.compile(
            (
                ActionInvocation(
                    skill_id="pick_up",
                    goal=GraspGoal(self._cube_semantics),
                    binding=pick_binding,
                    motion_policy=MotionPolicy(sample_count=PICK_SAMPLE_INTERVAL),
                    skill_options=self._pick_up_options(),
                ),
            ),
            self._action_engine.initial_context(
                scene=SceneSnapshot(
                    timestamp=0.0,
                    version=0,
                    entities={CUBE_UID: EntityState(source_pose)},
                ),
                control_dt=self.step_dt,
            ),
        )
        pick_success = pick_compiled.plan_success
        pick_trajectory = self._insert_grasp_hold(pick_compiled.trajectory.positions)
        held = pick_compiled.projected_context.get_held_object(CONTROL_PART)
        if held is None or not bool(pick_success.all().item()):
            return (
                torch.zeros_like(pick_success, dtype=torch.bool),
                self._ensure_nonempty_trajectory(pick_trajectory),
                source_pose,
                target_pose,
            )

        place_compiled = self._action_engine.compile(
            (
                ActionInvocation(
                    skill_id="place",
                    goal=PlaceGoal(torch.bmm(target_pose, held.object_to_eef)),
                    binding=place_binding,
                    motion_policy=MotionPolicy(sample_count=PLACE_SAMPLE_INTERVAL),
                    skill_options=PlaceOptions(
                        lift_height=0.10,
                        hand_interp_steps=HAND_INTERP_STEPS,
                    ),
                ),
            ),
            pick_compiled.projected_context,
        )
        place_success = place_compiled.plan_success
        trajectory = self._ensure_nonempty_trajectory(
            torch.cat((pick_trajectory, place_compiled.trajectory.positions), dim=1)
        )
        return (
            pick_success & place_success,
            trajectory,
            source_pose,
            target_pose,
        )

    def _insert_grasp_hold(self, trajectory: torch.Tensor) -> torch.Tensor:
        close_end_step = min(
            round((PICK_SAMPLE_INTERVAL - HAND_INTERP_STEPS) * 0.6) + HAND_INTERP_STEPS,
            trajectory.shape[1],
        )
        if trajectory.shape[1] < close_end_step:
            return trajectory
        grasp_hold = trajectory[:, close_end_step - 1 : close_end_step].repeat(
            1, GRASP_HOLD_STEPS, 1
        )
        return torch.cat(
            (
                trajectory[:, :close_end_step],
                grasp_hold,
                trajectory[:, close_end_step:],
            ),
            dim=1,
        )

    def _ensure_nonempty_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        if trajectory.shape[1] > 0:
            return trajectory
        return self.robot.get_qpos().clone().unsqueeze(1)

    def _iter_segment_actions(self, trajectory: torch.Tensor) -> Iterable[torch.Tensor]:
        """Replay the plan, then hold its released placement for settling."""
        yield from trajectory.unbind(dim=1)
        if trajectory.shape[1] > 0:
            hold_action = trajectory[:, -1].clone()
            for _ in range(PLACEMENT_SETTLE_CONTROL_STEPS):
                yield hold_action

    def _validate_pick_place(self, plan_success: torch.Tensor) -> torch.Tensor:
        success = plan_success.to(device=self.device) & self.is_task_success()
        if not bool(success.all().item()):
            cube_pos = self._cube.get_local_pose(to_matrix=True)[:, :3, 3]
            goal_pos = self._goal.get_local_pose(to_matrix=True)[:, :3, 3]
            logger.log_warning(
                "Pick-place validation failed: "
                f"planning_success={plan_success.detach().cpu().tolist()}, "
                f"cube_pos={cube_pos.detach().cpu().tolist()}, "
                f"goal_pos={goal_pos.detach().cpu().tolist()}"
            )
        return success

    def is_task_success(self, **kwargs: Any) -> torch.Tensor:
        del kwargs
        return (
            self._placement_in_goal_tolerance()
            & self._placement_released()
            & (self._placement_stable_steps >= PLACEMENT_STABLE_CONTROL_STEPS)
        )

    def compute_task_state(
        self, **kwargs: Any
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        """Report closed-loop success and distance to the visual goal.

        ``EmbodiedEnv.get_info`` consumes this contract on every ``step``;
        keeping it aligned with :meth:`is_task_success` makes policy rollouts
        and expert collection observe the same terminal signal.
        """
        del kwargs
        cube_pos = self._cube.get_local_pose(to_matrix=True)[:, :3, 3]
        goal_pos = self._goal.get_local_pose(to_matrix=True)[:, :3, 3].clone()
        goal_pos[:, 2] += GOAL_MARKER_HALF_HEIGHT + CUBE_HALF_HEIGHT
        distance = torch.linalg.vector_norm(cube_pos - goal_pos, dim=1)
        success = self.is_task_success()
        placement_released = self._placement_released()
        fail = torch.zeros_like(success)
        return (
            success,
            fail,
            {
                "distance_to_goal": distance,
                "placement_released": placement_released,
                "placement_stable_steps": self._placement_stable_steps.clone(),
                "cube_linear_speed": self._cube_linear_speed.clone(),
                "cube_angular_speed": self._cube_angular_speed.clone(),
            },
        )
