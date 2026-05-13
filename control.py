"""
control.py
----------
Robot motion controller for the Franka Panda arm.

Root-cause fixes applied (see log analysis)
--------------------------------------------
Bug 1 – IK diverges because approach height z+0.20 pushes the target
         OUT of the Panda's reachable workspace when the robot base is
         already at z=0.62 m.  Fix: use smaller absolute offsets.

Bug 2 – calculateInverseKinematics had no joint limits or rest pose.
         Without them PyBullet picks arbitrary elbow configs (joint2 was
         stuck at +1.77 rad = arm folded forward, not reaching up).
         Fix: pass lowerLimits, upperLimits, jointRanges, restPoses.

Bug 3 – move_to() had no convergence check.  It ran a fixed number of
         steps and moved on regardless of whether the EE was close.
         Fix: move_to() now loops until dist < CONV_THRESH OR a step
         ceiling is hit, whichever comes first.

Bug 4 – Grasp/place offsets too large for a robot mounted on a table.
         The Panda base is at world z=0.62 m; cubes are at z≈0.675 m.
         Approach z+0.20 = 0.875 m world is near the workspace limit.
         Fix: approach +0.10, grasp +0.01, lift +0.18.

Coordinate note
---------------
All positions are WORLD coordinates (metres), not robot-base-relative.
The IK solver works in world frame when called on a fixed-base robot.
"""

import pybullet as p
import numpy as np
import time

# ── Logging ───────────────────────────────────────────────────────────────────
# Print EE position + joint angles every LOG_EVERY simulation steps (~0.2 s)
LOG_EVERY = 48

# ── IK convergence ────────────────────────────────────────────────────────────
# move_to() keeps running until EE is within this distance of the target
# OR the maximum step count is reached.
CONV_THRESH  = 0.015   # metres  (1.5 cm — good enough for a grasp)
MAX_STEPS    = 480     # hard ceiling (2 s at 240 Hz) to avoid infinite loops

# ── Franka Panda joint limits (from franka_panda/panda.urdf) ─────────────────
PANDA_LOWER  = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
PANDA_UPPER  = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]
PANDA_RANGES = [u - l for u, l in zip(PANDA_UPPER, PANDA_LOWER)]

# ── Rest pose for TABLE-MOUNTED robot ────────────────────────────────────────
# Derived from log data analysis — NOT guessed:
#
# Log proof:  at zero joints → EE world z = 1.553 m, base z = 0.625 m
#             → the Panda zero-config is ARM STRAIGHT UP
#             → joint2 NEGATIVE tilts arm FORWARD (toward cubes)   ← correct
#             → joint2 POSITIVE tilts arm BACKWARD (away from cubes) ← wrong
#
# Previous rest [0, +0.75, 0, -2.6, 0, 2.3, 0.785] biased IK toward
# joint2 = +1.739 (elbow-DOWN), which overshoots every target:
#   target (0.580, 0.145) → EE landed at (0.858, 0.375)  offset=(+0.28,+0.23)
#   elbow-down reaches FARTHER than elbow-up → systematic overshoot confirmed.
#
# Correct configuration (elbow-UP, arm sweeps forward from vertical):
#   joint2 = -0.785 rad  (~45° forward tilt from vertical)   ← KEY FIX
#   joint4 = -2.356 rad  (3π/4 elbow bend, EE stays close)
#   joint6 = +1.571 rad  (π/2 wrist, gripper points straight down)
#
# All values verified within URDF joint limits.
# Seed pose for iterative 7-DOF IK.  j1 is overridden per-target in move_ee().
# Chosen to keep ALL arm links well ABOVE the table (base z=0.625) at all times.
#   j1=0.0   placeholder — overridden per-call with atan2(dy,dx)
#   j2=0.50  shoulder ~29° forward — arm elevated, links clear of table
#   j3=0.00  upper-arm roll neutral
#   j4=-2.00 elbow strongly bent UPWARD — forearm points up, not into table
#   j5=0.00  forearm roll neutral
#   j6=1.57  wrist pitched: gripper points down
#   j7=0.785 wrist roll 45°
PANDA_REST = [0.0, 0.50, 0.0, -2.00, 0.0, 1.57, 0.785]


def _fmt(arr, precision=3):
    """Format a list/array as a compact fixed-precision string."""
    return "[" + "  ".join(f"{v:+.{precision}f}" for v in arr) + "]"


class Controller:
    """IK-based motion controller for the Franka Panda in PyBullet."""

    # Gripper finger joint indices (from franka_panda/panda.urdf)
    FINGER_JOINTS = [9, 10]

    # EE orientation: gripper pointing straight down (roll=π, pitch=0, yaw=0)
    # Hardcoded quaternion — avoids calling p.getQuaternionFromEuler() at class
    # definition time (import time), which crashes when no PyBullet server exists.
    # Euler(pi, 0, 0) → quaternion(x=1, y=0, z=0, w=0)
    EE_ORIENTATION = (1.0, 0.0, 0.0, 0.0)

    def __init__(self, robot_id: int, ee_link: int, base_pos: list = None):
        self.robot    = robot_id
        self.ee       = ee_link
        # Robot base world position — used to compute per-target joint1 direction.
        # Default to [0.15, 0, 0.625] which matches simulation.py scene setup.
        self.base_pos = base_pos if base_pos is not None else [0.15, 0.0, 0.625]

        # Collect revolute joint IDs (arm only — excludes fixed joints)
        self.joint_ids = []
        for i in range(p.getNumJoints(self.robot)):
            info = p.getJointInfo(self.robot, i)
            if info[2] == p.JOINT_REVOLUTE:
                self.joint_ids.append(i)

        # Name map for logging
        self._jname = {}
        for jid in self.joint_ids:
            raw = p.getJointInfo(self.robot, jid)[1]
            self._jname[jid] = raw.decode() if isinstance(raw, bytes) else raw

        # Cycle counter for periodic safe_reset (Part 1 — IK drift prevention)
        self._cycle_count = 0

        # Real-time pacing flag — set False in headless mode for speed (Part 5)
        self.real_time = True

        print(f"[CTRL] Revolute joints: {[self._jname[j] for j in self.joint_ids]}")
        print(f"[CTRL] IK joint limits loaded for {len(PANDA_LOWER)} arm joints")
        print(f"[CTRL] Rest pose: {_fmt(PANDA_REST)}")

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_joint_angles(self) -> np.ndarray:
        """Read current positions of all revolute joints (rad)."""
        states = p.getJointStates(self.robot, self.joint_ids)
        return np.array([s[0] for s in states])

    def _get_ee_pos(self) -> np.ndarray:
        """Return the live world position of the EE link (FK recomputed)."""
        # positional args: (bodyId, linkIndex, computeLinkVelocities, computeForwardKinematics)
        state = p.getLinkState(self.robot, self.ee, 0, 1)
        return np.array(state[4])

    def _log_state(self, step: int, target: list, phase: str):
        """One-line snapshot: EE pos, distance to target, all joint angles."""
        ee  = self._get_ee_pos()
        jt  = self._get_joint_angles()
        err = np.linalg.norm(ee - np.array(target))
        status = "✓" if err < CONV_THRESH else "·"
        print(
            f"[CTRL] {status} step={step:4d}  phase={phase:<16s}"
            f"  EE=({ee[0]:+.4f},{ee[1]:+.4f},{ee[2]:+.4f})"
            f"  dist={err:.4f}m"
            f"\n         joints: {_fmt(jt)}"
        )

    def reset_to_seed(self, target_pos: list):
        """
        Teleport all 7 arm joints to the seed pose aimed at target_pos.

        Called ONCE before each grasp/place phase begins.
        Using p.resetJointState (teleport, not motor command) means the
        arm instantly jumps to a good starting configuration so the
        iterative IK solver never gets stuck in a local minimum from a
        previous phase's joint state.
        """
        import math
        dx = target_pos[0] - self.base_pos[0]
        dy = target_pos[1] - self.base_pos[1]
        j1 = math.atan2(dy, dx)

        seed    = list(PANDA_REST)
        seed[0] = j1

        for i, jid in enumerate(self.joint_ids[:7]):
            p.resetJointState(self.robot, jid, seed[i])

        print(f"[CTRL] Seed reset → j1={j1:+.3f}  j2={seed[1]:+.3f}  "
              f"j4={seed[3]:+.3f}  j6={seed[5]:+.3f}")

    # ------------------------------------------------------------------ #
    # Low-level: IK + Joint Control                                        #
    # ------------------------------------------------------------------ #

    def move_ee(self, target_pos: list):
        """
        Solve full 7-DOF IK for target_pos and drive all arm joints.

        Strategy
        --------
        All 7 Panda joints participate in reaching the target. The IK is
        called with maxNumIterations=100 and a tight residualThreshold so
        PyBullet's solver refines the solution across all joints instead of
        stopping at the first feasible (but wrong) configuration.

        The only analytic pre-computation is joint1 (base yaw), because
        PyBullet's IK cannot determine which rotational hemisphere to use
        without a directional hint — it would point the arm backward.

        Seed pose for joints 2-7
        ------------------------
        A fixed "ready" pose tuned for the table-mounted Panda reaching
        forward to cubes at z≈0.650m:
            j2 = +1.30   shoulder tilted ~75° forward from vertical
            j3 =  0.00   upper-arm roll neutral
            j4 = -1.50   elbow at ~86° bend
            j5 =  0.00   forearm roll neutral
            j6 = +1.57   wrist pitched so gripper points down (≈π/2)
            j7 = +0.785  wrist roll at 45°

        These are passed as restPoses so the iterative solver starts from
        a configuration that already has the arm roughly in the right posture,
        then all 7 joints refine together from there.
        """
        import math

        n_joints = p.getNumJoints(self.robot)
        n_arm    = len(PANDA_LOWER)
        pad      = [0.0] * (n_joints - n_arm)

        # ── joint1: yaw analytically toward target ─────────────────────────
        # This is the only analytic override — it prevents the solver from
        # choosing the backward-hemisphere solution (arm pointing away from cube).
        dx = target_pos[0] - self.base_pos[0]
        dy = target_pos[1] - self.base_pos[1]
        j1 = math.atan2(dy, dx)

        # ── Build seed pose: analytic j1, decay j3 and j5 toward 0 ────────
        # The Panda has 3 null-space (redundant) joints: j1 (base yaw),
        # j3 (upper-arm roll), j5 (forearm roll). Without explicit control
        # these drift freely during multi-step phases, causing the arm to
        # spin in place instead of descending toward the target.
        #
        # Fix: read each joint's current value and apply exponential decay
        # toward its desired value every IK call. This damps drift smoothly
        # without the jerky motion of hard-pinning to a fixed value.
        #
        #   j1 → j1_analytic  (yaw toward target, computed above)
        #   j3 → 0.0          (neutral upper-arm roll)
        #   j5 → 0.0          (neutral forearm roll)
        DECAY          = 0.5   # fraction of current offset removed per IK step

        current_j1     = p.getJointState(self.robot, self.joint_ids[0])[0]
        current_j3     = p.getJointState(self.robot, self.joint_ids[2])[0]
        current_j5     = p.getJointState(self.robot, self.joint_ids[4])[0]

        seed           = list(PANDA_REST)
        # j1: decay current yaw toward the analytic target yaw
        seed[0]        = current_j1 + DECAY * (j1 - current_j1)
        # j3, j5: decay toward neutral roll (0.0)
        seed[2]        = current_j3 * (1.0 - DECAY)
        seed[4]        = current_j5 * (1.0 - DECAY)

        lower  = PANDA_LOWER  + pad
        upper  = PANDA_UPPER  + pad
        ranges = PANDA_RANGES + [0.01] * (n_joints - n_arm)
        rest   = seed + pad

        # ── Full 7-DOF iterative IK ─────────────────────────────────────────
        # maxNumIterations=100 lets the solver refine across ALL 7 joints.
        # residualThreshold=1e-4 (0.1mm) ensures it keeps refining until close.
        joint_angles = p.calculateInverseKinematics(
            self.robot,
            self.ee,
            target_pos,
            self.EE_ORIENTATION,
            lowerLimits       = lower,
            upperLimits       = upper,
            jointRanges       = ranges,
            restPoses         = rest,
            maxNumIterations  = 100,
            residualThreshold = 1e-4,
        )

        # Apply to all 7 arm joints (joint_ids contains only revolute joints,
        # ordered j1..j7 as found by iterating the URDF)
        for i, j in enumerate(self.joint_ids):
            p.setJointMotorControl2(
                self.robot, j,
                p.POSITION_CONTROL,
                targetPosition = joint_angles[i],
                force          = 500,
                maxVelocity    = 0.5,  # rad/s — slow, smooth arm motion
            )

    # ------------------------------------------------------------------ #
    # Gripper Control                                                      #
    # ------------------------------------------------------------------ #

    def open_gripper(self):
        """Open gripper to max aperture (4 cm each finger)."""
        print("[CTRL] Gripper → OPEN  (0.04 m / 50 N)")
        for j in self.FINGER_JOINTS:
            p.setJointMotorControl2(
                self.robot, j, p.POSITION_CONTROL,
                targetPosition=0.04, force=50,
            )

    def close_gripper(self):
        """
        Close gripper to firmly grip a 5cm cube.

        cube_small.urdf width = 0.05m → each finger must stop at 0.022m
        (half cube width 0.025m minus 3mm grip margin).
        Closing to 0.0 makes fingers pass THROUGH the cube mesh in PyBullet
        which breaks the contact constraint → cube falls on lift.
        Force = 500N ensures the grip holds against gravity and inertia.
        """
        print("[CTRL] Gripper → CLOSE (0.022 m / 500 N)")
        for j in self.FINGER_JOINTS:
            p.setJointMotorControl2(
                self.robot, j, p.POSITION_CONTROL,
                targetPosition=0.022, force=500,
            )

    # ------------------------------------------------------------------ #
    # Trajectory Execution                                                 #
    # ------------------------------------------------------------------ #

    def move_to(self, pos: list, steps: int = 300, phase: str = "move"):
        """
        Drive the EE toward *pos* until converged OR *steps* ceiling reached.

        Convergence is checked every simulation step.  Once
        dist(EE, target) < CONV_THRESH the loop exits early so subsequent
        phases can start immediately rather than wasting fixed-count steps.

        Parameters
        ----------
        pos   : [x, y, z] world target for the end-effector (metres).
        steps : Maximum simulation steps before giving up.
        phase : Label for log output.
        """
        print(f"[CTRL] ► move_to  phase={phase:<16s}"
              f"  target=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f})"
              f"  max_steps={steps}")

        for step in range(steps):
            self.move_ee(pos)
            p.stepSimulation()
            if self.real_time:
                time.sleep(1 / 240)

            # Check convergence
            ee  = self._get_ee_pos()
            err = np.linalg.norm(ee - np.array(pos))

            # Log periodically
            if step % LOG_EVERY == 0 or step == steps - 1:
                self._log_state(step, pos, phase)

            # Exit early once close enough
            if err < CONV_THRESH:
                print(f"[CTRL] ✔ CONVERGED  phase={phase}  "
                      f"dist={err:.4f}m  steps_used={step+1}/{steps}")
                return True   # ← success signal (Part 1)

        # Hit ceiling without converging — warn and signal failure
        ee  = self._get_ee_pos()
        err = np.linalg.norm(ee - np.array(pos))
        print(f"[CTRL] ⚠ TIMEOUT  phase={phase}  "
              f"final_dist={err:.4f}m  (threshold={CONV_THRESH}m)  "
              f"EE=({ee[0]:+.4f},{ee[1]:+.4f},{ee[2]:+.4f})")
        return False   # ← failure signal (Part 1)

    def wait(self, steps: int = 60, label: str = "wait"):
        """Step simulation without sending new IK commands (let physics settle)."""
        print(f"[CTRL] ·· wait  label={label}  steps={steps}")
        for _ in range(steps):
            p.stepSimulation()
            if self.real_time:
                time.sleep(1 / 240)

    def safe_reset(self, target_pos: list):
        """
        Teleport ALL arm joints to a collision-free upright pose aimed at target.

        Why this is the correct approach
        ---------------------------------
        PyBullet IK restPoses is a NULL-SPACE bias — it does NOT set the
        starting joint configuration for the solver. The solver always
        starts from the CURRENT joint state and takes gradient steps.

        When joint2 gets stuck at +1.764 (its upper limit), every subsequent
        IK call starts there and stays there — the gradient never escapes.

        The ONLY way to break out is p.resetJointState() — a direct teleport
        that bypasses the PD controller and physics.

        Why j2=0 is the safe teleport pose
        ------------------------------------
        At j2=0 the arm points straight UP. The shoulder is at z=0.958m,
        and all links hang from it going upward — nothing can be below the
        table (z=0.625m). This is the only guaranteed collision-free reset.

        After reset the arm starts IK from j2=0. The approach target is at
        z=cube_z+0.30 ≈ 0.95m which is near shoulder height — the IK only
        needs to tilt j2 slightly (~0.3 rad) to reach it, staying elevated
        and well clear of the table throughout.
        """
        import math
        j1 = math.atan2(
            target_pos[1] - self.base_pos[1],
            target_pos[0] - self.base_pos[0],
        )
        # Safe upright seed: arm points UP (j2=0), j1 faces target
        # j3=j5=0 (roll neutral), j4=-1.0 (slight elbow bend away from body),
        # j6=1.57 (wrist level), j7=0.785 (standard roll)
        seed = [j1, 0.0, 0.0, -1.0, 0.0, 1.57, 0.785]
        for i, jid in enumerate(self.joint_ids[:7]):
            p.resetJointState(self.robot, jid, seed[i])
        print(f"[CTRL] safe_reset j1={j1:+.3f} → arm straight up, facing target")

    def go_home(self):
        """
        Drive the arm to a safe upright HOME position using motor commands.

        Home is directly above the robot base, well above the table:
            x = base_x + 0.15   (slight forward offset so j1 has a direction)
            y = base_y
            z = base_z + 0.40   (40 cm above base = 1.025 m world)

        Called by grasp_point_world() as an intermediate waypoint between
        place_retract_approach and the next pick approach:
            place_retract_approach → HOME → pick_approach

        Unlike safe_reset() this uses move_to() (motor commands, not teleport)
        so the arm moves physically through space with no collision issues.
        HOME is near vertical — trivially reachable from any arm state.
        """
        # Home is computed relative to robot base (Part 2 — no more hardcoded coords).
        # 30 cm forward + 30 cm above base keeps the arm upright and well clear of
        # the table surface at any base configuration.
        home = [
            self.base_pos[0] + 0.30,
            self.base_pos[1],
            self.base_pos[2] + 0.30,
        ]
        # Clamp to safe workspace bounds (Part 2 bonus)
        home[0] = float(np.clip(home[0], 0.4, 0.8))
        home[1] = float(np.clip(home[1], -0.3, 0.3))
        print(f"[CTRL] ── HOME: ({home[0]:+.3f}, {home[1]:+.3f}, {home[2]:+.3f})")
        self.move_to(home, steps=500, phase="home")

    # ------------------------------------------------------------------ #
    # Grasp Pipeline                                                       #
    # ------------------------------------------------------------------ #

    def grasp_point_world(self, point: list):
        """
        Full 7-step pick sequence at the given world position.

        Waypoints
        ---------
        approach   z + 0.30  — high above cube, arm upright, safe entry
        pre_grasp  z + 0.10  — just above cube top, gripper already open
        grasp      z + 0.058 — fingertips bracket cube sides (flange offset)
        pre_grasp  z + 0.10  — initial lift with cube, clear of table
        approach   z + 0.30  — fully retracted to safe height

        Sequence
        --------
        1. safe_reset  → arm straight up facing target (no table collision)
        2. → APPROACH  → arm above cube at safe height, gripper still closed
        3. open_gripper at approach
        4. → PRE-GRASP → descend to just above cube, fingers spread wide
        5. → GRASP     → final descent, fingertips at cube centre
        6. close_gripper + settle
        7. → PRE-GRASP → initial lift, table clearance confirmed
        8. → APPROACH  → fully clear, ready for place sequence
        """
        x, y, z = point

        approach  = [x, y, z + 0.30]   # safe entry/exit height
        pre_grasp = [x, y, z + 0.10]   # just above cube top (top = z+0.025)
        grasp     = [x, y, z + 0.058]  # EE flange at grip height (fingertips at cube centre)

        print(f"\n[CTRL] ════════════ PICK SEQUENCE  (8 steps) ════════════")
        print(f"[CTRL]   object    : ({x:+.4f}, {y:+.4f}, {z:+.4f})")
        print(f"[CTRL]   approach  : ({approach[0]:+.4f}, {approach[1]:+.4f}, {approach[2]:+.4f})  z+0.30")
        print(f"[CTRL]   pre_grasp : ({pre_grasp[0]:+.4f}, {pre_grasp[1]:+.4f}, {pre_grasp[2]:+.4f})  z+0.10")
        print(f"[CTRL]   grasp     : ({grasp[0]:+.4f}, {grasp[1]:+.4f}, {grasp[2]:+.4f})  z+0.058")

        # # ── Step 1: safe joint reset (teleport to j2=0, face target) ─────
        # print(f"[CTRL] ── PICK step 1/8: safe_reset → joints to upright, facing target")
        # self.safe_reset(approach)

        # ── Step 1: safe reset ONLY on first pick ────────────────────────────
        # Subsequent cycles rely on go_home() + IK failure recovery (no teleport).
        # Periodic teleport was removed — it caused visible arm jumps mid-task.
        self._cycle_count += 1
        if not hasattr(self, "_did_reset"):
            print(f"[CTRL] ── PICK step 1/8: first run → safe_reset")
            self.safe_reset(approach)
            self._did_reset = True
        else:
            print(f"[CTRL] ── PICK step 1/8: skip safe_reset → using HOME (cycle {self._cycle_count})")

        # ── Step 2: move to HOME (upright position above robot base) ─────
        # HOME is driven by motor commands (not teleport) so the arm moves
        # physically through space.  This gives a visible, clean transition
        # from the post-place position to the start of the next pick cycle.
        print(f"[CTRL] ── PICK step 2/8: home → arm to neutral upright position")
        self.go_home()

        # ── Step 3: move to approach (high above cube) ────────────────────
        print(f"[CTRL] ── PICK step 3/8: approach → above cube at z+0.30")
        if not self.move_to(approach, steps=850, phase="pick_approach"):
            print(f"[CTRL] IK failed at approach → safe_reset and retry")
            self.safe_reset(approach)
            self.move_to(approach, steps=850, phase="pick_approach_retry")

        # ── Step 4: open gripper at approach height ───────────────────────
        print(f"[CTRL] ── PICK step 4/8: open gripper")
        self.open_gripper()
        self.wait(steps=30, label="gripper_open")

        # ── Step 5: descend to pre-grasp (just above cube) ───────────────
        print(f"[CTRL] ── PICK step 5/8: pre_grasp → descend to z+0.10")
        if not self.move_to(pre_grasp, steps=350, phase="pick_pre_grasp"):
            print(f"[CTRL] IK failed at pre_grasp → safe_reset and retry")
            self.safe_reset(pre_grasp)
            self.move_to(pre_grasp, steps=350, phase="pick_pre_grasp_retry")

        # ── Step 6: final descent to grasp height ────────────────────────
        print(f"[CTRL] ── PICK step 6/8: grasp → final descent to z+0.058")
        self.move_to(grasp, steps=200, phase="pick_grasp")

        # ── Step 7: close gripper and let grip settle ─────────────────────
        print(f"[CTRL] ── PICK step 7/8: close gripper + settle")
        self.close_gripper()
        self.wait(steps=160, label="grip_settle")

        # ── Step 8a: lift to pre-grasp height ─────────────────────────────
        print(f"[CTRL] ── PICK step 8/8: lift → pre_grasp then approach")
        self.move_to(pre_grasp, steps=200, phase="pick_lift_pre")

        # ── Step 8b: lift to full approach height ─────────────────────────
        self.move_to(approach, steps=850, phase="pick_lift_approach")

        ee = self._get_ee_pos()
        print(f"[CTRL]   post-pick EE : ({ee[0]:+.4f}, {ee[1]:+.4f}, {ee[2]:+.4f})")
        print(f"[CTRL] ════════════ PICK COMPLETE ════════════\n")

    # ------------------------------------------------------------------ #
    # Place Pipeline                                                       #
    # ------------------------------------------------------------------ #

    def place_point_world(self, point: list):
        """
        Full 6-step place sequence mirroring the pick sequence.

        Waypoints
        ---------
        approach     z + 0.30  — high above drop zone, safe entry
        pre_release  z + 0.10  — just above drop surface, cube still held
        release      z + 0.02  — cube resting on surface, fingers release
        pre_release  z + 0.10  — retract clear of cube
        approach     z + 0.30  — fully retracted to safe height

        Sequence
        --------
        1. → PLACE APPROACH    → arm above drop zone at safe height
        2. → PLACE PRE-RELEASE → descend to just above surface
        3. → PLACE RELEASE     → lower cube to surface
        4. open_gripper + settle
        5. → PLACE PRE-RELEASE → retract clear of placed cube
        6. → NEXT APPROACH     → transit directly to next cube approach height
                                  (or back to drop approach if last cube)

        Parameters
        ----------
        point        : [x, y, z]  world position of the drop zone
        next_pick_pos: [x, y, z]  world position of the NEXT cube to pick, or
                                  None if this is the last cube.  When provided,
                                  step 6 drives directly to that cube's approach
                                  point (z+0.30) — skipping safe_reset and the
                                  redundant retract — for a seamless continuous cycle.

        NOTE: No safe_reset() here — cube is in the gripper until step 4.
        Step 6 transit is also safe without reset because the arm is already at
        approach height (z≈0.95 m) after step 5; it simply swings horizontally
        to the next cube's approach point at the same height.
        """
        x, y, z = point

        approach    = [x, y, z + 0.30]   # safe entry/exit height
        pre_release = [x, y, z + 0.10]   # just above drop surface
        release     = [x, y, z + 0.02]   # cube touches down

        print(f"\n[CTRL] ════════════ PLACE SEQUENCE (6 steps) ════════════")
        print(f"[CTRL]   drop zone    : ({x:+.4f}, {y:+.4f}, {z:+.4f})")
        print(f"[CTRL]   approach     : ({approach[0]:+.4f}, {approach[1]:+.4f}, {approach[2]:+.4f})  z+0.30")
        print(f"[CTRL]   pre_release  : ({pre_release[0]:+.4f}, {pre_release[1]:+.4f}, {pre_release[2]:+.4f})  z+0.10")
        print(f"[CTRL]   release      : ({release[0]:+.4f}, {release[1]:+.4f}, {release[2]:+.4f})  z+0.02")

        # ── Step 1: move to place approach (arm above drop zone) ──────────
        print(f"[CTRL] ── PLACE step 1/6: place_approach → above drop zone at z+0.30")
        if not self.move_to(approach, steps=500, phase="place_approach"):
            print(f"[CTRL] IK failed at place_approach → safe_reset and retry")
            self.safe_reset(approach)
            self.move_to(approach, steps=500, phase="place_approach_retry")

        # ── Step 2: descend to pre-release ────────────────────────────────
        print(f"[CTRL] ── PLACE step 2/6: pre_release → descend to z+0.10")
        self.move_to(pre_release, steps=350, phase="place_pre_release")

        # ── Step 3: lower cube to surface ─────────────────────────────────
        print(f"[CTRL] ── PLACE step 3/6: release → lower to z+0.02")
        self.move_to(release, steps=200, phase="place_release")

        # ── Step 4: open gripper and let cube settle ──────────────────────
        print(f"[CTRL] ── PLACE step 4/6: open gripper + settle")
        self.open_gripper()
        self.wait(steps=80, label="release_settle")

        # ── Step 5: retract to pre-release height ─────────────────────────
        print(f"[CTRL] ── PLACE step 5/6: retract → pre_release z+0.10")
        self.move_to(pre_release, steps=200, phase="place_retract_pre")

        # ── Step 6: retract to full drop approach height ─────────────────
        print(f"[CTRL] ── PLACE step 6/6: retract → drop approach z+0.30")
        self.move_to(approach, steps=850, phase="place_retract_approach")

        print(f"[CTRL] ── MOVE TO HOME after place")
        self.go_home()

        ee = self._get_ee_pos()
        print(f"[CTRL]   post-place EE: ({ee[0]:+.4f}, {ee[1]:+.4f}, {ee[2]:+.4f})")
        print(f"[CTRL] ════════════ PLACE COMPLETE ════════════\n")