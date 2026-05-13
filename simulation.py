"""
simulation.py
-------------
PyBullet simulation environment for the Franka Panda pick-and-place task.

Responsibilities:
  - Connect to PyBullet (GUI or headless DIRECT mode).
  - Load the plane, table, and Franka Panda robot.
  - Spawn N randomly placed / randomly coloured cubes on the table.
  - Provide overhead camera and a true wrist-mounted camera (follows the
    panda_hand link in real-time).
  - Both cameras return: RGB, metric-depth, segmentation mask,
    view_matrix, projection_matrix.

Camera note
-----------
PyBullet's getCameraImage returns 5 channels:
  img[2] → RGBA         (H, W, 4) uint8
  img[3] → depth buffer (H, W)    float32   0..1  (non-linear)
  img[4] → segmentation (H, W)    int32     body_id | (link_idx << 24)

We linearise the depth buffer with:
    d = far * near / (far - (far - near) * depth_buffer)


The Franka Panda hand link index is 11 (panda_hand).  Link 8 is the flange
centre – acceptable for IK target but the hand mesh/camera sits on link 11.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import random


class Simulation:
    """Wraps the PyBullet physics world and scene assets."""

    # Camera image resolution (pixels)
    IMG_W = 640
    IMG_H = 480

    # Depth-buffer linearisation parameters (near/far clip planes, metres)
    NEAR = 0.01
    FAR  = 3.0

    # Panda link indices
    EE_LINK   = 8   # flange – used for IK
    HAND_LINK = 11  # panda_hand – wrist camera is mounted here

    def __init__(self, gui: bool = True):
        """
        Parameters
        ----------
        gui : bool
            True  → open the PyBullet OpenGL window (interactive).
            False → headless DIRECT mode (faster, no window).
        """
        self.gui = gui

        # Connect to the physics server
        if gui:
            self.client = p.connect(p.GUI)
            # Disable the three synthetic camera preview panels in the
            # Explorer tab (RGB / Depth / Segmentation). These panels
            # flicker and compete with the overhead camera feed.
            # The overhead view is shown in the dedicated OpenCV window instead.
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW,          0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW,        0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW,   0)
        else:
            self.client = p.connect(p.DIRECT)

        # Make built-in URDFs (plane, table, panda, …) findable
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Earth-like gravity along –Z
        p.setGravity(0, 0, -9.81)

        self._load_scene()

    # ------------------------------------------------------------------ #
    # Scene Setup                                                          #
    # ------------------------------------------------------------------ #

    def _load_scene(self):
        

        # Ground plane
        self.plane = p.loadURDF("plane.urdf")

        # Table: centred at x=0.6 m, surface at z = 0.625 m
        self.table = p.loadURDF(
            "table/table.urdf",
            basePosition=[0.6, 0, 0],
        )
        self._table_surface_z = 0.625   # metres — measured from pybullet table.urdf

        # Franka Panda — mounted on the table surface, base at z = TABLE_SURFACE_Z
        # Robot is placed at x=0.3 so it faces toward the centre of the table
        self.robot = p.loadURDF(
            "franka_panda/panda.urdf",
            basePosition=[0.15, 0.0, self._table_surface_z],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )

        # Link indices (see class constants)
        self.ee_link   = self.EE_LINK    # flange  – IK target
        self.hand_link = self.HAND_LINK  # panda_hand – wrist camera

        # Spawn 3 randomly placed coloured cubes on the table
        self.spawn_random_cubes(3)

    # ------------------------------------------------------------------ #
    # Accessors                                                            #
    # ------------------------------------------------------------------ #

    def get_robot_id(self) -> int:
        """Return the PyBullet body ID of the Panda robot."""
        return self.robot

    def get_ee_link(self) -> int:
        """Return the flange link index used as the IK end-effector target."""
        return self.ee_link

    def get_hand_link(self) -> int:
        """Return the panda_hand link index (wrist camera mount)."""
        return self.hand_link

    def get_cube_ids(self) -> list:
        """Return the list of PyBullet body IDs for the spawned cubes."""
        return self.cubes

    @property
    def TABLE_SURFACE_Z(self):
        """Table surface height in world metres (read-only property)."""
        return self._table_surface_z

    # ------------------------------------------------------------------ #
    # Cameras                                                              #
    # ------------------------------------------------------------------ #

    def _decode(self, img):
        """Decode raw PyBullet getCameraImage output into (rgb, depth, seg)."""
        rgb          = np.reshape(img[2], (self.IMG_H, self.IMG_W, 4)).astype(np.uint8)
        depth_buffer = np.reshape(img[3], (self.IMG_H, self.IMG_W)).astype(np.float32)
        seg          = np.reshape(img[4], (self.IMG_H, self.IMG_W)).astype(np.int32)
        depth = (self.FAR * self.NEAR / (
            self.FAR - (self.FAR - self.NEAR) * depth_buffer
        )).astype(np.float32)
        return rgb, depth, seg

    def _capture_overhead(self, view_matrix, projection_matrix):
        """
        Render with ER_TINY_RENDERER so the result appears in PyBullet's
        Explorer tab (Synthetic Camera RGB/Depth/Segmentation panels).
        Only the overhead camera should use this renderer — if the wrist
        camera also calls it the Explorer panels flicker between both feeds.
        """
        img = p.getCameraImage(
            self.IMG_W, self.IMG_H,
            view_matrix, projection_matrix,
            renderer=p.ER_TINY_RENDERER,
        )
        rgb, depth, seg = self._decode(img)
        return rgb, depth, seg, view_matrix, projection_matrix

    def _capture_wrist(self, view_matrix, projection_matrix):
        """
        Render with ER_BULLET_HARDWARE_OPENGL so the image is computed
        offscreen without writing to the Explorer tab panels.
        The wrist feed is shown in a separate OpenCV window ("Wrist RGB")
        and must NOT share the Explorer renderer with the overhead camera.
        """
        try:
            img = p.getCameraImage(
                self.IMG_W, self.IMG_H,
                view_matrix, projection_matrix,
                renderer=p.ER_BULLET_HARDWARE_OPENGL,
            )
        except Exception:
            # Fall back to tiny renderer if OpenGL offscreen is unavailable
            img = p.getCameraImage(
                self.IMG_W, self.IMG_H,
                view_matrix, projection_matrix,
                renderer=p.ER_TINY_RENDERER,
            )
        rgb, depth, seg = self._decode(img)
        return rgb, depth, seg, view_matrix, projection_matrix

    def get_overhead_camera(self):
        """
        Overhead (top-down) camera, fixed above the table centre.

        Camera is stationary – pose does not change during the simulation,
        so the overhead view is stable and used for initial detection.

        Returns
        -------
        rgb   : (H, W, 4)  uint8   RGBA image
        depth : (H, W)     float32 metric depth in metres
        seg   : (H, W)     int32   segmentation mask (PyBullet body IDs)
        view_matrix       : 16-element flat list (column-major 4×4)
        projection_matrix : 16-element flat list (column-major 4×4)
        """
        camera_pos    = [0.6, 0, 1.3]    # 67 cm above table (table at x=0.6)
        camera_target = [0.6, 0, 0.65]   # centred on table top

        view_matrix = p.computeViewMatrix(
            cameraEyePosition    = camera_pos,
            cameraTargetPosition = camera_target,
            cameraUpVector       = [0, 1, 0],
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov    = 60,
            aspect = self.IMG_W / self.IMG_H,
            nearVal= self.NEAR,
            farVal = self.FAR,
        )

        return self._capture_overhead(view_matrix, projection_matrix)

    def get_wrist_camera(self):
        
        # ── Get live world pose of panda_hand (link 11) ────────────────────
        # computeForwardKinematics=1  is CRITICAL – forces PyBullet to recompute
        # the full FK chain so the pose reflects current joint angles.
        # positional: (bodyId, linkIndex, computeLinkVelocities=0, computeForwardKinematics=1)
        state = p.getLinkState(self.robot, self.hand_link, 0, 1)
        hand_pos   = np.array(state[4])   # world position  (3,)
        hand_quat  = state[5]             # world quaternion (4,)

        # ── Build rotation matrix from the live quaternion ─────────────────
        # getMatrixFromQuaternion returns a flat 9-element row-major 3×3 matrix
        rot = np.array(p.getMatrixFromQuaternion(hand_quat)).reshape(3, 3)

        # Column vectors of rot:
        #   rot[:,0] = local X  (right)
        #   rot[:,1] = local Y  (up of the hand)
        #   rot[:,2] = local Z  (pointing out of the palm / forward)

        # Camera body sits 8 cm behind the palm (along −Z of the hand frame)
        eye    = hand_pos + rot @ np.array([0.0,  0.0, -0.08])

        # Camera looks 15 cm out along the palm's +Z (toward the object below)
        target = hand_pos + rot @ np.array([0.0,  0.0,  0.15])

        # Up vector = hand local Y  (keeps the horizon level as wrist rotates)
        up     = rot[:, 1].tolist()

        view_matrix = p.computeViewMatrix(
            cameraEyePosition    = eye.tolist(),
            cameraTargetPosition = target.tolist(),
            cameraUpVector       = up,
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov    = 70,
            aspect = self.IMG_W / self.IMG_H,
            nearVal= self.NEAR,
            farVal = self.FAR,
        )

        return self._capture_wrist(view_matrix, projection_matrix)

    # ------------------------------------------------------------------ #
    # Object Spawning                                                      #
    # ------------------------------------------------------------------ #

    def spawn_random_cubes(self, n: int):
       
        self.cubes = []

        print(f"[SIM] Spawning {n} cubes (random drop, random yaw) ...")

        # Robot base position (world)
        BASE_X, BASE_Y = 0.15, 0.0
        # Reach limits — comfortable IK zone
        MIN_REACH = 0.28   # m — avoid near-singularity
        MAX_REACH = 0.72   # m — slightly extended reach envelope
        # Minimum separation between cube centres (spawn-time check)
        MIN_SEP   = 0.09   # m — reduced so scenes are more varied
        MAX_TRIES = 300    # rejection-sampling ceiling per cube

        placed = []   # (x, y) of accepted spawn positions

        for _ in range(n):
            for attempt in range(MAX_TRIES):
                # Wider bounding box for more positional variety
                x = random.uniform(0.35, 0.80)
                y = random.uniform(-0.40, 0.40)

                # Check 1: within reachable annulus
                r = ((x - BASE_X)**2 + (y - BASE_Y)**2) ** 0.5
                if not (MIN_REACH <= r <= MAX_REACH):
                    continue

                # Check 2: far enough from all previously placed cubes
                too_close = any(
                    ((x - px)**2 + (y - py)**2) < MIN_SEP**2
                    for px, py in placed
                )
                if too_close:
                    continue

                break   # accepted

            placed.append((x, y))

            # Random yaw (rotation around Z) so cubes land at different angles
            yaw   = random.uniform(0, 2 * 3.14159)
            quat  = p.getQuaternionFromEuler([0, 0, yaw])

            # Drop from a random height above the table (5–25 cm) so each
            # cube falls naturally and lands slightly differently every run
            drop_height = random.uniform(0.05, 0.25)
            spawn_z     = self._table_surface_z + 0.025 + drop_height

            cube_id = p.loadURDF("cube_small.urdf", [x, y, spawn_z],
                                 baseOrientation=quat)

            # Vivid saturated random colour — avoid near-black/near-white
            h = random.random()          # hue 0–1
            import colorsys
            r_c, g_c, b_c = colorsys.hsv_to_rgb(h, 0.85, 0.95)
            p.changeVisualShape(cube_id, -1, rgbaColor=[r_c, g_c, b_c, 1.0])

            self.cubes.append(cube_id)
            print(f"[SIM]   cube_id={cube_id}  "
                  f"pos=({x:.3f}, {y:.3f}, {spawn_z:.3f})  "
                  f"yaw={yaw:.2f} rad  drop={drop_height*100:.0f} cm")

    # ------------------------------------------------------------------ #
    # Simulation Step                                                       #
    # ------------------------------------------------------------------ #

    def step(self):
        """
        Advance the physics simulation by one time-step (1/240 s).
        In GUI mode, sleep to keep real-time pacing.
        """
        p.stepSimulation()
        if self.gui:
            time.sleep(1 / 240)
