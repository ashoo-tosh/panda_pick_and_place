"""
main.py
-------
Entry point for the autonomous pick-and-place pipeline.

Pipeline
--------
1.  Init simulation (GUI or headless).
2.  Settle physics (2 s).
3.  Capture overhead RGB-D + segmentation.
4.  Detect cube centroids with seg-mask vision (stable, no flicker).
5.  For each cube:
      a. grasp_point_world()  → pick
      b. place_point_world()  → drop at DROP_ZONE
6.  Live camera windows refresh DURING every move_to() call via a
    background display thread, so you always see the latest view.

Camera windows (GUI mode)
-------------------------
  "Overhead RGB"       – colour view from above (annotated with detections)
  "Overhead Depth"     – jet-coloured depth, clipped 0.6–1.2 m
  "Overhead Seg Mask"  – stable segmentation, one colour per body ID
  "Wrist RGB"          – follows panda_hand link (live, using FK=1)

Debug output (terminal)
-----------------------
  [SIM]   simulation lifecycle messages
  [VIS]   detection results: cube_id, pixel, depth, world coords
  [CTRL]  every motion phase: target, EE position, joint angles every 0.2 s

Usage
-----
    python main.py              # GUI + full logging
    python main.py --headless   # headless, no windows
"""

import argparse
import threading
import time
import cv2
import numpy as np
import pybullet as p

from simulation import Simulation
from vision     import Vision
from control    import Controller


# ── Config ───────────────────────────────────────────────────────────────────

# Three drop zones spaced 11-15cm apart — cubes placed here in sequence.
# All positions are on the table surface (z=0.650) and within arm reach.
DROP_ZONES = [
    [0.55, -0.15, 0.650],   # zone 1
    [0.55,  0.00, 0.650],   # zone 2  (15cm offset in y)
    [0.65, -0.10, 0.650],   # zone 3  (10cm offset in x)
]
SETTLE_TIME_S = 2.0                    # physics settle time at startup


# ── Argument parsing ─────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Panda Pick-and-Place")
    parser.add_argument("--headless", action="store_true",
                        help="No GUI windows (for CI / automated testing)")
    return parser.parse_args()


# ── Live display thread ───────────────────────────────────────────────────────

class CameraDisplay:
    """
    Background thread that refreshes all camera windows every ~33 ms
    (≈30 fps display rate) independent of the main control loop.

    The main thread writes new frames to shared numpy arrays; this thread
    reads and shows them.  A threading.Lock prevents torn reads.
    """

    def __init__(self, sim: Simulation, vision: Vision, cube_ids: list):
        self.sim       = sim
        self.vision    = vision
        self.cube_ids  = cube_ids
        self._stop     = threading.Event()
        self._lock     = threading.Lock()
        self._detections = []    # latest detections (set from main thread)
        self._thread   = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2.0)
        cv2.destroyAllWindows()

    def set_detections(self, detections: list):
        with self._lock:
            self._detections = detections

    def _loop(self):
        """Continuously grab camera images and update windows."""
        while not self._stop.is_set():

            try:
                # ── Wrist camera FIRST ───────────────────────────────────
                # Must be rendered BEFORE overhead so that the overhead
                # camera is always the LAST getCameraImage call in each loop.
                # PyBullet's Explorer tab shows whichever camera rendered last,
                # so ending on overhead keeps those panels showing overhead only.
                rgb_wr, dep_wr, seg_wr, _, _ = self.sim.get_wrist_camera()
                bgr_wr = cv2.cvtColor(
                    rgb_wr[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR
                )
                cv2.imshow("Wrist RGB", bgr_wr)

                # ── Overhead camera LAST ─────────────────────────────────
                # Rendered last → Explorer tab always shows overhead view.
                rgb_oh, dep_oh, seg_oh, vm_oh, pm_oh = \
                    self.sim.get_overhead_camera()

                bgr_oh = cv2.cvtColor(
                    rgb_oh[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2BGR
                )

                with self._lock:
                    dets = list(self._detections)

                annotated = self.vision.annotate_detections(bgr_oh, dets)
                dep_vis   = self.vision.make_depth_visual(dep_oh)
                seg_vis   = self.vision.make_seg_visual(seg_oh, self.cube_ids)

                cv2.imshow("Overhead RGB",      annotated)
                cv2.imshow("Overhead Depth",    dep_vis)
                cv2.imshow("Overhead Seg Mask", seg_vis)

            except Exception as exc:
                # Don't crash the display thread on transient PyBullet errors
                print(f"[DISP] Warning: {exc}")

            key = cv2.waitKey(33) & 0xFF
            if key == ord("q"):
                self._stop.set()
                break

        cv2.destroyAllWindows()

    def should_quit(self) -> bool:
        return self._stop.is_set()


# ── Helpers ──────────────────────────────────────────────────────────────────

def settle(sim: Simulation, seconds: float):
    """Step simulation for *seconds* real-time seconds (physics warm-up)."""
    steps = int(seconds * 240)
    print(f"[SIM] Settling for {seconds} s  ({steps} steps) …")
    for _ in range(steps):
        sim.step()
    print("[SIM] Settle done.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    gui  = not args.headless

    # ── 1. Init simulation ───────────────────────────────────────────────
    print("[SIM] Starting simulation …")
    sim = Simulation(gui=gui)
    print(f"[SIM] Robot body ID : {sim.get_robot_id()}")
    print(f"[SIM] EE link index : {sim.get_ee_link()}  (flange, IK target)")
    print(f"[SIM] Hand link idx : {sim.get_hand_link()}  (panda_hand, wrist cam)")
    print(f"[SIM] Cube IDs      : {sim.get_cube_ids()}")

    # Pass robot base position so IK can compute per-target joint1 direction.
    # This prevents the arm from pointing backward away from the cubes.
    robot_base_pos = [0.15, 0.0, sim.TABLE_SURFACE_Z]
    ctrl: Controller = Controller(
        sim.get_robot_id(), sim.get_ee_link(), base_pos=robot_base_pos
    )
    # Part 5: disable real-time sleep in headless mode for maximum speed
    ctrl.real_time = gui
    vision = Vision()

    # ── 2. Physics settle ────────────────────────────────────────────────
    settle(sim, SETTLE_TIME_S)



    # ── 3. Helper: detect current positions of remaining cubes ─────────
    def detect_cubes(cube_ids: list) -> list:
        """
        Capture one fresh overhead frame and return world positions.
        Called before EACH grasp so the arm always uses the current
        cube position, not a stale reading from the initial scan.
        Falls back to PyBullet ground-truth for any missed cube.
        """
        rgb, depth, seg, vm, pm = sim.get_overhead_camera()
        dets = vision.detect_objects_seg(rgb, depth, seg, vm, pm, cube_ids)
        detected_ids = {d["cube_id"] for d in dets}
        for cid in cube_ids:
            if cid not in detected_ids:
                gt_pos, _ = p.getBasePositionAndOrientation(cid)
                gt_pos    = np.array(gt_pos)
                # Apply same x-filter as vision — skip if cube is unreachable
                # (knocked near robot base or off the table surface)
                if gt_pos[0] < 0.38:
                    print(f"[VIS] GT SKIPPED  cube_id={cid}"
                          f"  x={gt_pos[0]:+.4f} unreachable (< 0.38m)")
                    continue
                print(f"[VIS] GT fallback  cube_id={cid}"
                      f"  world=({gt_pos[0]:+.4f},{gt_pos[1]:+.4f},{gt_pos[2]:+.4f})")
                dets.append({"cube_id": cid, "world_pos": gt_pos,
                             "pixel": (-1, -1), "area": 0.0})
        return dets

    # Initial scan just for the display thread start
    print("[SIM] Initial overhead scan …")
    detections = detect_cubes(sim.get_cube_ids())
    print(f"[VIS] Found {len(detections)} cubes:")
    for i, d in enumerate(detections):
        wp = d["world_pos"]
        print(f"       [{i+1}] cube_id={d['cube_id']}"
              f"  world=({wp[0]:+.4f}, {wp[1]:+.4f}, {wp[2]:+.4f})")

    # ── 4. Start live display thread (GUI only) ──────────────────────────
    display = None
    if gui:
        display = CameraDisplay(sim, vision, sim.get_cube_ids())
        display.set_detections(detections)
        display.start()
        print("[DISP] Camera windows started (press 'q' in any window to quit).")

    # ── 4b. START button (GUI only) ──────────────────────────────────────
    # A PyBullet debug button appears in the GUI sidebar.
    # The main loop spins (keeping physics + cameras alive) until pressed.
    if gui:
        btn = p.addUserDebugParameter('START PICKING', 1, 0, 0)
        print('[MAIN] =================================================')
        print('[MAIN]  Press  START PICKING  in the PyBullet sidebar')
        print('[MAIN]  (left panel) to begin the pick-and-place task.')
        print('[MAIN] =================================================')

        last_val = p.readUserDebugParameter(btn)
        while True:
            sim.step()   # keep physics + cameras alive while waiting
            cur_val = p.readUserDebugParameter(btn)
            if cur_val != last_val:  # button clicked — value increments
                print('[MAIN] START pressed — beginning pick-and-place!')
                break
            if display and display.should_quit():
                print('[MAIN] Quit before start.')
                display.stop()
                return


    # ── 5. Helper: check if cube is still in the gripper ───────────────
    def cube_is_held(cid: int) -> bool:
        """
        Return True if the cube is currently held by the gripper.
        Uses PyBullet ground-truth: cube z should be well above the table
        surface (z=0.650m). If it has fallen it will be near z=0.650.
        Threshold: cube z > 0.750m means it was lifted off the table.
        """
        pos, _ = p.getBasePositionAndOrientation(cid)
        held = pos[2] > 0.750
        if not held:
            print(f"[MAIN] ⚠ DROP DETECTED  cube_id={cid}"
                  f"  cube_z={pos[2]:+.4f}  (expected >0.750 if held)")
        return held

    def cube_is_placed(cid: int, drop: list) -> bool:
        """
        Return True if the cube landed near the intended drop zone.
        Checks: cube z ≈ table surface (0.620–0.700m) AND horizontal
        distance from drop zone centre < 0.15m.
        """
        pos, _ = p.getBasePositionAndOrientation(cid)
        dxy = ((pos[0] - drop[0])**2 + (pos[1] - drop[1])**2) ** 0.5
        on_surface = 0.620 < pos[2] < 0.700
        near_zone  = dxy < 0.15
        placed = on_surface and near_zone
        if not placed:
            print(f"[MAIN] ⚠ PLACE MISS  cube_id={cid}"
                  f"  cube=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})"
                  f"  drop=({drop[0]:+.3f},{drop[1]:+.3f})"
                  f"  dxy={dxy:.3f}m  on_surface={on_surface}")
        return placed

    # ── 5. Pick-and-place loop with drop-recovery ────────────────────────
    # Each cube gets up to MAX_RETRIES attempts.
    # If the cube falls during lift:   re-detect on table, retry pick.
    # If the cube falls during place:  re-detect on table, retry pick+place.
    MAX_RETRIES = 3
    picked_ids: list = []
    failed_ids: list = []   # Part 7: track genuinely failed cubes separately

    for idx in range(len(sim.get_cube_ids())):

        if display and display.should_quit():
            print("[MAIN] User quit early.")
            break

        remaining = [c for c in sim.get_cube_ids() if c not in picked_ids]
        if not remaining:
            break

        drop = DROP_ZONES[idx % len(DROP_ZONES)]

        # ── Per-cube retry loop ──────────────────────────────────────────
        for attempt in range(1, MAX_RETRIES + 1):

            if display and display.should_quit():
                break

            # ── Fresh detection ──────────────────────────────────────────
            print(f"\n[MAIN] Scanning for cube {idx+1}/{len(sim.get_cube_ids())}"
                  f"  (attempt {attempt}/{MAX_RETRIES}) …")
            current_dets = detect_cubes(remaining)
            if not current_dets:
                print("[MAIN] ✗ No cube detected — skipping this slot.")
                break

            det = current_dets[0]
            cid = det["cube_id"]
            wp  = det["world_pos"]

            if display:
                display.set_detections(current_dets)

            print(f"\n[MAIN] ══════════════════════════════════════════")
            print(f"[MAIN]  Object {idx+1}/{len(sim.get_cube_ids())}"
                  f"  cube_id={cid}  attempt={attempt}/{MAX_RETRIES}")
            print(f"[MAIN]  Target pos: ({wp[0]:+.4f}, {wp[1]:+.4f}, {wp[2]:+.4f})")
            print(f"[MAIN]  Drop zone : {drop}")
            print(f"[MAIN] ══════════════════════════════════════════")

            # ── PICK ─────────────────────────────────────────────────────
            ctrl.grasp_point_world(list(wp))

            # ── Check cube is held after pick ────────────────────────────
            if not cube_is_held(cid):
                print(f"[MAIN] ↺ Cube dropped during pick — retrying"
                      f" (attempt {attempt}/{MAX_RETRIES})")
                if attempt == MAX_RETRIES:
                    print(f"[MAIN] ✗ Max retries reached for cube_id={cid}."
                          f" Moving to next cube.")
                    failed_ids.append(cid)   # Part 7: record as failed, not picked
                    picked_ids.append(cid)   # also mark done so loop advances
                continue   # retry pick

            if display and display.should_quit():
                break

            # ── PLACE ────────────────────────────────────────────────────
            print(f"[MAIN] Placing → DROP_ZONE {drop}")
            ctrl.place_point_world(drop)

            # ── Check cube landed at drop zone ───────────────────────────
            if not cube_is_placed(cid, drop):
                print(f"[MAIN] ↺ Cube missed drop zone — retrying"
                      f" (attempt {attempt}/{MAX_RETRIES})")
                if attempt == MAX_RETRIES:
                    print(f"[MAIN] ✗ Max retries reached for cube_id={cid}."
                          f" Moving to next cube.")
                    failed_ids.append(cid)   # Part 7: record as failed
                    picked_ids.append(cid)
                continue   # retry pick + place

            # ── Success ──────────────────────────────────────────────────
            picked_ids.append(cid)
            print(f"[MAIN] ✔ Object {idx+1} placed successfully"
                  f" (attempt {attempt}/{MAX_RETRIES}).")
            break   # move to next cube


    # ── Final summary ────────────────────────────────────────────────────
    successful = [c for c in picked_ids if c not in failed_ids]
    print(f"\n[MAIN] ✅  Task complete.")
    print(f"[MAIN]   Placed successfully : {successful} ({len(successful)} cubes)")
    if failed_ids:
        print(f"[MAIN]   ✗ Failed cubes     : {failed_ids} ({len(failed_ids)} cubes)")

    # ── 6. Keep windows open ─────────────────────────────────────────────
    if display:
        print("[MAIN] Displaying live cameras – press 'q' to exit.")
        while not display.should_quit():
            time.sleep(0.1)
        display.stop()


if __name__ == "__main__":
    import traceback
    try:
        main()
    except Exception as e:
        print(f"\n[FATAL] Simulation crashed: {e}")
        traceback.print_exc()