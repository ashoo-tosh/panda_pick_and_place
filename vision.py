"""
vision.py
---------
Computer-vision utilities for the pick-and-place pipeline.

Two detection strategies are provided:

1. detect_objects_seg()  [PRIMARY – stable, never flickers]
   Uses PyBullet's built-in segmentation mask (img[4]) to find object
   pixels exactly.  No HSV thresholding, no bilateral filter – the mask
   is a ground-truth integer label image, one value per body/link.
   This is the correct approach for a simulated environment.

2. detect_objects_rgb()  [FALLBACK – colour-based, for reference]
   HSV thresholding on the RGB image, useful if you later move to a
   real camera where segmentation masks are unavailable.

pixel_to_world()
   Full 3-D back-projection:
       P_world = inv(V) · inv(P) · P_clip
   where P_clip comes from pixel (u,v) and re-encoded depth-buffer value.
"""

import numpy as np
import cv2


class Vision:
    """3-D perception: segmentation-based detection + pixel unprojection."""

    MIN_CONTOUR_AREA = 600   # px²  – real cubes ≈1000-1300px², arm leak ≈200px²

    def __init__(self):
        pass

    # ------------------------------------------------------------------ #
    # 3-D Back-Projection                                                  #
    # ------------------------------------------------------------------ #

    def pixel_to_world(
        self,
        u: float,
        v: float,
        depth_metric: float,
        view_matrix,
        projection_matrix,
        width: int,
        height: int,
    ) -> np.ndarray:
        """
        Back-project pixel (u, v) + metric depth → 3-D world coordinates.

        Maths
        -----
            x_ndc =  2*u/W - 1
            y_ndc =  1 - 2*v/H          (flip Y: image top-down vs OpenGL bottom-up)
            z_ndc =  2*depth_buf - 1    (re-encode metric depth → NDC)

            P_eye   = inv(P_proj) · [x_ndc, y_ndc, z_ndc, 1]ᵀ
            P_world = inv(P_view) · P_eye
        """
        view = np.array(view_matrix,       dtype=np.float64).reshape(4, 4, order="F")
        proj = np.array(projection_matrix, dtype=np.float64).reshape(4, 4, order="F")

        view_inv = np.linalg.inv(view)
        proj_inv = np.linalg.inv(proj)

        x_ndc =  (2.0 * u) / width  - 1.0
        y_ndc =  1.0 - (2.0 * v) / height

        near, far = 0.01, 3.0
        depth_buf = (far - far * near / depth_metric) / (far - near)
        z_ndc     = 2.0 * depth_buf - 1.0

        clip  = np.array([x_ndc, y_ndc, z_ndc, 1.0])
        eye   = proj_inv @ clip
        eye  /= eye[3]

        world  = view_inv @ eye
        world /= world[3]

        return world[:3]

    # ------------------------------------------------------------------ #
    # Primary: Segmentation-Mask Detection                                 #
    # ------------------------------------------------------------------ #

    def detect_objects_seg(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        seg: np.ndarray,
        view_matrix,
        projection_matrix,
        cube_ids: list,
    ) -> list:
        """
        Detect cube centroids using PyBullet's segmentation mask.

        The segmentation mask stores (body_id | link_idx<<24) per pixel.
        We isolate pixels per cube body ID, find the 2-D centroid,
        back-project → 3-D world position.

        Returns
        -------
        List of dicts:
            { "cube_id", "world_pos" (np.ndarray 3,), "pixel" (cx,cy), "area" }
        """
        H, W    = depth.shape
        results = []
        body_mask = seg & 0x00FFFFFF   # lower 24 bits = body id

        for cube_id in cube_ids:

            cube_mask = (body_mask == cube_id).astype(np.uint8) * 255

            if cube_mask.sum() == 0:
                print(f"[VIS] cube_id={cube_id}  → 0 pixels in seg mask (hidden?)")
                continue

            contours, _ = cv2.findContours(
                cube_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area < self.MIN_CONTOUR_AREA:
                    continue

                M = cv2.moments(cnt)
                if M["m00"] == 0:
                    continue

                cx = int(np.clip(int(M["m10"] / M["m00"]), 0, W - 1))
                cy = int(np.clip(int(M["m01"] / M["m00"]), 0, H - 1))

                d = float(depth[cy, cx])
                if d < 0.01 or d > 2.9:
                    print(f"[VIS] cube_id={cube_id}  pixel=({cx},{cy})"
                          f"  depth={d:.4f} m → out of range, skip")
                    continue

                world = self.pixel_to_world(
                    cx, cy, d, view_matrix, projection_matrix, W, H
                )

                # Reject detections too close to robot base (x < 0.38m)
                # — arm links sometimes leak into the segmentation mask
                # and appear near x=0.3 which is the robot base position.
                if world[0] < 0.38:
                    print(f"[VIS] REJECTED  cube_id={cube_id}"
                          f"  x={world[0]:+.4f} < 0.38 (likely arm, not cube)")
                    continue

                print(
                    f"[VIS] DETECTED  cube_id={cube_id}"
                    f"  pixel=({cx:3d},{cy:3d})  depth={d:.4f} m"
                    f"  world=({world[0]:+.4f}, {world[1]:+.4f}, {world[2]:+.4f})"
                    f"  seg_area={area:.0f} px²"
                )

                results.append({
                    "cube_id"   : cube_id,
                    "world_pos" : world,
                    "pixel"     : (cx, cy),
                    "area"      : area,
                })

        return results

    # ------------------------------------------------------------------ #
    # Visualisation helpers                                                #
    # ------------------------------------------------------------------ #

    def make_seg_visual(self, seg: np.ndarray, cube_ids: list) -> np.ndarray:
        """
        Render the segmentation mask as a colour image for display.
        Each body ID gets a stable hue; cubes are shown bright/saturated.
        """
        H, W      = seg.shape
        vis       = np.zeros((H, W, 3), dtype=np.uint8)
        body_mask = seg & 0x00FFFFFF

        for uid in np.unique(body_mask):
            if uid < 0:
                continue
            hue = int((uid * 37 + 13) % 180)
            sat = 255 if uid in cube_ids else 100
            val = 220 if uid in cube_ids else 80
            colour_bgr = cv2.cvtColor(
                np.uint8([[[hue, sat, val]]]), cv2.COLOR_HSV2BGR
            )[0, 0]
            vis[body_mask == uid] = colour_bgr

        # White outline around each cube blob
        for cube_id in cube_ids:
            m = (body_mask == cube_id).astype(np.uint8) * 255
            ctrs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, ctrs, -1, (255, 255, 255), 1)

        return vis

    def make_depth_visual(self, depth: np.ndarray) -> np.ndarray:
        """
        Jet-colourised depth clipped to [0.60, 1.20] m to highlight table objects.
        """
        d_clip = np.clip(depth, 0.60, 1.20)
        d_norm = ((d_clip - 0.60) / 0.60 * 255).astype(np.uint8)
        return cv2.applyColorMap(d_norm, cv2.COLORMAP_JET)

    def annotate_detections(self, bgr: np.ndarray, detections: list) -> np.ndarray:
        """Draw green circles + world-coord labels on a BGR overhead image."""
        out = bgr.copy()
        for det in detections:
            cx, cy = det["pixel"]
            wp     = det["world_pos"]
            label  = f"id={det['cube_id']} ({wp[0]:.3f},{wp[1]:.3f},{wp[2]:.3f})"
            cv2.circle(out, (cx, cy), 12, (0, 255, 0), 2)
            cv2.putText(out, label, (cx + 14, cy + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 0), 1)
        return out