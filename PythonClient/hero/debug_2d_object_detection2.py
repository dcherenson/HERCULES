#!/usr/bin/env python3
"""
corrected_instance_segmentation_viewer.py

Fixes:
  * Uses the proper camera name ("front_center") to obtain the configured 1920x1080 image. 
  * Auto-detects whether to flip segmentation image (avoids upside-down display). 
  * Samples the true instance-level color of BP_SplineHuman_Type10 via projection; fallback to palette or user click.
  * Draws a single bounding box around the target object using its actual color.
  * Forces OpenCV windows to 1920x1080.
  * Warns if instance segmentation likely isn't enabled (few unique colors vs semantic-only).
  * Shows mapping from color to object and prints out the target object's color explicitly.

Sources for behaviors:
  - AirSim image API, flip semantics, camera naming & segmentation retrieval. 
  - Palette mismatch / instance-vs-semantic segmentation color differences.
  - Regex/object naming caveats in Cosys-AirSim.
"""

import sys
import time
import traceback
from collections import defaultdict

import numpy as np
import requests

# try to make cosysairsim importable via project helper
try:
    import setup_path  # noqa: F401
except ImportError:
    pass

# prefer cosysairsim client if present
try:
    import cosysairsim as airsim
except ImportError:
    import airsim

# optional display
try:
    import cv2
except ImportError:
    cv2 = None

# --------- Configuration (no CLI) ----------
AIRSIM_IP = "127.0.0.1"
AIRSIM_PORT = 41451
CAMERA_NAME = "front_center"  # must match the name in settings.json to get 1920x1080. 
TARGET_REGEX = "BP_SplineHuman_Type10"
TARGET_SEG_ID = 200  # optional override (semantic) for the target
MATCH_THRESHOLD = 10.0
SHOW_WINDOWS = True  # requires OpenCV
NO_ASSIGN = False  # if True, skip attempting to set segmentation ID
VERBOSE = True  # dump all scene objects with their IDs
SEG_RGB_URL = "https://microsoft.github.io/AirSim/seg_rgbs.txt"
WINDOW_DISPLAY_SIZE = (1920, 1080)  # as requested: resize both windows to this
# -------------------------------------------


def fetch_palette(url=SEG_RGB_URL):
    """Fetch AirSim's base segmentation palette (object ID → RGB)."""
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        text = resp.text.strip()
        parts = text.split()
        palette = {}
        i = 0
        while i < len(parts):
            try:
                obj_id = int(parts[i])
                j = i + 1
                bracketed = []
                while j < len(parts):
                    bracketed.append(parts[j])
                    if parts[j].endswith("]"):
                        break
                    j += 1
                color_blob = " ".join(bracketed)
                nums = color_blob.strip("[]").replace(",", " ").split()
                if len(nums) >= 3:
                    r, g, b = int(nums[0]), int(nums[1]), int(nums[2])
                    palette[obj_id] = (r, g, b)
                i = j + 1
            except Exception:
                i += 1
        return palette
    except Exception:
        return {}


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def canonical_seg_id(raw):
    """Normalize possible dict/int return to int or None."""
    if raw is None:
        return None
    if isinstance(raw, dict):
        raw = raw.get("return", raw)
    try:
        return int(raw)
    except Exception:
        return None


def quat_to_rot_matrix(q):
    """Convert AirSim quaternion to rotation matrix (camera orientation)."""
    w = q.w_val
    x = q.x_val
    y = q.y_val
    z = q.z_val
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - z * w)
    r02 = 2 * (x * z + y * w)
    r10 = 2 * (x * y + z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - x * w)
    r20 = 2 * (x * z - y * w)
    r21 = 2 * (y * z + x * w)
    r22 = 1 - 2 * (x * x + y * y)
    return np.array([[r00, r01, r02], [r10, r11, r12], [r20, r21, r22]], dtype=np.float64)


def project_world_point_to_image(point_pose, camera_info, image_width, image_height):
    """
    Project a world point to image pixel coordinates using camera pose & FOV.
    Returns (u,v) or None if behind / invalid.
    """
    pw = np.array(
        [
            point_pose.position.x_val,
            point_pose.position.y_val,
            point_pose.position.z_val,
        ],
        dtype=np.float64,
    )

    cam_pose = camera_info.pose
    pc = np.array(
        [
            cam_pose.position.x_val,
            cam_pose.position.y_val,
            cam_pose.position.z_val,
        ],
        dtype=np.float64,
    )

    R_c2w = quat_to_rot_matrix(cam_pose.orientation)
    # camera coordinates: R^T * (pw - pc)
    p_cam = R_c2w.T @ (pw - pc)

    if p_cam[2] <= 0:
        return None  # behind the camera

    # Get FOV in degrees (horizontal)
    fov_deg = getattr(camera_info, "fov", None)
    if fov_deg is None:
        fov_deg = getattr(camera_info, "fov_degrees", None)
    if fov_deg is None:
        fov_deg = 90.0  # fallback

    # focal length from horizontal FOV
    f = image_width / (2.0 * np.tan(np.deg2rad(fov_deg) / 2.0) + 1e-8)

    u = (p_cam[0] * f / p_cam[2]) + (image_width / 2.0)
    v = (-p_cam[1] * f / p_cam[2]) + (image_height / 2.0)

    ui = int(round(u))
    vi = int(round(v))
    if 0 <= ui < image_width and 0 <= vi < image_height:
        return ui, vi
    # alternative y (in case sign convention differs)
    vi_alt = int(round((p_cam[1] * f / p_cam[2]) + (image_height / 2.0)))
    if 0 <= ui < image_width and 0 <= vi_alt < image_height:
        return ui, vi_alt
    return None


def is_background_color(color):
    """Heuristic: white-ish is background/unassigned in segmentation (semantic fallback)."""
    if color is None:
        return True
    # treat near-white as background
    return all(int(c) >= 250 for c in color)


def make_mask(img, color, tolerance=5):
    """
    Exact or tolerant mask for a given RGB tuple on image.
    tolerance: Euclidean distance in RGB space.
    """
    if color is None:
        return None
    target = np.array(color, dtype=np.int32)
    if tolerance == 0:
        mask = np.all(img == target.astype(np.uint8), axis=2).astype(np.uint8) * 255
    else:
        diff = img.astype(np.int32) - target
        dist2 = np.sum(diff * diff, axis=2)
        mask = (dist2 <= (tolerance * tolerance)).astype(np.uint8) * 255
    return mask


def extract_largest_bbox_from_mask(mask):
    """Return (x,y,w,h) for the largest contour in mask, or None."""
    if cv2 is None or mask is None:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    # choose largest by area
    best = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(best)
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def main():
    try:
        client = airsim.VehicleClient(ip=AIRSIM_IP, port=AIRSIM_PORT)
        client.confirmConnection()
        print(f"Connected to AirSim at {AIRSIM_IP}:{AIRSIM_PORT}")

        if not NO_ASSIGN:
            print(f"Attempting to assign segmentation ID {TARGET_SEG_ID} to '{TARGET_REGEX}' (regex enabled)...")
            try:
                success = client.simSetSegmentationObjectID(TARGET_REGEX, TARGET_SEG_ID, True)
                print(f"simSetSegmentationObjectID returned: {success}")
            except Exception as e:
                print(f"Warning: failed to assign segmentation ID: {e}")
            time.sleep(0.1)

        # Load palette (for semantic fallback)
        palette = {}
        try:
            palette = fetch_palette()
            print(f"Loaded base segmentation palette with {len(palette)} entries from {SEG_RGB_URL}")
        except Exception as e:
            print(f"Warning: failed to fetch palette: {e}")

        # Scene objects
        scene_objects = client.simListSceneObjects(".*") or []
        oid_to_names = defaultdict(list)
        print(f"Found {len(scene_objects)} scene objects; querying their segmentation IDs...")
        for name in scene_objects:
            try:
                raw = client.simGetSegmentationObjectID(name)
                seg_id = canonical_seg_id(raw)
                if seg_id is not None and seg_id >= 0:
                    oid_to_names[seg_id].append(name)
            except Exception:
                continue

        if oid_to_names:
            print(f"Discovered {len(oid_to_names)} distinct segmentation object IDs in scene.")
        else:
            print("Warning: no segmentation object IDs retrieved; check your Cosys-AirSim segmentation / settings.")

        if VERBOSE:
            print("\n=== Scene objects with reported segmentation IDs ===")
            for name in scene_objects:
                try:
                    raw = client.simGetSegmentationObjectID(name)
                    seg_id = canonical_seg_id(raw)
                    if seg_id is None:
                        seg_id = -1
                    print(f"[segID={seg_id:3}] {name}")
                except Exception:
                    continue

        # Find target object(s)
        matched_objects = [o for o in scene_objects if TARGET_REGEX.lower() in o.lower()]
        if matched_objects:
            print(f"\nObjects matching '{TARGET_REGEX}' (case-insensitive substring):")
            for o in matched_objects:
                sid = canonical_seg_id(client.simGetSegmentationObjectID(o))
                print(f"  {o} -> segmentation ID: {sid}")
        else:
            print(f"\nNo object containing '{TARGET_REGEX}' found; check scene objects output.")

        exact_id = canonical_seg_id(client.simGetSegmentationObjectID(TARGET_REGEX))
        if exact_id is not None and exact_id >= 0:
            print(f"\nExact lookup for '{TARGET_REGEX}' returned segmentation ID: {exact_id}")
        else:
            print(f"\nExact lookup for '{TARGET_REGEX}' did not yield a valid segmentation ID.")

        # Fetch segmentation image from named camera
        responses = client.simGetImages(
            [airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Segmentation, False, False)]
        )
        if not responses:
            print("Error: no image response received from camera", CAMERA_NAME)
            return
        seg_resp = responses[0]
        if seg_resp.width == 0 or seg_resp.height == 0:
            print("Error: empty segmentation image.")
            return

        print(f"Segmentation image received: {seg_resp.width}x{seg_resp.height}")
        if (seg_resp.width, seg_resp.height) != WINDOW_DISPLAY_SIZE:
            print(f"Warning: expected 1920x1080 per settings.json; got {seg_resp.width}x{seg_resp.height}. "
                  f"Make sure you're requesting the named camera ('{CAMERA_NAME}') and your settings.json has the correct capture settings. ")

        # Build numpy image
        img1d = np.frombuffer(seg_resp.image_data_uint8, dtype=np.uint8)
        try:
            img_rgb = img1d.reshape(seg_resp.height, seg_resp.width, 3)
        except Exception:
            print("Error: malformed image buffer; cannot reshape.")
            print(f"Expected {seg_resp.height * seg_resp.width * 3}, got {len(img1d)}")
            return

        # We'll decide orientation later; keep raw_img for inspection
        raw_img = img_rgb  # original as received

        # Count unique colors to heuristically check if instance segmentation is active
        flat = raw_img.reshape(-1, 3)
        unique_colors, counts = np.unique(flat, axis=0, return_counts=True)
        print(f"\nFound {len(unique_colors)} unique segmentation colors in image.")
        if len(unique_colors) <= 10:
            print(
                "Warning: very few unique colors detected. This likely means only semantic segmentation is active "
                "(not per-instance). To get unique per-object colors, enable instance-level segmentation / ground truth in your settings.json "
                "(e.g., add appropriate flags such as \"GroundTruth\": true and any Cosys-AirSim-specific InstanceSegmentation options). "
            )

        # Build inverse palette for exact semantic mapping (if useful)
        inv_palette = {tuple(v): k for k, v in palette.items()}

        # Dump color → object mapping for diagnostics
        print("\n=== Color → Object mapping (sorted by frequency) ===")
        order = np.argsort(-counts)
        unique_colors = unique_colors[order]
        counts = counts[order]
        for idx, (color, cnt) in enumerate(zip(unique_colors, counts)):
            color_tuple = tuple(int(x) for x in color.tolist())
            hexc = rgb_to_hex(color_tuple)
            line = f"{idx+1:2d}. Color {color_tuple} ({hexc}), pixels: {cnt}"
            matched_oid = None
            distance = None
            exact = False
            if color_tuple in inv_palette:
                matched_oid = inv_palette[color_tuple]
                exact = True
            elif palette:
                # find nearest semantic fallback
                # simple brute force
                best_id, best_dist = None, float("inf")
                for oid, pal_color in palette.items():
                    pal_arr = np.array(pal_color, dtype=np.int32)
                    dist = np.linalg.norm(np.array(color_tuple, dtype=np.int32) - pal_arr)
                    if dist < best_dist:
                        best_dist = dist
                        best_id = oid
                matched_oid, distance = best_id, best_dist
            if matched_oid is not None:
                if exact:
                    line += f" → exact semantic object ID {matched_oid}"
                else:
                    line += f" → nearest semantic object ID {matched_oid} (distance {distance:.2f})"
                    if distance and distance > MATCH_THRESHOLD:
                        line += " [> threshold]"
                names = oid_to_names.get(matched_oid, [])
                if names:
                    line += f", mesh/actor names: {names}"
                else:
                    line += ", mesh/actor names: <none found>"
            else:
                line += " → no semantic palette match"
            print(line)

        # Determine the object-of-interest's color and bounding box
        interest_color = None
        orientation_flipped = False  # whether we should flip for display
        target_name = matched_objects[0] if matched_objects else None
        print("\n=== BP_SplineHuman_Type10 Summary ===")
        if target_name:
            actual_id = canonical_seg_id(client.simGetSegmentationObjectID(target_name))
            print(f"Target object: '{target_name}' segmentation ID: {actual_id}")
            if actual_id is not None and actual_id >= 0:
                # base semantic palette color (fallback)
                base_color = palette.get(actual_id)
                if base_color:
                    print(f"  → Semantic (base) palette color for ID {actual_id}: {base_color} ({rgb_to_hex(base_color)})")
                else:
                    print(f"  → No base palette entry for semantic ID {actual_id}; likely custom/instance color.")

                # Try to sample true instance-level color via projection
                try:
                    obj_pose = client.simGetObjectPose(target_name)
                    if obj_pose is not None:
                        camera_info = client.simGetCameraInfo(CAMERA_NAME)
                        proj = project_world_point_to_image(obj_pose, camera_info, seg_resp.width, seg_resp.height)
                        if proj is not None:
                            u, v = proj
                            h = seg_resp.height
                            # candidate without flip
                            color_no_flip = tuple(int(x) for x in raw_img[v, u].tolist())
                            # candidate if the image were vertically flipped (simulate): projection would appear at mirrored y
                            color_flipped_candidate = tuple(int(x) for x in raw_img[h - 1 - v, u].tolist())

                            # Decide orientation by checking which sample is non-background
                            if not is_background_color(color_no_flip):
                                interest_color = color_no_flip
                                orientation_flipped = False
                                print(f"  → Sampled instance segmentation color (no flip) at ({u},{v}): {interest_color} ({rgb_to_hex(interest_color)})")
                            elif not is_background_color(color_flipped_candidate):
                                interest_color = color_flipped_candidate
                                orientation_flipped = True
                                print(f"  → Sampled instance segmentation color (with flip) at mirror location: {interest_color} ({rgb_to_hex(interest_color)})")
                            else:
                                print("  → Projection landed on background both with and without flip; falling back to semantic color or user pick.")
                        else:
                            print("  → Projection of object into image failed / object may be out of view.")
                    else:
                        print("  → Could not retrieve object pose.")
                except Exception as e:
                    print(f"  → Exception during projection/color sampling: {e}")

                # Fallback if no instance-level color determined
                if interest_color is None:
                    if base_color:
                        interest_color = base_color
                        print(f"  → Using semantic fallback color: {interest_color} ({rgb_to_hex(interest_color)})")
                    else:
                        print("  → No fallback color available; will require manual selection.")
            else:
                print(f"  → Could not resolve segmentation ID for '{target_name}'.")
        else:
            print("No target object found; cannot summarize.")

        # Prepare display image with orientation
        display_img = np.flipud(raw_img) if orientation_flipped else raw_img

        # If interest_color still missing or suspicious (background), enable interactive picking
        if (interest_color is None) or is_background_color(interest_color):
            if SHOW_WINDOWS and cv2 is not None:
                print("Interactive fallback: please click on the target object in the raw segmentation window to select its color.")
                picked = {"color": None}

                def on_mouse(event, x, y, flags, param):
                    if event == cv2.EVENT_LBUTTONDOWN:
                        picked["color"] = tuple(int(c) for c in display_img[y, x])
                        print(f"Picked color {picked['color']} at ({x},{y})")

                win_raw = "Segmentation (raw) - pick target color"
                seg_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)
                cv2.namedWindow(win_raw, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(win_raw, WINDOW_DISPLAY_SIZE[0], WINDOW_DISPLAY_SIZE[1])
                cv2.setMouseCallback(win_raw, on_mouse)
                cv2.imshow(win_raw, seg_bgr)
                print("Click on the object, then press any key...")
                cv2.waitKey(0)
                cv2.destroyWindow(win_raw)
                if picked["color"] is not None:
                    interest_color = picked["color"]
                    print(f"Using user-selected interest color: {interest_color} ({rgb_to_hex(interest_color)})")
                else:
                    print("No interactive color picked; continuing without overlay.")
            else:
                print("No valid interest color and interactive selection unavailable (OpenCV disabled).")

        # Build mask and bounding box for target only
        overlay = None
        if interest_color is not None:
            # Mask with some tolerance in case of minor compression differences
            mask = make_mask(display_img, interest_color, tolerance=5)
            bbox = extract_largest_bbox_from_mask(mask)
            overlay = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR) if cv2 else None
            if bbox and cv2 is not None:
                x, y, w, h = bbox
                label = target_name if target_name else TARGET_REGEX
                cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 0), 2)
                display_label = label if len(label) <= 30 else label[:27] + "..."
                cv2.putText(
                    overlay,
                    display_label,
                    (x, max(0, y - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    lineType=cv2.LINE_AA,
                )
                print(f"Bounding box for target: x={x}, y={y}, w={w}, h={h}")
            else:
                print("Could not extract a bounding box for the target (mask empty or no contour).")
        else:
            print("No interest color determined; skipping overlay creation.")

        # Display windows: raw and overlay
        if SHOW_WINDOWS and cv2 is not None:
            win_raw = "Segmentation (raw)"
            win_target = "Target Overlay"
            seg_bgr = cv2.cvtColor(display_img, cv2.COLOR_RGB2BGR)

            # Raw window
            cv2.namedWindow(win_raw, cv2.WINDOW_NORMAL)
            cv2.imshow(win_raw, seg_bgr)
            cv2.resizeWindow(win_raw, WINDOW_DISPLAY_SIZE[0], WINDOW_DISPLAY_SIZE[1])

            # Overlay
            if overlay is not None:
                cv2.namedWindow(win_target, cv2.WINDOW_NORMAL)
                cv2.imshow(win_target, overlay)
                cv2.resizeWindow(win_target, WINDOW_DISPLAY_SIZE[0], WINDOW_DISPLAY_SIZE[1])
            else:
                print("Skipping overlay window; nothing to show for target.")

            print("Press any key in any window to exit.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            if cv2 is None:
                print("OpenCV not installed; cannot show image windows.")
            else:
                print("Window display disabled (SHOW_WINDOWS=False).")

    except KeyboardInterrupt:
        print("Interrupted by user.")
    except Exception:
        print("Fatal error:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
