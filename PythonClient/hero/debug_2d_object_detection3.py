import setup_path                  # keep if you're using the local repo copy
import cosysairsim as airsim
import numpy as np
import csv
import cv2

# ---- user params ----
PORT = 41451
CAMERA_NAME = "front_center"
VEHICLE_NAME = ""  # empty for default / single-vehicle setups
CSV_FILENAME = "instance_segmentation_colormap.csv"
# If the image is upside down, set to True; otherwise leave False.
FLIP_VERTICAL = False

# target mesh to highlight
TARGET_MESH = "BP_SplineHuman_Type10_C_UAID_E08F4CF5208A437A02_1596611129"
# ---------------------

# 1. connect
client = airsim.MultirotorClient(port=PORT)
client.confirmConnection()

# 2. get the current object list and segmentation colormap, then dump to CSV
objects = client.simListInstanceSegmentationObjects()  # mesh names
color_map = client.simGetSegmentationColorMap()        # Nx3 array of RGB colors

with open(CSV_FILENAME, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["ObjectName", "R", "G", "B"])
    for idx, name in enumerate(objects):
        col = color_map[idx]
        r, g, b = int(col[0]), int(col[1]), int(col[2])
        writer.writerow([name, r, g, b])

# 3. reload CSV to build both name->color and color->name maps
name_to_color = {}
color_to_name = {}
with open(CSV_FILENAME, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        name = row["ObjectName"]
        r, g, b = int(row["R"]), int(row["G"]), int(row["B"])
        name_to_color[name] = (r, g, b)
        color_to_name[(r, g, b)] = name

# 4. get the uncompressed instance segmentation image
resp = client.simGetImages(
    [airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Segmentation, False, False)],
    vehicle_name=VEHICLE_NAME,
)[0]

if resp.width == 0 or resp.height == 0:
    print("Empty segmentation image.")
    exit(1)

# 5. convert to numpy RGB (AirSim returns RGB order)
img1d = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
img_rgb = img1d.reshape(resp.height, resp.width, 3)

# apply vertical flip only if needed
if FLIP_VERTICAL:
    img_rgb = np.flipud(img_rgb)

# 6. build mask for target mesh and get its bounding box
target_color = name_to_color.get(TARGET_MESH)
if target_color is None:
    print(f"Target mesh '{TARGET_MESH}' not found in CSV.")
else:
    # create binary mask where pixels equal the target RGB color (exact match)
    color_arr = np.array(target_color, dtype=np.uint8)
    mask = np.all(img_rgb == color_arr, axis=2).astype(np.uint8) * 255  # single-channel 0/255 mask

    # find contours on the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"Target mesh '{TARGET_MESH}' has no visible pixels in the segmentation image.")
    else:
        # combine all contour points to get overall bounding box
        all_pts = np.vstack([c.reshape(-1, 2) for c in contours])
        x_min = int(all_pts[:, 0].min())
        y_min = int(all_pts[:, 1].min())
        x_max = int(all_pts[:, 0].max())
        y_max = int(all_pts[:, 1].max())

        print(f"Target mesh '{TARGET_MESH}' bounding box (x_min, y_min, x_max, y_max):",
              (x_min, y_min, x_max, y_max))

        # prepare display image (BGR for OpenCV)
        display = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

        # draw bounding box (white, thickness 2)
        cv2.rectangle(display, (x_min, y_min), (x_max, y_max), (255, 255, 255), 2)

        # draw mesh name (truncated if too long)
        max_len = 30
        disp_name = TARGET_MESH if len(TARGET_MESH) <= max_len else TARGET_MESH[:27] + "..."
        cv2.putText(display, disp_name, (x_min, max(0, y_min - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, lineType=cv2.LINE_AA)

        # 7. show segmentation image with bounding box
        window_name = "Segmentation with Target Mesh Box"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, resp.width, resp.height)
        cv2.imshow(window_name, display)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# 8. fallback: list all meshes seen by unique colors (optional)
unique_colors = np.unique(img_rgb.reshape(-1, 3), axis=0)
found_meshes = set()
for c in unique_colors:
    color_tuple = (int(c[0]), int(c[1]), int(c[2]))
    if color_tuple == (0, 0, 0):
        continue
    name = color_to_name.get(color_tuple)
    if name:
        found_meshes.add(name)
    else:
        print(f"Found unknown segmentation color in image: {color_tuple}")

if found_meshes:
    print("Meshes present in current camera segmentation view:")
    for nm in sorted(found_meshes):
        print("  ", nm)
else:
    print("No known meshes found in segmentation image (only background or unmatched colors).")
