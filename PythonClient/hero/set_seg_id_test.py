import setup_path               # ensures AirSim Python client is on PYTHONPATH
import cosysairsim as airsim
import numpy as np
import cv2
import re

# 0. ---- USER PARAMETERS ----
SEG_ID    = 200                 # the segmentation ID to assign
REGEX     = ".*BP_SplineHuman.*" # regex to match all SplineHuman meshes
CAM_NAME  = "front_center"      # use the string name of your camera
FLIP_VERT = False               # True if your seg image comes back upside-down
# ----------------------------

def seg_id_to_rgb(seg_id):
    # AirSim maps seg_id→(R,G,B) as (seg_id,0,0)
    return (seg_id, 0, 0)

# 1. Connect
client = airsim.MultirotorClient()
client.confirmConnection()
print(f"Connected!  Client Ver:{client.getClientVersion()}  Server Ver:{client.getServerVersion()}")

# 2. List all scene objects
objs = client.simListSceneObjects()
print("\n=== All Scene Objects ===")
for name in objs:
    print(" ", name)

# 3. Filter to matching BP_SplineHuman names
pattern = re.compile(REGEX)
matches = [n for n in objs if pattern.match(n)]
print(f"\nFound {len(matches)} objects matching '{REGEX}':")
for n in matches:
    print(" ", n)

# 4. Assign SEG_ID to each human
print(f"\nSetting segmentation ID = {SEG_ID} on each human:")
for name in matches:
    ok = client.simSetSegmentationObjectID(name, SEG_ID, False)
    color = seg_id_to_rgb(SEG_ID)
    status = "OK" if ok else "FAIL"
    print(f" {status}: '{name}' → seg_id={SEG_ID}, color=RGB{color}")

# 5. Grab a full-HD segmentation image (1920×1080)
resp = client.simGetImages([
    airsim.ImageRequest(CAM_NAME, airsim.ImageType.Segmentation, False, False)
])[0]
if resp.width == 0 or resp.height == 0:
    raise RuntimeError("Empty segmentation image.")

print(f"\nSeg image resolution: {resp.width}×{resp.height}")

# 6. Decode to RGB numpy array
img1d   = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
img_rgb = img1d.reshape(resp.height, resp.width, 3)
if FLIP_VERT:
    img_rgb = np.flipud(img_rgb)

# 7. Convert to BGR for OpenCV and display
img_bgr     = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
window_name = f"Segmentation (ID={SEG_ID})"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, resp.width // 2, resp.height // 2)
cv2.imshow(window_name, img_bgr)
print("\nPress any key in the image window to exit.")
cv2.waitKey(0)
cv2.destroyAllWindows()
