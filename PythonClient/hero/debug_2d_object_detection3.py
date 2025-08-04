import setup_path                  
import cosysairsim as airsim
import numpy as np

# 1. connect
client = airsim.MultirotorClient(port=41451)  # or airsim.CarClient()
client.confirmConnection()

# 2. get camera FOV (optional, but tells you the camera's field of view)
cam_info = client.simGetCameraInfo("front_center")  # cam_info.fov contains FOV in degrees

# 3. list visible segmentation objects (names) and their poses
names = client.simListInstanceSegmentationObjects()
visible_poses = client.simListInstanceSegmentationPoses(only_visible=True)  # objects currently in view

# 4. get the instance segmentation image (uncompressed for exact colors)
resp = client.simGetImages([
    airsim.ImageRequest("front_center", airsim.ImageType.Segmentation, False, False)
])[0]
img = np.frombuffer(resp.image_data_uint8, dtype=np.uint8).reshape(resp.height, resp.width, 3)
img = np.flipud(img)  # correct orientation

# 5. get the global segmentation colormap (index → RGB)
color_map = client.simGetSegmentationColorMap()

# 6. map each mesh name to its assigned segmentation ID and color
for name in names:
    seg_id = client.simGetSegmentationObjectID(name)  # -1 means hidden/not assigned
    if seg_id >= 0:
        color = color_map[seg_id]  # RGB tuple for that instance
        # name, seg_id, and color are your “label” info

# 7. (alternative) infer which instances are present by unique colors in the image
unique_colors = np.unique(img.reshape(-1, 3), axis=0)
