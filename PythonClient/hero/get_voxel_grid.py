import setup_path
import cosysairsim as airsim
import os

c = airsim.VehicleClient()
center = airsim.Vector3r(0, 0, 0)
output_path = os.path.join(os.getcwd(), "AusLandscape_map_test1.binvox")
c.simCreateVoxelGrid(center, 100, 100, 100, 0.5, output_path) #works well; tried 300 but is sparse and breaks in viewvox executable