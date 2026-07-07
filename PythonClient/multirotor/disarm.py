import setup_path
import hercules_cosysairsim as airsim

client = airsim.MultirotorClient()
client.armDisarm(False)
