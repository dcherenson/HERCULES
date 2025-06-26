#!/usr/bin/env python3

import setup_path
import cosysairsim as airsim     


client = airsim.VehicleClient()
client.confirmConnection()

# Give every mesh whose name starts with "Crowd_" the ID 200
client.simSetSegmentationObjectID("BP_CrowdCharacter*", 200, True)
