#!/usr/bin/env python3
"""Print the vehicle names the RUNNING sim has registered (Hero car server = 41452).

    cd .../Cosys-AirSim/PythonClient && python3 list_vehicles.py

This tells us the pre-placed pawn's actual name -- the thing that decides how we
add a second Husky (see the multi-robot spawn discussion).
"""
import sys
sys.path.insert(0, ".")
import hercules_cosysairsim as airsim

c = airsim.CarClient(ip="127.0.0.1", port=41452, timeout_value=10)
c.confirmConnection()
print("listVehicles():", c.listVehicles())
