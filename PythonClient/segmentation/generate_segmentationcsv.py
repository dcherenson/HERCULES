#!/usr/bin/env python3

import os
import csv
import setup_path
import cosysairsim as airsim

def main():
    # Connect to Cosys-AirSim Multirotor client
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # List all instance-segmentation object names
    objs = client.simListInstanceSegmentationObjects()

    # Ensure the segmentation folder exists
    seg_dir = "/home/sgarimella34/multi-robot-coordination/Cosys-AirSim/PythonClient/segmentation"
    os.makedirs(seg_dir, exist_ok=True)

    # Path where the second script will look for segmentation.csv
    seg_file = os.path.join(seg_dir, "segmentation.csv")

    # Write out one prefix per line
    prefixes = {name.split("_")[0] for name in objs}
    with open(seg_file, "w", newline="") as f:
        writer = csv.writer(f)
        for p in sorted(prefixes):
            writer.writerow([p])

    print(f"Wrote segmentation.csv with {len(prefixes)} prefixes to:\n  {seg_file}")

if __name__ == "__main__":
    main()
