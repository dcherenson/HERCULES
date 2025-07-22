#!/usr/bin/env python3

import setup_path
import cosysairsim as airsim
import os
import math
import msgpackrpc.error

def generate_voxel_patch(client, center, patch_size, resolution, output_file):
    """
    Create a 100*100*100 m cube centered at `center`.
    """
    px = int(patch_size)
    res = float(resolution)

    # Primary signature: (center, x, y, z, resolution, filename)
    try:
        client.simCreateVoxelGrid(center, px, px, px, res, output_file)
        print(f"Saved: {output_file}")
        return
    except msgpackrpc.error.RPCError:
        # Fallback: (center, x, y, z, filename, resolution)
        try:
            client.simCreateVoxelGrid(center, px, px, px, output_file, res)
            print(f"Saved (alt-order): {output_file}")
            return
        except Exception as e:
            print(f"Failed to write {output_file}: {e}")

def main():
    # ─── User parameters ─────────────────────────────────────────────────
    world_size   = 700.0               # X/Y world span in meters
    patch_size   = 100.0               # each cube is 100 m on a side
    stack_height = 600.0               # total vertical height to cover in meters
    resolution   = 0.5                 # voxel size in meters
    world_center = (0.0, 0.0, 0.0)     # ground-level origin (X, Y, Z)
    output_dir   = "/home/sgarimella34/multi-robot-coordination/data_binvox_octomap/customcity_0p5mcubed"
    port         = 41452               # AirSim RPC port
    # ─────────────────────────────────────────────────────────────────────

    os.makedirs(output_dir, exist_ok=True)
    client = airsim.VehicleClient(port=port)

    num_xy = int(world_size / patch_size)
    num_z  = int(math.ceil(stack_height / patch_size))

    min_x = -world_size / 2
    min_y = -world_size / 2
    ground_z = world_center[2]

    for ix in range(num_xy):
        for iy in range(num_xy):
            cx = min_x + ix * patch_size + patch_size / 2
            cy = min_y + iy * patch_size + patch_size / 2

            for iz in range(num_z):
                # center Z for layer iz: start first layer at ground_z
                # then move down (negative) by patch_size each layer
                cz = ground_z - iz * patch_size
                center = airsim.Vector3r(cx, cy, cz)

                fname = f"patch_{cx:.1f}_{cy:.1f}_layer{iz}.binvox"
                outp  = os.path.join(output_dir, fname)
                print(f"Generating {fname} at center ({cx:.1f}, {cy:.1f}, {cz:.1f})")

                generate_voxel_patch(client,
                                     center,
                                     patch_size,
                                     resolution,
                                     outp)

if __name__ == "__main__":
    main()
