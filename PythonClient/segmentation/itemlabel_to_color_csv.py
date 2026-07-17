#!/usr/bin/env python3
"""
Merge UE5 actor labels with AirSim segmentation colors.

Inputs:
  1) Labels CSV from UE Python:
        Label,Name
        TreeCluster_North,StaticMeshActor_214
        ...

  2) AirSim segmentation CSV:
        ObjectName,SegmentationID,R,G,B
        StaticMeshActor_214,5,255,0,0
        ...

Output:
  CSV with:
        Label,ObjectName,SegmentationID,R,G,B

Usage:
  python3 itemlabel_to_color_csv.py \
      --labels_csv ue_actor_label_to_name.csv \
      --colors_csv airsim_segmentation_colormap_list_2025_11_23_23_37_23.csv \
      --out_csv label_color_map.csv
"""

import csv
import argparse


def load_labels(labels_csv):
    """Return dict: internal_name -> label."""
    name_to_label = {}
    with open(labels_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        # Expect columns: Label,Name
        if "Label" not in reader.fieldnames or "Name" not in reader.fieldnames:
            raise ValueError(
                f"{labels_csv} must have columns 'Label' and 'Name', got {reader.fieldnames}"
            )
        for row in reader:
            name = row["Name"]
            label = row["Label"]
            # Last one wins if duplicates; usually there is one per actor
            name_to_label[name] = label
    return name_to_label


def load_colors(colors_csv):
    """Return dict: internal_name -> (seg_id, r, g, b)."""
    name_to_color = {}
    with open(colors_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        # Expect columns: ObjectName,SegmentationID,R,G,B
        required = {"ObjectName", "SegmentationID", "R", "G", "B"}
        if not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{colors_csv} must have columns {required}, got {reader.fieldnames}"
            )
        for row in reader:
            name = row["ObjectName"]
            seg_id = row["SegmentationID"]
            r = row["R"]
            g = row["G"]
            b = row["B"]
            name_to_color[name] = (seg_id, r, g, b)
    return name_to_color


def merge_and_write(labels_csv, colors_csv, out_csv, include_unlabeled=False):
    name_to_label = load_labels(labels_csv)
    name_to_color = load_colors(colors_csv)

    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Label", "ObjectName", "SegmentationID", "R", "G", "B"])

        for name, (seg_id, r, g, b) in name_to_color.items():
            label = name_to_label.get(name)
            if label is None:
                if not include_unlabeled:
                    # Skip objects that do not have a label mapping
                    continue
                # Fallback: use object name as label
                label = name
            writer.writerow([label, name, seg_id, r, g, b])

    print(f"Wrote merged CSV to: {out_csv}")
    print(f"Total AirSim objects: {len(name_to_color)}")
    print(f"Total UE label entries: {len(name_to_label)}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge UE actor labels with AirSim segmentation colors."
    )
    parser.add_argument(
        "--labels_csv",
        required=True,
        help="Path to ue_actor_label_to_name.csv (Label,Name)",
    )
    parser.add_argument(
        "--colors_csv",
        required=True,
        help="Path to airsim_segmentation_colormap_list_*.csv (ObjectName,SegmentationID,R,G,B)",
    )
    parser.add_argument(
        "--out_csv",
        default="label_color_map.csv",
        help="Output CSV path (default: label_color_map.csv)",
    )
    parser.add_argument(
        "--include_unlabeled",
        action="store_true",
        help="If set, include objects without label mapping (label = ObjectName)",
    )

    args = parser.parse_args()
    merge_and_write(args.labels_csv, args.colors_csv, args.out_csv, args.include_unlabeled)


if __name__ == "__main__":
    main()
