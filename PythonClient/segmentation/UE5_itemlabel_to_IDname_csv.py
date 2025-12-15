import unreal
import csv
import os

world = unreal.EditorLevelLibrary.get_editor_world()
actors = unreal.EditorLevelLibrary.get_all_level_actors()

csv_path = os.path.join(os.path.expanduser("~"), "ue_actor_label_to_name_ausenv_aroundseq.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Label", "Name"])
    for a in actors:
        writer.writerow([a.get_actor_label(), a.get_name()])

print("Saved to:", csv_path)
