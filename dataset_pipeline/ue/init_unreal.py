"""
init_unreal.py -- auto-run by the UE editor at startup (PythonScriptPlugin).

Registers a lightweight file-watcher that lets external scripts (e.g. the
HERCULES dataset pipeline) request an actor Label->Name CSV dump without
pasting code into the Python console.

Protocol (files live in the project's Saved/ dir):
  request:  Saved/label_dump_request.json   {"out_csv": "/abs/path/out.csv"}
  result:   Saved/label_dump_result.json    {"ok": true/false, "count": N,
                                             "out_csv": "...", "error": "..."}
The request file is deleted after processing.
"""

import json
import os
import time
import traceback

import unreal

_POLL_INTERVAL_S = 1.0
_state = {"last_check": 0.0, "handle": None}

_SAVED_DIR = unreal.Paths.convert_relative_path_to_full(unreal.Paths.project_saved_dir())
_REQUEST_PATH = os.path.join(_SAVED_DIR, "label_dump_request.json")
_RESULT_PATH = os.path.join(_SAVED_DIR, "label_dump_result.json")


def _get_actors():
    """Return actors from the PIE world if playing, else the editor world."""
    game_world = unreal.EditorLevelLibrary.get_game_world()
    if game_world is not None:
        return unreal.GameplayStatics.get_all_actors_of_class(game_world, unreal.Actor)
    return unreal.EditorLevelLibrary.get_all_level_actors()


def _dump_labels(out_csv):
    import csv
    actors = _get_actors()
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Label", "Name"])
        for a in actors:
            writer.writerow([a.get_actor_label(), a.get_name()])
    return len(actors)


def _process_request():
    try:
        with open(_REQUEST_PATH, "r") as f:
            request = json.load(f)
        out_csv = request["out_csv"]
        count = _dump_labels(out_csv)
        result = {"ok": True, "count": count, "out_csv": out_csv}
        unreal.log("label_dump: wrote {} actors to {}".format(count, out_csv))
    except Exception as e:
        result = {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
        unreal.log_error("label_dump failed: {}".format(e))
    finally:
        try:
            os.remove(_REQUEST_PATH)
        except OSError:
            pass
    with open(_RESULT_PATH, "w") as f:
        json.dump(result, f)


def _tick(_delta_seconds):
    now = time.monotonic()
    if now - _state["last_check"] < _POLL_INTERVAL_S:
        return
    _state["last_check"] = now
    if os.path.isfile(_REQUEST_PATH):
        _process_request()


def register():
    if _state["handle"] is None:
        _state["handle"] = unreal.register_slate_post_tick_callback(_tick)
        unreal.log("label_dump watcher registered (watching {})".format(_REQUEST_PATH))


register()
