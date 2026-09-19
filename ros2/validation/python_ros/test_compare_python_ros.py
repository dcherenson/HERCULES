import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from compare_python_ros import (  # noqa: E402
    compare_datasets,
    load_jsonl,
    normalize_raw,
    resample_dataset,
    run_comparison,
)


def _python_row(timestamp, position):
    return {
        "step": int(round((timestamp - 10.0) / 0.1)),
        "timestamp": timestamp,
        "vehicle_types": {"Drone1": "drone"},
        "states": {
            "Drone1": {
                "position": position,
                "velocity": [1.0, 0.0, 0.0],
                "yaw": 0.0,
                "yaw_rate": 0.0,
            }
        },
        "desired_slots": {"Drone1": [position[0], 0.0, -5.0]},
        "slot_errors": {"Drone1": 0.0},
        "commands": {"Drone1": [1.0, 0.0, 0.0]},
        "target_truth": {
            "position": [position[0], 0.0, -1.0],
            "velocity": [1.0, 0.0, 0.0],
            "index": 1,
            "sample_count": 10,
        },
        "cbf": {
            "Drone1": {
                "safe_control": [1.0, 0.0, 0.0],
                "success": True,
                "minimum_barrier": 0.2,
                "solve_time_ms": 2.0,
            }
        },
        "target_tracking": {
            "agents": {
                "Drone1": {
                    "active": True,
                    "measurement": {"valid": True, "visible": True},
                    "estimate": {
                        "position": [position[0], 0.0, -1.0],
                        "velocity": [1.0, 0.0, 0.0],
                        "iterations": 2,
                        "consensus_residual": 0.01,
                    },
                }
            }
        },
    }


def _ros_row(timestamp, position):
    return {
        "step_id": int(round((timestamp - 20.0) / 0.1)),
        "timestamp": timestamp,
        "vehicle_types": {"Drone1": "drone"},
        "states": {
            "Drone1": {
                "position": position,
                "velocity": [1.0, 0.0, 0.0],
                "yaw": 0.0,
            }
        },
        "desired_slots": {"Drone1": [position[0], 0.0, -5.0]},
        "slot_errors": {"Drone1": 0.0},
        "commands": {"Drone1": [1.0, 0.0, 0.0]},
        "target_truth": {
            "position": [position[0], 0.0, -1.0],
            "velocity": [1.0, 0.0, 0.0],
            "index": 1,
            "sample_count": 10,
        },
        "cbf": {
            "schema_version": 1,
            "enabled": False,
            "entries": [
                {
                    "agent_id": "Drone1",
                    "enabled": False,
                    "nominal_control": [1.0, 0.0, 0.0],
                    "safe_control": [1.0, 0.0, 0.0],
                    "success": True,
                    "solver_time_ms": 0.0,
                }
            ],
        },
    }


class ComparePythonRosTest(unittest.TestCase):
    def test_normalizes_both_cbf_shapes_and_interpolates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python_path = root / "python.jsonl"
            ros_path = root / "ros.jsonl"
            python_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (_python_row(10.0, [0.0, 0.0, -5.0]), _python_row(10.2, [2.0, 0.0, -5.0]))
                )
                + "\n",
                encoding="utf-8",
            )
            ros_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (_ros_row(20.0, [0.0, 0.0, -5.0]), _ros_row(20.2, [2.0, 0.0, -5.0]))
                )
                + "\n",
                encoding="utf-8",
            )
            python = normalize_raw(python_path, "python", load_jsonl(python_path))
            ros = normalize_raw(ros_path, "ros", load_jsonl(ros_path))
            grid = [0.0, 0.1, 0.2]
            python = resample_dataset(python, grid)
            ros = resample_dataset(ros, grid)
            comparison = compare_datasets(python, ros)

            self.assertEqual(python["time"], grid)
            self.assertAlmostEqual(python["records"][1]["agents"]["Drone1"]["position"][0], 1.0)
            self.assertTrue(python["records"][0]["agents"]["Drone1"]["cbf_enabled"])
            self.assertFalse(ros["records"][0]["agents"]["Drone1"]["cbf_enabled"])
            self.assertEqual(comparison["common_agents"], ["Drone1"])
            self.assertAlmostEqual(comparison["global"]["position_diff_m"]["max"], 0.0)

    def test_cli_emits_expected_artifacts_without_plot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python_path = root / "python.jsonl"
            ros_path = root / "ros.jsonl"
            python_path.write_text(json.dumps(_python_row(10.0, [0.0, 0.0, -5.0])) + "\n", encoding="utf-8")
            ros_path.write_text(json.dumps(_ros_row(20.0, [0.0, 0.0, -5.0])) + "\n", encoding="utf-8")
            output = root / "comparison"
            args = type(
                "Args",
                (),
                {
                    "python_run": str(python_path),
                    "ros_run": str(ros_path),
                    "output_dir": output,
                    "dt": 0.1,
                    "no_plot": True,
                    "strict": False,
                },
            )()
            run_comparison(args)
            for name in (
                "normalized_python.csv",
                "normalized_ros.csv",
                "normalized_python.json",
                "normalized_ros.json",
                "comparison.csv",
                "comparison.json",
                "comparison.md",
            ):
                self.assertTrue((output / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
