"""A duplicate launch dictionary key silently disabled MA-ICP tracking."""
import ast
from pathlib import Path


def test_launch_parameter_dictionaries_have_no_duplicate_literal_keys():
    source_root = Path(__file__).resolve().parents[2]
    paths = [source_root / 'hercules_mission_ros/launch/rural_nominal.launch.py',
             source_root / 'hercules_maicp/launch/case_study.launch.py']
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Dict):
                keys = [key.value for key in node.keys
                        if isinstance(key, ast.Constant) and isinstance(key.value, str)]
                assert len(keys) == len(set(keys)), f'{path}:{node.lineno}: duplicate parameter'
