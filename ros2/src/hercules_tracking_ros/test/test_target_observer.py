import pathlib


def test_observer_uses_executable_python_module_and_all_existing_cameras():
    source = pathlib.Path(__file__).parents[4] / "PythonClient" / "distributed_mission" / "modules" / "target_observation.py"
    text = source.read_text()
    assert "simGetDetections" in text
    assert "DepthPerspective" in text
    wrapper = pathlib.Path(__file__).parents[1] / "scripts" / "target_observer_node.py"
    wrapper_text = wrapper.read_text()
    for agent in ("Drone1", "Drone2", "SimpleFlight", "Drone4", "Drone5", "Husky1", "Husky2", "Husky3"):
        assert agent in wrapper_text or "AGENTS" in wrapper_text
    assert '"target_bottom"' in wrapper_text
    assert '"front_center"' in wrapper_text
    assert "MissionTimeMapper" in wrapper_text
