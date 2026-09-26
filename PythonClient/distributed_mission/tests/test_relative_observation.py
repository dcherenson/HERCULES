import numpy as np

from modules.relative_observation import RelativeObservation, RelativeObservationWorker


class _FakeClient:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.__class__.instances.append(self)

    def close(self):
        self.closed = True


class _FakeAirSim:
    MultirotorClient = _FakeClient


def test_relative_worker_reuses_one_client_across_camera_round_robin(monkeypatch):
    _FakeClient.instances = []
    worker = RelativeObservationWorker(
        _FakeAirSim,
        41451,
        {"Drone1": ("localization_front", "target_bottom")},
        ["Drone1", "Drone2"],
        rate_hz=1000.0,
    )
    captures = []

    def fake_capture(client, observer, camera):
        captures.append((client, observer, camera))
        if len(captures) == 4:
            worker._stop.set()
        return RelativeObservation(
            observer_id=observer,
            observed_id="",
            relative_position=np.zeros(2),
            covariance=np.eye(2),
            range=0.0,
            bearing=0.0,
            timestamp=0.0,
            capture_id=f"capture_{len(captures)}",
            sensor=camera,
            valid=False,
        )

    monkeypatch.setattr(worker, "_capture", fake_capture)
    worker._run()

    assert len(_FakeClient.instances) == 1
    assert all(client is _FakeClient.instances[0] for client, _, _ in captures)
    assert [(observer, camera) for _, observer, camera in captures] == [
        ("Drone1", "localization_front"),
        ("Drone1", "target_bottom"),
        ("Drone1", "localization_front"),
        ("Drone1", "target_bottom"),
    ]
    assert worker.capture_count == 4
    assert worker.error_count == 0
    assert _FakeClient.instances[0].closed
