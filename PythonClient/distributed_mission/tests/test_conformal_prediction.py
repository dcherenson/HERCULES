from modules.conformal_prediction import ConformalPredictionModule


def test_placeholder_robustness_margin_is_zero():
    module = ConformalPredictionModule("Drone1")
    assert module.compute_robustness_margin({}, {}) == 0.0
