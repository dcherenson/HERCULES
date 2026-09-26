"""Small, ROS-independent configuration objects for the MAICP core.

The values in :func:`paper_config` mirror the defaults used by the paper.  The
configuration intentionally contains numerical algorithm parameters only; ROS
topics, node names, and mission executors belong to the integration package.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import ceil, isfinite, log, sqrt
from typing import Mapping


@dataclass(frozen=True)
class ClassConfig:
    """Configuration for one score class.

    ``n`` is the number of class members in one mission and ``ell`` is the
    permitted number of robot-level score violations.  The paper uses the
    resulting rank ``n - ell`` as the mission-level score.

    The alias properties at the bottom keep call sites readable when they use
    the terminology from the paper (``robots_per_mission``, ``violations``,
    and ``kappa``).
    """

    name: str
    n: int = 3
    alpha: float = 0.10
    delta_cal: float = 0.99
    ell: int = 1
    scale: float = 1.0
    weight: float = 1.0
    initial_margin: float = 0.2
    margin_min: float = 0.0
    margin_max: float = 100.0
    configured_kappa: float = 0.6

    @property
    def mission_rank(self) -> int:
        """The required order-statistic rank, ``K_c = n - ell``."""

        return self.n - self.ell

    @property
    def robots_per_mission(self) -> int:
        return self.n

    @property
    def violations(self) -> int:
        return self.ell

    @property
    def kappa(self) -> float:
        """Configured sensitivity used by the numerical baseline.

        A configured value is deliberately not a formal sensitivity
        certificate.  ``calibrate_round`` reports that distinction through its
        ``certified`` field.
        """

        return self.configured_kappa

    def adjusted_alpha(self, missions: int) -> float:
        """Finite-mission correction from the MAICP paper."""

        if missions <= 0:
            raise ValueError("missions must be positive")
        return self.alpha - sqrt(log(1.0 / self.delta_cal) / (2.0 * missions))

    def paper_rank(self, missions: int) -> int:
        """Rank used by adaptive MAICP threshold calibration."""

        adjusted = self.adjusted_alpha(missions)
        if adjusted <= 0.0:
            raise ValueError(
                f"adjusted alpha is non-positive for class {self.name!r} "
                f"with {missions} calibration missions"
            )
        return int(ceil((1.0 - adjusted) * missions))

    def validate(self, *, missions: int | None = None) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("class name must be a non-empty string")
        if not isinstance(self.n, int) or self.n < 1:
            raise ValueError(f"class {self.name!r} must have a positive n")
        if not isinstance(self.ell, int) or not 0 <= self.ell < self.n:
            raise ValueError(f"class {self.name!r} must satisfy 0 <= ell < n")
        if not isfinite(float(self.alpha)) or not 0.0 < self.alpha < 1.0:
            raise ValueError(f"class {self.name!r} alpha must lie in (0, 1)")
        if not isfinite(float(self.delta_cal)) or not 0.0 < self.delta_cal < 1.0:
            raise ValueError(f"class {self.name!r} delta_cal must lie in (0, 1)")
        for field_name, value in (("scale", self.scale), ("weight", self.weight)):
            if not isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(
                    f"class {self.name!r} {field_name} must be finite and positive"
                )
        if not isfinite(float(self.configured_kappa)) or self.configured_kappa < 0.0:
            raise ValueError(
                f"class {self.name!r} configured_kappa must be finite and nonnegative"
            )
        for field_name, value in (
            ("initial_margin", self.initial_margin),
            ("margin_min", self.margin_min),
            ("margin_max", self.margin_max),
        ):
            if not isfinite(float(value)):
                raise ValueError(f"class {self.name!r} {field_name} must be finite")
        if self.margin_min > self.margin_max:
            raise ValueError(f"class {self.name!r} has an inverted margin interval")
        if not self.margin_min <= self.initial_margin <= self.margin_max:
            raise ValueError(f"class {self.name!r} initial margin is outside its interval")
        if missions is not None and self.adjusted_alpha(missions) <= 0.0:
            raise ValueError(
                f"class {self.name!r} has non-positive adjusted alpha for "
                f"{missions} calibration missions"
            )


def _paper_classes() -> dict[str, ClassConfig]:
    return {
        "uav": ClassConfig(name="uav"),
        "ugv": ClassConfig(name="ugv"),
    }


@dataclass(frozen=True)
class MAICPConfig:
    """Numerical MAICP parameters.

    ``classes`` is keyed by stable string class names.  A caller may replace
    it with another mapping (for example, a single-class unit test), and every
    numerical helper accepts the same object without importing ROS types.
    """

    outer_replicates: int = 3
    rounds: int = 5
    calibration_missions: int = 200
    deployment_missions: int = 200
    K_S: int = 100
    K_q: int = 100
    admm_steps: int = 50
    simulation_steps: int = 100
    mission_duration_sec: float = 10.0
    alpha: float = 0.10
    delta_cal: float = 0.99
    pinball_gamma_0: float = 0.05
    pinball_exponent: float = 0.75
    proximal_weight: float = 0.25
    classes: Mapping[str, ClassConfig] = field(default_factory=_paper_classes)

    # Readable aliases used by some numerical callers and by the paper's
    # notation.  They are properties rather than duplicate mutable fields.
    @property
    def gamma_0(self) -> float:
        return self.pinball_gamma_0

    @property
    def exponent(self) -> float:
        return self.pinball_exponent

    @property
    def lambda_prox(self) -> float:
        return self.proximal_weight

    @property
    def initial_margins(self) -> dict[str, float]:
        return {name: cls.initial_margin for name, cls in self.classes.items()}

    @property
    def kappas(self) -> dict[str, float]:
        return {name: cls.configured_kappa for name, cls in self.classes.items()}

    @property
    def kappa(self) -> float:
        """Common configured kappa for the paper's two-class defaults."""

        values = set(self.kappas.values())
        if len(values) != 1:
            raise ValueError("kappa is class-dependent for this configuration")
        return float(next(iter(values)))

    @property
    def mission_pinball_steps(self) -> int:
        return self.K_S

    @property
    def threshold_pinball_steps(self) -> int:
        return self.K_q

    @property
    def K_ADMM(self) -> int:
        return self.admm_steps

    @property
    def tracking_admm_steps(self) -> int:
        return self.admm_steps

    @property
    def mission_duration(self) -> float:
        return self.mission_duration_sec

    @property
    def steps(self) -> int:
        return self.simulation_steps

    @property
    def class_configs(self) -> Mapping[str, ClassConfig]:
        return self.classes

    def class_config(self, name: object) -> ClassConfig:
        """Resolve a class key while accepting enum-like keys by value/name."""

        if name in self.classes:
            return self.classes[name]  # type: ignore[index]
        text = getattr(name, "value", name)
        if text in self.classes:
            return self.classes[text]  # type: ignore[index]
        text = str(text).lower()
        if text in self.classes:
            return self.classes[text]
        for key, value in self.classes.items():
            if str(key).lower() == text or value.name.lower() == text:
                return value
        raise KeyError(f"Unknown class {name!r}")

    def validate(self) -> None:
        for field_name in (
            "outer_replicates",
            "rounds",
            "calibration_missions",
            "deployment_missions",
            "K_S",
            "K_q",
            "admm_steps",
            "simulation_steps",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a nonnegative integer")
        if self.calibration_missions < 1 or self.deployment_missions < 1:
            raise ValueError("calibration and deployment mission counts must be positive")
        if not isfinite(float(self.mission_duration_sec)) or self.mission_duration_sec <= 0.0:
            raise ValueError("mission_duration_sec must be finite and positive")
        if not isfinite(float(self.alpha)) or not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must lie in (0, 1)")
        if not isfinite(float(self.delta_cal)) or not 0.0 < self.delta_cal < 1.0:
            raise ValueError("delta_cal must lie in (0, 1)")
        if not isfinite(float(self.pinball_gamma_0)) or self.pinball_gamma_0 <= 0.0:
            raise ValueError("pinball_gamma_0 must be finite and positive")
        if not isfinite(float(self.pinball_exponent)) or not 0.5 < self.pinball_exponent <= 1.0:
            raise ValueError("pinball_exponent must lie in (0.5, 1]")
        if not isfinite(float(self.proximal_weight)) or self.proximal_weight <= 0.0:
            raise ValueError("proximal_weight must be finite and positive")
        if not self.classes:
            raise ValueError("at least one class is required")
        names: set[str] = set()
        for key, cls in self.classes.items():
            if not isinstance(cls, ClassConfig):
                raise TypeError(f"class {key!r} is not a ClassConfig")
            cls.validate(missions=self.calibration_missions)
            canonical = cls.name.lower()
            if canonical in names:
                raise ValueError(f"duplicate class name {cls.name!r}")
            names.add(canonical)


# Public factory used by both the ROS wrapper and standalone numerical tests.
def paper_config(**overrides: object) -> MAICPConfig:
    """Return validated paper defaults with optional dataclass overrides.

    ``classes`` may be supplied as a complete mapping.  Scalar overrides are
    passed to :class:`MAICPConfig`; this keeps the factory explicit and avoids
    hidden global configuration state.
    """

    if "classes" not in overrides:
        # Keep the scalar top-level knobs and the per-class paper parameters
        # synchronized when an experiment changes alpha or delta_cal.
        alpha = float(overrides.get("alpha", 0.10))
        delta_cal = float(overrides.get("delta_cal", 0.99))
        overrides["classes"] = {
            name: replace(cls, alpha=alpha, delta_cal=delta_cal)
            for name, cls in _paper_classes().items()
        }
    config = MAICPConfig(**overrides)  # type: ignore[arg-type]
    config.validate()
    return config


def default_config(**overrides: object) -> MAICPConfig:
    """Alias for :func:`paper_config`."""

    return paper_config(**overrides)


# A constant is convenient for callers that only need immutable defaults.  It
# is built once at import time and remains safe because MAICPConfig is frozen.
PAPER_CONFIG = paper_config()


__all__ = [
    "ClassConfig",
    "MAICPConfig",
    "PAPER_CONFIG",
    "default_config",
    "paper_config",
]
