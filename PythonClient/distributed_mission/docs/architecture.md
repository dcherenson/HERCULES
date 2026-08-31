# Distributed mission architecture

The orchestrator runs each control tick as sense, estimate, perceive, nominal
formation control, CBF filtering, actuation, and simulation stepping.

`Husky1` is the fixed leader. It follows the mission goal; all other Huskies
and UAVs follow configurable body-frame slots around that leader. CBF output
has priority over formation tracking, so formation error can temporarily grow
when a safety constraint activates.

Localization currently copies AirSim truth into the same state interface that a
future estimator will implement. The CBF module receives only position,
velocity, vehicle type, uncertainty, neighbors, and local obstacle proxies.

Hero mode exposes multirotor and car RPC services separately. The
`AirSimFacade` hides this detail and creates one `MultirotorClient` on 41451
and one `CarClient` on 41452. No agent creates clients directly.

The conformal-prediction module is currently a placeholder and returns a zero
robustness margin. A future estimator can replace this without changing the
CBF interface.
