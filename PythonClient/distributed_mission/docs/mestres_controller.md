# Mestres controller adaptation

The Mestres implementation uses the paper's local QP and projected
primal-dual/distributed structure as the selectable `mestres` backend.

UAVs use 3D double-integrator relative-degree-two barriers:

```text
h0 = ||p_i - p_j||^2 - D^2
psi1 = h0_dot + k1*h0
psi1_dot + k2*psi1 >= ||grad_[p,v](psi1)|| r_i
```

Huskies use the planar lookahead-point unicycle model. Pair constraints are
constructed only among agents of the same vehicle type. Obstacle constraints
are local because obstacles come from the ego sensor.

This is a sampled-data implementation and does not claim continuous-time
guarantees under arbitrary sensing, actuation, or solver delay.
