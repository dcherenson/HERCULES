# Nominal leader-follower formation control

The nominal policy is intentionally obstacle-unaware so safety behavior can be
diagnosed independently from planning.

`Husky1` follows the mission goal. Every other vehicle has a body-frame slot
`delta_i` and receives

```text
p_i_ref = p_leader + R(yaw_leader) delta_i
v_i_ref = v_leader + omega_leader x R(yaw_leader) delta_i
```

UAVs and Wang-mode Huskies use bounded position/velocity PD acceleration.
Mestres-mode Huskies use bounded linear speed and heading-error yaw rate.
The default slots and gains are in `modules/formation_control.py` and are
intended to be changed through configuration later. The default UAV slots are
the four corners and center of a 4 m x 4 m horizontal box. The UGV slots are
an equilateral triangle of side length 4 m, with `Husky1` at the front vertex.
UAV altitude is held at a fixed global AirSim NED reference (`-5 m` by
default), rather than being offset from the Husky body origin.

The formation controller does not provide obstacle routing or deadlock
resolution. A safe stop in front of a block is an acceptable v1 result.
