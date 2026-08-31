# Wang controller and limitations

The `wang` backend uses bounded double-integrator acceleration for UAVs and
Huskies. Same-type pair constraints use the decentralized split corresponding
to Eq. 12. Local obstacle proxies are treated as stationary bodies.

The v1 implementation does not include the paper's full theorem-backed
feasible-certificate construction. Infeasible QPs trigger bounded maximum
braking and are logged as fallback events. Wang Husky acceleration is then
projected into speed and steering commands, which further weakens the direct
double-integrator guarantee.
