# A coordinator freezes the simulated communication graph once per estimator epoch.
uint64 epoch_id
builtin_interfaces/Time stamp
string target_id
string[] agent_ids
uint8[] adjacency                 # row-major N x N, where N = agent_ids.size()
