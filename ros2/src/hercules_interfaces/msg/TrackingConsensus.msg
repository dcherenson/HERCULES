# Identity is explicit so a future synchronous transport cannot mix epochs or rounds.
string sender_id
string target_id
uint64 epoch_id
uint32 round_id
builtin_interfaces/Time estimate_stamp

# N epoch times in ascending order. trajectory_state has exactly 4*N entries,
# flattened epoch-major as [x0,y0,vx0,vy0,x1,y1,vx1,vy1,...].
float64[] trajectory_epoch_times
float64[] trajectory_state

# Row-major 4x4 covariance of the latest trajectory state.
float64[16] state_covariance
bool active
