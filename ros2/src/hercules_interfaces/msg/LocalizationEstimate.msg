# Cooperative estimate of one agent's own state.
# This message is deliberately algorithm-agnostic. The selected algorithm is
# reported as metadata; numerical fusion belongs outside the ROS transport.
std_msgs/Header header
string agent_id
string frame_id
float64[3] position
float64[3] velocity
float64[4] orientation             # [w, x, y, z]
float64 yaw
float64 yaw_rate
float64[9] covariance               # row-major 3x3 position covariance
float64 robustness_margin
string algorithm
uint64 sequence
bool initialized
bool valid
bool stale
