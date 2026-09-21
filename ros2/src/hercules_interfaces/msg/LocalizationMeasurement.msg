# Local self-state observation supplied to the cooperative-localization adapter.
# Position and velocity use the frame named by frame_id; no NED/ENU conversion
# is performed at this interface boundary.
std_msgs/Header header
string agent_id
string source_id
string frame_id
float64[3] position
float64[3] velocity
float64[4] orientation             # [w, x, y, z]
float64 yaw
float64 yaw_rate
float64[9] covariance               # row-major 3x3 position covariance
float64 robustness_margin
bool valid
