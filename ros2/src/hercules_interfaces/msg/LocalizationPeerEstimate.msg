# Estimate announced by one agent for cooperative localization.
# receiver_id is optional and empty for a broadcast announcement.
std_msgs/Header header
string sender_id
string receiver_id
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
bool valid
