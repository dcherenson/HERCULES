# Pairwise planar relative observation used by recursive decentralized
# localization. Coordinates are expressed in frame_id and are not transformed
# by the ROS transport.
std_msgs/Header header
string source_id
string target_id
string observer_id
string observed_id
string sensor_id
string capture_id
string frame_id
float64[2] relative_position
float64[2] relative_velocity
float64[4] covariance              # row-major 2x2 covariance
float64 range
float64 bearing
float64 timestamp
uint64 sequence
bool valid
