# Planar Gaussian belief exchanged by the global covariance-intersection
# implementation. `global_mean` is ordered as [x1,y1,...,xn, yaw_sender]
# and `global_covariance` is the matching row-major matrix. The fixed-size
# marginal fields are retained for simple peer transports.
std_msgs/Header header
string agent_id
string sender_id
string frame_id
string[] ordered_agent_ids
float64[] global_mean
float64[] global_covariance
float64[2] mean
float64[4] covariance              # row-major 2x2 covariance
float64 timestamp
float64 confidence_weight
string source_id
uint64 sequence
bool valid
