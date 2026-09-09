# Canonical simulator ground truth. This is not a localization estimate.
std_msgs/Header header
string agent_id
string vehicle_type

# AirSim world NED: +X north, +Y east, +Z down. Units are metres and m/s.
float64[3] position
float64[3] velocity

# Canonical AirSim quaternion in explicit [w, x, y, z] order.
float64[4] orientation
float64 yaw
float64 yaw_rate

bool valid
