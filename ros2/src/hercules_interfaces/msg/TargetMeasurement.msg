# Abstract planar measurement; no simulator or NED/ENU frame conversion is implied.
string target_id
builtin_interfaces/Time stamp          # mission/estimator timestamp
builtin_interfaces/Time capture_stamp  # original wall-clock capture timestamp
builtin_interfaces/Time receipt_stamp  # ROS publication/receipt-domain timestamp
float64[2] position             # [x, y]
float64[4] covariance           # row-major 2x2 covariance
bool valid
string source_id                # producing agent or measurement source
string capture_id               # empty when the source has no capture identity
string sensor_id                # empty when not sensor-specific
bool visible
