string sender_id
string receiver_id
string target_id
uint64 epoch_id
builtin_interfaces/Time stamp
float64[16] information_matrix    # row-major 4x4
float64[4] information_vector
bool has_state_covariance
float64[16] state_covariance      # row-major 4x4 when present
