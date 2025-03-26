// UGV_Waypoint_Following.cpp
//
// This script loads a list of waypoints from a text file and
// commands the UGV to sequentially pass through them using the
// CarRpcLibClient's moveOnPath function.
//
// Usage: UGV_Waypoint_Following <UGVName> <WaypointFilePath>

#include "common/common_utils/StrictMode.hpp"
STRICT_MODE_OFF
#ifndef RPCLIB_MSGPACK
#define RPCLIB_MSGPACK clmdep_msgpack
#endif
#include "rpc/rpc_error.h"
STRICT_MODE_ON

#include "vehicles/car/api/CarRpcLibClient.hpp"
#include "common/Common.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <thread>

using namespace msr::airlib;
using std::cerr;
using std::cout;
using std::endl;
using std::string;
using std::vector;

// Utility: sleep for a given number of seconds.
void sleep_for_seconds(int seconds)
{
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

// Load waypoints from a file. If the file has three values per line (x, y, z)
// they are used; if only two values are present, z is set to fixed_z.
std::vector<Vector3r> loadWaypoints(const string &file_path, float fixed_z)
{
    std::vector<Vector3r> waypoints;
    std::ifstream file(file_path);
    if (!file.is_open())
    {
        cerr << "Error opening waypoint file: " << file_path << endl;
        return waypoints;
    }

    string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        float x, y, z;
        if (iss >> x >> y >> z)
            waypoints.push_back(Vector3r(x, y, z));
        else if (iss >> x >> y)
            waypoints.push_back(Vector3r(x, y, fixed_z));
    }
    file.close();
    return waypoints;
}

int main(int argc, char *argv[])
{
    if (argc != 4)
    {
        cerr << "Usage: " << argv[0] << " <UGVName> <UGVLinearVelocity> <WaypointFilePath>" << endl;
        return 1;
    }

    string ugv_name = argv[1];
    string ugv_linear_vel = argv[2];
    string waypoint_file = argv[3];

    // For a ground vehicle, you can assume a fixed z value (e.g. 0 meters).
    float fixed_z = 0.0f;

    // Load the waypoints.
    std::vector<Vector3r> waypoints = loadWaypoints(waypoint_file, fixed_z);
    if (waypoints.empty())
    {
        cerr << "[" << ugv_name << "] No valid waypoints found in file: " << waypoint_file << endl;
        return 1;
    }
    cout << "[" << ugv_name << "] Loaded " << waypoints.size() << " waypoints." << endl;

    // Create a CarRpcLibClient instance.
    // (Adjust the IP address, port, and timeout as needed.)
    // CarRpcLibClient client("127.0.0.1", 41451, 60);
    msr::airlib::CarRpcLibClient client;
    client.confirmConnection();
    client.enableApiControl(true, ugv_name);

    // (Optional) You may wish to perform any additional initialization here.

    cout << "[" << ugv_name << "] Starting waypoint navigation..." << endl;

    // Define parameters for the moveOnPath function.
    float desired_velocity = std::stof(argv[2]); // Desired speed in m/s.

    float timeout_sec = 1000.0f;     // Total allowed time (in seconds) for the maneuver.
    float lookahead = 4.0f;          // Lookahead distance (in meters).

    // Call the moveOnPath function (which you added to CarRpcLibClient).
    bool success = client.moveOnPath(waypoints, desired_velocity, timeout_sec,
                                     lookahead, ugv_name);
    if (!success)
    {
        cerr << "[" << ugv_name << "] Failed to complete the waypoint mission." << endl;
        return 1;
    }

    cout << "[" << ugv_name << "] Waypoint mission complete." << endl;
    return 0;
}
