// Created by Sandilya Sai Garimella.
/*
This script enables a drone to load its waypoints and follow them in HERCULES simulator. Each drone must
launch an instance of this script's executable with the drone name, corresponding waypoints text file,
and an optional velocity parameter.
*/

#include "common/common_utils/StrictMode.hpp"
STRICT_MODE_OFF
#ifndef RPCLIB_MSGPACK
#define RPCLIB_MSGPACK clmdep_msgpack
#endif
#include "rpc/rpc_error.h"
STRICT_MODE_ON

#include "vehicles/multirotor/api/MultirotorRpcLibClient.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <thread>

using namespace msr::airlib;

void sleep_for_seconds(int seconds)
{
    std::this_thread::sleep_for(std::chrono::seconds(seconds));
}

std::vector<Vector3r> loadWaypoints(const std::string &file_path, float fixed_z)
{
    std::vector<Vector3r> waypoints;
    std::ifstream file(file_path);

    if (!file.is_open())
    {
        std::cerr << "Error opening waypoint file: " << file_path << std::endl;
        return waypoints;
    }

    std::string line;
    while (std::getline(file, line))
    {
        std::istringstream iss(line);
        float x, y, z;
        if (iss >> x >> y >> z)
        {
            waypoints.emplace_back(Vector3r(x, y, fixed_z));
        }
    }

    file.close();
    return waypoints;
}

int main(int argc, char *argv[])
{
    if (argc < 3 || argc > 5)
    {
        std::cerr << "Usage: " << argv[0] << " <DroneName> <WaypointFilePath> [WaypointVelocity] [FlyAltitude]" << std::endl;
        return 1;
    }

    std::string drone_name = argv[1];
    std::string waypoint_file = argv[2];
    float waypoint_flight_velocity = (argc >= 4) ? std::stof(argv[3]) : 2.0f;
    float fly_altitude = (argc == 5) ? std::stof(argv[4]) : -35.0f;
    float return_home_velocity = 3.0f;

    try
    {
        MultirotorRpcLibClient client("localhost", 41451, 60.0f);
        client.confirmConnection();
        client.enableApiControl(true, drone_name);
        client.armDisarm(true, drone_name);

        std::cout << "[" << drone_name << "] Taking off..." << std::endl;
        client.takeoffAsync(20.0f, drone_name)->waitOnLastTask();
        sleep_for_seconds(1);

        auto state = client.getMultirotorState(drone_name);
        if (state.landed_state == LandedState::Landed)
        {
            std::cerr << "[" << drone_name << "] Takeoff failed. Exiting..." << std::endl;
            return 1;
        }

        std::cout << "[" << drone_name << "] Moving to altitude " << -fly_altitude << " meters..." << std::endl;
        client.moveToZAsync(fly_altitude, 5.0f, 30.0f, YawMode(false, 0), -1.0f, 1.0f, drone_name)->waitOnLastTask();

        // // Block until user hits Enter
        // std::cout << "[" << drone_name << "] Reached target altitude. Press ENTER to begin waypoint navigation..." << std::endl;
        // std::cin.get();

        std::vector<Vector3r> waypoints = loadWaypoints(waypoint_file, fly_altitude);
        if (waypoints.empty())
        {
            std::cerr << "[" << drone_name << "] No valid waypoints found. Exiting..." << std::endl;
            return 1;
        }

        std::cout << "[" << drone_name << "] Flying through " << waypoints.size()
                  << " waypoints at velocity " << waypoint_flight_velocity << " m/s..." << std::endl;

        client.moveOnPathAsync(waypoints, waypoint_flight_velocity, 120.0f,
                               DrivetrainType::ForwardOnly,
                               YawMode(false, 0), 20.0f, 1.0f,
                               drone_name)
            ->waitOnLastTask();

        sleep_for_seconds(2);

        std::cout << "[" << drone_name << "] Returning to origin and landing..." << std::endl;
        client.moveToPositionAsync(0, 0, fly_altitude, return_home_velocity, 30.0f,
                                   DrivetrainType::ForwardOnly,
                                   YawMode(false, 0), -1.0f, 1.0f,
                                   drone_name)
            ->waitOnLastTask();

        std::cout << "[" << drone_name << "] Hovering before landing..." << std::endl;
        client.hoverAsync(drone_name)->waitOnLastTask();
        sleep_for_seconds(5);

        client.moveToZAsync(-2.0f, 5.0f, 30.0f, YawMode(false, 0), -1.0f, 1.0f, drone_name)->waitOnLastTask();
        sleep_for_seconds(2);

        std::cout << "[" << drone_name << "] Initiating landing..." << std::endl;
        client.landAsync(60.0f, drone_name)->waitOnLastTask();

        client.armDisarm(false, drone_name);
        client.enableApiControl(false, drone_name);

        std::cout << "[" << drone_name << "] Flight complete." << std::endl;
    }
    catch (rpc::rpc_error &e)
    {
        std::cerr << "RPC Exception occurred for drone [" << drone_name << "]: "
                  << e.get_error().as<std::string>() << std::endl;
        return 1;
    }

    return 0;
}
