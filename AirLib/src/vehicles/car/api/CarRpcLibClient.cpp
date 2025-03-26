// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// in header only mode, control library is not available
#ifndef AIRLIB_HEADER_ONLY
// RPC code requires C++14. If build system like Unreal doesn't support it then use compiled binaries
#ifndef AIRLIB_NO_RPC
// if using Unreal Build system then include precompiled header file first

#include "vehicles/car/api/CarRpcLibClient.hpp"

#include "common/Common.hpp"
#include "common/ClockFactory.hpp"
#include <thread>
STRICT_MODE_OFF

#ifndef RPCLIB_MSGPACK
#define RPCLIB_MSGPACK clmdep_msgpack
#endif // !RPCLIB_MSGPACK

#ifdef nil
#undef nil
#endif // nil

#include "common/common_utils/WindowsApisCommonPre.hpp"
#undef FLOAT
#undef check
#include "rpc/client.h"
// TODO: HACK: UE4 defines macro with stupid names like "check" that conflicts with msgpack library
#ifndef check
#define check(expr) (static_cast<void>((expr)))
#endif
#include "common/common_utils/WindowsApisCommonPost.hpp"

#include "vehicles/car/api/CarRpcLibAdaptors.hpp"

STRICT_MODE_ON
#ifdef _MSC_VER
__pragma(warning(disable : 4239))
#endif

    namespace msr
{
    namespace airlib
    {

        typedef msr::airlib_rpclib::CarRpcLibAdaptors CarRpcLibAdaptors;

        CarRpcLibClient::CarRpcLibClient(const string &ip_address, uint16_t port, float timeout_sec)
            : RpcLibClientBase(ip_address, port, timeout_sec)
        {
        }

        CarRpcLibClient::~CarRpcLibClient()
        {
        }

        void CarRpcLibClient::setCarControls(const CarApiBase::CarControls &controls, const std::string &vehicle_name)
        {
            static_cast<rpc::client *>(getClient())->call("setCarControls", CarRpcLibAdaptors::CarControls(controls), vehicle_name);
        }

        CarApiBase::CarState CarRpcLibClient::getCarState(const std::string &vehicle_name)
        {
            return static_cast<rpc::client *>(getClient())->call("getCarState", vehicle_name).as<CarRpcLibAdaptors::CarState>().to();
        }
        CarApiBase::CarControls CarRpcLibClient::getCarControls(const std::string &vehicle_name)
        {
            return static_cast<rpc::client *>(getClient())->call("getCarControls", vehicle_name).as<CarRpcLibAdaptors::CarControls>().to();
        }

        // helper functions
        // Normalize an angle to the range [-pi, pi].
        float normalizeAngle(float angle)
        {
            while (angle > M_PI)
                angle -= 2 * M_PI;
            while (angle < -M_PI)
                angle += 2 * M_PI;
            return angle;
        }

        // Compute Euclidean distance in the XY (horizontal) plane.
        float distance2D(const Vector3r &a, const Vector3r &b)
        {
            return std::sqrt(std::pow(a.x() - b.x(), 2) + std::pow(a.y() - b.y(), 2));
        }

        // Helper function to compute a lookahead point on the path.
        // This function walks through the path segments and returns a point that is
        // 'lookahead_distance' ahead of the current position.
        Vector3r getLookaheadPoint(const Vector3r &current, const vector<Vector3r> &path, float lookahead_distance)
        {
            // Iterate over segments defined by consecutive waypoints.
            for (size_t i = 0; i < path.size() - 1; ++i)
            {
                Vector3r start = path[i];
                Vector3r end = path[i + 1];
                Vector3r segment = end - start;
                float seg_length = segment.norm();
                if (seg_length < 1e-6f)
                    continue; // Skip very short segments.
                Vector3r dir = segment / seg_length;
                float t = (current - start).dot(dir);
                t = std::max(0.0f, std::min(t, seg_length));
                Vector3r proj = start + dir * t;
                float dist_to_proj = distance2D(current, proj);
                // If the lookahead circle intersects this segment:
                if (dist_to_proj <= lookahead_distance && (lookahead_distance - dist_to_proj) <= (seg_length - t))
                {
                    float offset = lookahead_distance - dist_to_proj;
                    return proj + dir * offset;
                }
            }
            // If no segment meets the condition, return the final waypoint.
            return path.back();
        }

        // Extract yaw (heading angle) from a quaternion.
        // (This uses the standard conversion formula and assumes the quaternion is in NED frame.)
        float getYawFromQuaternion(const Quaternionr &q)
        {
            return std::atan2(2 * (q.w() * q.z() + q.x() * q.y()),
                              1 - 2 * (q.y() * q.y() + q.z() * q.z()));
        }

        bool CarRpcLibClient::moveOnPath(const vector<Vector3r> &path, float desired_velocity, float timeout_sec,
                                         float lookahead, const std::string &vehicle_name)
        {
            if (path.empty())
            {
                std::cerr << "Error: Provided path is empty." << std::endl;
                return false;
            }

            auto start_time = std::chrono::steady_clock::now();
            float control_period = 0.1f; // 100 ms control loop period.

            // Controller gains (tune these based on your vehicle dynamics).
            const float Kp_steering = 1.0f;        // Proportional gain for steering.
            const float max_steering_angle = 0.5f; // Maximum steering angle in radians.
            const float throttle_gain = 0.5f;      // Gain for throttle command.
            const float threshold = 1.0f;          // Distance threshold for waypoint completion.

            // Detect if the path is a closed loop.
            bool closed_loop = (distance2D(path.front(), path.back()) < 1e-3);
            bool passed_start = false; // Will be set true once the UGV leaves the start area.

            while (true)
            {
                // Check for timeout.
                auto current_time = std::chrono::steady_clock::now();
                float elapsed_sec = std::chrono::duration_cast<std::chrono::duration<float>>(current_time - start_time).count();
                if (elapsed_sec > timeout_sec)
                {
                    std::cerr << "Timeout reached while following the path." << std::endl;
                    break;
                }

                // Retrieve current car state.
                CarApiBase::CarState car_state = this->getCarState(vehicle_name);
                Vector3r current_pos = car_state.kinematics_estimated.pose.position;
                float current_speed = car_state.speed; // In m/s.

                // For this example, we use a fixed lookahead distance.
                float current_lookahead = lookahead;
                Vector3r target = getLookaheadPoint(current_pos, path, current_lookahead);

                // Compute desired heading from current position to target.
                float desired_heading = std::atan2(target.y() - current_pos.y(), target.x() - current_pos.x());
                float current_heading = getYawFromQuaternion(car_state.kinematics_estimated.pose.orientation);
                float heading_error = normalizeAngle(desired_heading - current_heading);

                // Compute steering command.
                float steering_cmd = Kp_steering * heading_error;
                if (steering_cmd > max_steering_angle)
                    steering_cmd = max_steering_angle;
                else if (steering_cmd < -max_steering_angle)
                    steering_cmd = -max_steering_angle;

                // Compute throttle command.
                float throttle_cmd = 0.0f;
                if (current_speed < desired_velocity)
                {
                    throttle_cmd = throttle_gain * (desired_velocity - current_speed);
                    if (throttle_cmd > 1.0f)
                        throttle_cmd = 1.0f;
                }
                else
                {
                    throttle_cmd = 0.0f;
                }

                // Send the command using the client instance.
                CarApiBase::CarControls controls;
                controls.steering = steering_cmd;
                controls.throttle = throttle_cmd;
                controls.handbrake = false;
                controls.is_manual_gear = false;
                this->setCarControls(controls, vehicle_name);

                // Termination check:
                if (closed_loop)
                {
                    // For closed-loop paths, first ensure the UGV has left the vicinity of the starting point.
                    if (!passed_start && distance2D(current_pos, path.front()) > threshold * 2)
                        passed_start = true;
                    // If the UGV has left and then returned close to the starting point, finish.
                    if (passed_start && distance2D(current_pos, path.front()) < threshold)
                    {
                        std::cout << "Closed-loop: Returned to starting point." << std::endl;
                        break;
                    }
                }
                else
                {
                    // For open paths, check if we're near the final waypoint.
                    if (distance2D(current_pos, path.back()) < threshold)
                    {
                        std::cout << "Destination reached." << std::endl;
                        break;
                    }
                    // Additional check: if there are at least two waypoints, check if the UGV has passed the last checkpoint.
                    if (path.size() >= 2)
                    {
                        Vector3r last_segment = path.back() - path[path.size() - 2];
                        float segment_length = last_segment.norm();
                        // Avoid division by zero.
                        if (segment_length > 1e-6f)
                        {
                            Vector3r dir = last_segment / segment_length;
                            Vector3r to_vehicle = current_pos - path[path.size() - 2];
                            float proj = to_vehicle.dot(dir);
                            if (proj > segment_length)
                            {
                                std::cout << "Destination passed." << std::endl;
                                break;
                            }
                        }
                    }
                }

                // Wait for the control period.
                std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(control_period * 1000)));
            }

            // Stop the vehicle by applying the handbrake.
            CarApiBase::CarControls stop_controls;
            stop_controls.throttle = 0;
            stop_controls.steering = 0;
            stop_controls.handbrake = true;
            this->setCarControls(stop_controls, vehicle_name);

            return true;
        }

    }
} // namespace

#endif
#endif
