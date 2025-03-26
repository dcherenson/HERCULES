// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef air_CarRpcLibClient_hpp
#define air_CarRpcLibClient_hpp

#include "common/Common.hpp"
#include <functional>
#include "common/CommonStructs.hpp"
#include "vehicles/car/api/CarApiBase.hpp"
#include "api/RpcLibClientBase.hpp"
#include "common/ImageCaptureBase.hpp"

namespace msr
{
    namespace airlib
    {

        class CarRpcLibClient : public RpcLibClientBase
        {
        public:
            CarRpcLibClient(const string &ip_address = "localhost", uint16_t port = RpcLibPort, float timeout_sec = 60);

            void setCarControls(const CarApiBase::CarControls &controls, const std::string &vehicle_name = "");
            CarApiBase::CarState getCarState(const std::string &vehicle_name = "");
            CarApiBase::CarControls getCarControls(const std::string &vehicle_name = "");
            virtual ~CarRpcLibClient(); // required for pimpl

            // Synchronous waypoint controller for the car.
            // 'path' is a vector of waypoints (in NED coordinates).
            // 'desired_velocity' is the target speed in m/s.
            // 'timeout_sec' is the overall time allowed to complete the maneuver.
            // 'lookahead' is the fixed lookahead distance (in meters).
            // 'adaptive_lookahead' can be used if you later wish to adjust the lookahead adaptively.
            // 'vehicle_name' identifies the car.
            bool moveOnPath(const std::vector<Vector3r> &path, float desired_velocity, float timeout_sec,
                            float lookahead, const std::string &vehicle_name);
        };
    }
} // namespace
#endif
