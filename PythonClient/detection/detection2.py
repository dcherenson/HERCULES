import setup_path 
import cosysairsim as airsim
import cv2
import numpy as np 
import pprint

# connect to the AirSim simulator
client = airsim.VehicleClient()
client.confirmConnection()

# set camera name and image type to request images and detections
camera_name = "front_center"
image_type = airsim.ImageType.Scene

# set detection radius in [cm]
client.simSetDetectionFilterRadius(camera_name, image_type, 200 * 100) 

client.simClearDetectionMeshNames(camera_name, image_type)

# add desired object name to detect in wild card/regex format
client.simAddDetectionFilterMeshName(camera_name, image_type, "BP_CrowdCharacter*")

while True:
    rawImage = client.simGetImage(camera_name, image_type)
    if not rawImage:
        continue
    png = cv2.imdecode(airsim.string_to_uint8_array(rawImage), cv2.IMREAD_UNCHANGED)
    detectedObjects = client.simGetDetections(camera_name, image_type)
    if detectedObjects:
        for detectedObject in detectedObjects:
            s = pprint.pformat(detectedObject)
            print("Cylinder: %s" % s)

            cv2.rectangle(png, (int(detectedObject.box2D.min.x_val), int(detectedObject.box2D.min.y_val)), (int(detectedObject.box2D.max.x_val), int(detectedObject.box2D.max.y_val)), (0, 0, 255), 5)
            cv2.putText(png, detectedObject.name, (int(detectedObject.box2D.min.x_val), int(detectedObject.box2D.min.y_val - 10)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

    image_resized = cv2.resize(png, (0, 0), fx=0.25, fy=0.25)
    cv2.imshow('Image Redimensionnée', image_resized)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # elif cv2.waitKey(1) & 0xFF == ord('c'):
    #     client.simClearDetectionMeshNames(camera_name, image_type)
    # elif cv2.waitKey(1) & 0xFF == ord('a'):
    #     client.simAddDetectionFilterMeshName(camera_name, image_type,  "StaticMeshActor_.*_.*_.*$")
cv2.destroyAllWindows()