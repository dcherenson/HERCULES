"""Smoke test for the server-side synthetic image types ThermalIR (11) and
NightVision (12), synthesized on CPU in the AirSim plugin from underlying
captures rendered in the same simGetImages batch.

Run with the simulator up (any environment):
    python test_synthetic_ir_nvg.py
Outputs PNGs to /tmp/synthetic_test/ and asserts basic sanity:
  - ThermalIR is non-constant (not all one value)
  - NightVision is green-dominant (mean G > mean R and mean B)
"""

import os

import numpy as np

import hercules_cosysairsim as airsim

OUT_DIR = "/tmp/synthetic_test"
CAMERA = "front_center"


def save_png(path, img):
    try:
        from PIL import Image
        Image.fromarray(img).save(path)
    except ImportError:
        try:
            import cv2
            cv2.imwrite(path, img[:, :, ::-1] if img.ndim == 3 else img)
        except ImportError:
            import matplotlib.image
            matplotlib.image.imsave(path, img)
    print(f"saved {path}")


def to_array(response):
    img = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    return img.reshape(response.height, response.width, 3)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    client = airsim.MultirotorClient()
    client.confirmConnection()

    # --- call 1: mixed batch of normal + synthetic types, all same frame ---
    requests = [
        airsim.ImageRequest(CAMERA, airsim.ImageType.Scene, False, False),
        airsim.ImageRequest(CAMERA, airsim.ImageType.Segmentation, False, False),
        airsim.ImageRequest(CAMERA, airsim.ImageType.ThermalIR, False, False),
        airsim.ImageRequest(CAMERA, airsim.ImageType.NightVision, False, False),
    ]
    names = ["Scene", "Segmentation", "ThermalIR", "NightVision"]
    responses = client.simGetImages(requests)
    assert len(responses) == len(requests), (
        f"expected {len(requests)} responses, got {len(responses)}")

    images = {}
    for name, response in zip(names, responses):
        assert response.width > 0 and response.height > 0, (
            f"{name}: empty response (message: {response.message!r})")
        img = to_array(response)
        images[name] = img
        print(f"{name}: shape={img.shape} dtype={img.dtype} "
              f"image_type={response.image_type} timestamp={response.time_stamp}")
        save_png(os.path.join(OUT_DIR, f"{name}.png"), img)

    # --- call 2: ONLY the synthetic types ---
    responses2 = client.simGetImages([
        airsim.ImageRequest(CAMERA, airsim.ImageType.ThermalIR, False, False),
        airsim.ImageRequest(CAMERA, airsim.ImageType.NightVision, False, False),
    ])
    assert len(responses2) == 2, f"expected 2 responses, got {len(responses2)}"
    thermal2 = to_array(responses2[0])
    nvg2 = to_array(responses2[1])
    print(f"call2 ThermalIR: shape={thermal2.shape} dtype={thermal2.dtype}")
    print(f"call2 NightVision: shape={nvg2.shape} dtype={nvg2.dtype}")
    save_png(os.path.join(OUT_DIR, "ThermalIR_only.png"), thermal2)
    save_png(os.path.join(OUT_DIR, "NightVision_only.png"), nvg2)

    # --- sanity checks ---
    thermal = images["ThermalIR"]
    assert thermal.min() != thermal.max(), "ThermalIR image is constant"

    nvg = images["NightVision"]
    mean_r = nvg[:, :, 0].mean()
    mean_g = nvg[:, :, 1].mean()
    mean_b = nvg[:, :, 2].mean()
    print(f"NightVision channel means: R={mean_r:.2f} G={mean_g:.2f} B={mean_b:.2f}")
    assert mean_g > mean_r and mean_g > mean_b, (
        f"NightVision is not green-dominant (R={mean_r:.2f}, G={mean_g:.2f}, B={mean_b:.2f})")

    print("All checks passed.")


if __name__ == "__main__":
    main()
