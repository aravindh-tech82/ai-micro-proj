"""
predict.py
==========
Prediction Intelligence Layer for:
"Traffic Sign Recognition Using Neural Networks"

Loads the final improved CNN model:
models/best_traffic_sign_cnn_improved.pth

Integrates:
- Real-time CNN inference with softmax probability
- Domain Knowledge Base for all 43 GTSRB classes:
  * Category (Speed Limit, Prohibition, Warning, Priority, Mandatory, Other)
  * Driving Meaning
  * Recommended Driver Action
- Explainable AI via Grad-CAM (Heatmap & Overlay)

Exports:
  predict_traffic_sign(image)
"""

import os
import sys
import json
import numpy as np

# Ensure local source imports work
sys.path.insert(0, os.path.dirname(__file__))
from gradcam import generate_gradcam

# Comprehensive GTSRB Domain Knowledge Base for all 43 classes
TRAFFIC_SIGN_KNOWLEDGE_BASE = {
    0: {
        "name": "Speed limit (20km/h)",
        "category": "Speed Limit",
        "meaning": "Maximum vehicle speed permitted is 20 km/h.",
        "recommended_action": "Reduce speed to 20 km/h or below and watch for slow road conditions or pedestrian activity."
    },
    1: {
        "name": "Speed limit (30km/h)",
        "category": "Speed Limit",
        "meaning": "Maximum vehicle speed permitted is 30 km/h.",
        "recommended_action": "Maintain vehicle speed at or below 30 km/h, commonly near schools or residential zones."
    },
    2: {
        "name": "Speed limit (50km/h)",
        "category": "Speed Limit",
        "meaning": "Maximum vehicle speed permitted is 50 km/h.",
        "recommended_action": "Maintain vehicle speed at or below 50 km/h, standard within urban city limits."
    },
    3: {
        "name": "Speed limit (60km/h)",
        "category": "Speed Limit",
        "meaning": "Maximum vehicle speed permitted is 60 km/h.",
        "recommended_action": "Do not exceed 60 km/h; monitor road traffic flow."
    },
    4: {
        "name": "Speed limit (70km/h)",
        "category": "Speed Limit",
        "meaning": "Maximum vehicle speed permitted is 70 km/h.",
        "recommended_action": "Do not exceed 70 km/h; adjust speed for road geometry and traffic density."
    },
    5: {
        "name": "Speed limit (80km/h)",
        "category": "Speed Limit",
        "meaning": "Maximum vehicle speed permitted is 80 km/h.",
        "recommended_action": "Do not exceed 80 km/h; maintain a safe following distance."
    },
    6: {
        "name": "End of speed limit (80km/h)",
        "category": "Other",
        "meaning": "The 80 km/h speed restriction ends.",
        "recommended_action": "Resume standard road speed limits unless other restrictions apply."
    },
    7: {
        "name": "Speed limit (100km/h)",
        "category": "Speed Limit",
        "meaning": "Maximum vehicle speed permitted is 100 km/h.",
        "recommended_action": "Do not exceed 100 km/h, standard on rural highways."
    },
    8: {
        "name": "Speed limit (120km/h)",
        "category": "Speed Limit",
        "meaning": "Maximum vehicle speed permitted is 120 km/h.",
        "recommended_action": "Do not exceed 120 km/h; monitor highway traffic and maintain ample braking gap."
    },
    9: {
        "name": "No passing",
        "category": "Prohibition",
        "meaning": "Overtaking all motor vehicles is strictly prohibited.",
        "recommended_action": "Stay in lane; do not overtake any motor vehicle until restriction ends."
    },
    10: {
        "name": "No passing for vehicles over 3.5 metric tons",
        "category": "Prohibition",
        "meaning": "Heavy goods vehicles exceeding 3.5 metric tons are prohibited from overtaking.",
        "recommended_action": "Heavy transport vehicles must stay in lane and avoid overtaking."
    },
    11: {
        "name": "Right-of-way at next intersection",
        "category": "Priority",
        "meaning": "Drivers on this roadway have right-of-way at the upcoming intersection only.",
        "recommended_action": "Proceed with caution through the intersection; cross traffic must yield."
    },
    12: {
        "name": "Priority road",
        "category": "Priority",
        "meaning": "Drivers on this road have right-of-way at all subsequent intersections until terminated.",
        "recommended_action": "Proceed along the roadway; intersecting traffic must yield to you."
    },
    13: {
        "name": "Yield",
        "category": "Priority",
        "meaning": "Give way to other road users.",
        "recommended_action": "Slow down and yield when necessary."
    },
    14: {
        "name": "Stop",
        "category": "Prohibition / Regulatory",
        "meaning": "The driver must come to a complete stop.",
        "recommended_action": "Stop completely and proceed only when safe."
    },
    15: {
        "name": "No vehicles",
        "category": "Prohibition",
        "meaning": "All vehicular traffic is prohibited in both directions.",
        "recommended_action": "Do not enter this roadway; choose an alternate route."
    },
    16: {
        "name": "Vehicles over 3.5 metric tons prohibited",
        "category": "Prohibition",
        "meaning": "Vehicles with a gross vehicle weight exceeding 3.5 metric tons may not enter.",
        "recommended_action": "Heavy freight vehicles must turn around or take an approved detour."
    },
    17: {
        "name": "No entry",
        "category": "Prohibition",
        "meaning": "Entry is strictly prohibited for all vehicles from this direction.",
        "recommended_action": "Do not enter; typically indicates a one-way street or closed ramp."
    },
    18: {
        "name": "General caution",
        "category": "Warning",
        "meaning": "General unspecified hazard or danger ahead on the roadway.",
        "recommended_action": "Reduce speed, stay alert, and scan the road ahead for unexpected hazards."
    },
    19: {
        "name": "Dangerous curve to the left",
        "category": "Warning",
        "meaning": "A sharp, hazardous bend to the left ahead.",
        "recommended_action": "Decelerate before entering the curve and steer smoothly left."
    },
    20: {
        "name": "Dangerous curve to the right",
        "category": "Warning",
        "meaning": "A sharp, hazardous bend to the right ahead.",
        "recommended_action": "Decelerate before entering the curve and steer smoothly right."
    },
    21: {
        "name": "Double curve",
        "category": "Warning",
        "meaning": "A series of sharp curves ahead, beginning with the indicated direction.",
        "recommended_action": "Reduce speed substantially and prepare for sequential turns."
    },
    22: {
        "name": "Bumpy road",
        "category": "Warning",
        "meaning": "The road surface ahead is uneven, bumpy, or damaged.",
        "recommended_action": "Slow down to prevent vehicle suspension damage or loss of traction."
    },
    23: {
        "name": "Slippery road",
        "category": "Warning",
        "meaning": "Road surface may be slick or lose friction due to rain, mud, or frost.",
        "recommended_action": "Avoid harsh acceleration or braking; increase following distance."
    },
    24: {
        "name": "Road narrows on the right",
        "category": "Warning",
        "meaning": "The roadway narrows on the right-hand side ahead.",
        "recommended_action": "Merge safely toward the left and allow merging vehicles space."
    },
    25: {
        "name": "Road work",
        "category": "Warning",
        "meaning": "Construction or road maintenance operations ahead.",
        "recommended_action": "Slow down, watch for road workers and machinery, and follow flagger instructions."
    },
    26: {
        "name": "Traffic signals",
        "category": "Warning",
        "meaning": "Traffic light signals ahead, which may require stopping.",
        "recommended_action": "Prepare to slow down or stop if the signal turns amber or red."
    },
    27: {
        "name": "Pedestrians",
        "category": "Warning",
        "meaning": "Pedestrians likely to be crossing or walking along the roadway ahead.",
        "recommended_action": "Slow down, scan sidewalks and crosswalks, and be ready to stop for pedestrians."
    },
    28: {
        "name": "Children crossing",
        "category": "Warning",
        "meaning": "Children likely in the roadway, commonly near schools or playgrounds.",
        "recommended_action": "Drive slowly, watch roadside areas attentively, and be prepared to stop immediately."
    },
    29: {
        "name": "Bicycles crossing",
        "category": "Warning",
        "meaning": "Cyclists entering or crossing the roadway ahead.",
        "recommended_action": "Scan for cyclists, provide ample passing clearance, and yield when required."
    },
    30: {
        "name": "Beware of ice/snow",
        "category": "Warning",
        "meaning": "Hazardous roadway conditions with snow, slush, or black ice ahead.",
        "recommended_action": "Reduce speed significantly, brake gently, and maintain a large safety gap."
    },
    31: {
        "name": "Wild animals crossing",
        "category": "Warning",
        "meaning": "High likelihood of deer or large animals darting onto the road.",
        "recommended_action": "Scan road shoulders, reduce night-time driving speed, and do not swerve recklessly."
    },
    32: {
        "name": "End of all speed and passing limits",
        "category": "Other",
        "meaning": "All previously posted speed and passing restrictions are lifted.",
        "recommended_action": "Drive at default highway speed limits; exercise safe passing judgment."
    },
    33: {
        "name": "Turn right ahead",
        "category": "Mandatory",
        "meaning": "A right turn is required ahead.",
        "recommended_action": "Prepare to turn right."
    },
    34: {
        "name": "Turn left ahead",
        "category": "Mandatory",
        "meaning": "A left turn is required ahead.",
        "recommended_action": "Prepare to turn left; activate turn signal early."
    },
    35: {
        "name": "Ahead only",
        "category": "Mandatory",
        "meaning": "Vehicles must proceed straight ahead only; turning is prohibited.",
        "recommended_action": "Continue straight; do not turn left or right at the junction."
    },
    36: {
        "name": "Go straight or right",
        "category": "Mandatory",
        "meaning": "Vehicles may proceed straight or turn right; left turn prohibited.",
        "recommended_action": "Choose straight or right lane; do not attempt a left turn."
    },
    37: {
        "name": "Go straight or left",
        "category": "Mandatory",
        "meaning": "Vehicles may proceed straight or turn left; right turn prohibited.",
        "recommended_action": "Choose straight or left lane; do not attempt a right turn."
    },
    38: {
        "name": "Keep right",
        "category": "Mandatory",
        "meaning": "Traffic must keep to the right side.",
        "recommended_action": "Keep the vehicle to the right."
    },
    39: {
        "name": "Keep left",
        "category": "Mandatory",
        "meaning": "Traffic must pass an obstacle on the left-hand side.",
        "recommended_action": "Navigate the vehicle to the left of the barrier or island."
    },
    40: {
        "name": "Roundabout mandatory",
        "category": "Mandatory",
        "meaning": "Mandatory traffic circle / roundabout ahead.",
        "recommended_action": "Yield to traffic inside the roundabout and enter when clear."
    },
    41: {
        "name": "End of no passing",
        "category": "Other",
        "meaning": "Overtaking restriction for motor vehicles has ended.",
        "recommended_action": "Overtaking is permitted when safe and clear."
    },
    42: {
        "name": "End of no passing by vehicles over 3.5 metric tons",
        "category": "Other",
        "meaning": "Overtaking restriction for heavy vehicles over 3.5 tons has ended.",
        "recommended_action": "Heavy vehicles may overtake when road conditions and markings permit."
    }
}


def predict_traffic_sign(image, original_image=None, **kwargs):
    """
    Predicts the traffic sign class and provides domain intelligence and Grad-CAM explainability.

    Args:
        image (numpy.ndarray or torch.Tensor):
            Input image of shape (32, 32, 3) or (3, 32, 32), range [0, 1] or [0, 255].
        original_image (numpy.ndarray or PIL.Image, optional):
            Original uncompressed/unresized image for high-resolution Grad-CAM display.

    Returns:
        dict:
            predicted_class_id (int): Predicted GTSRB class ID (0 to 42)
            predicted_class (str): Official GTSRB sign name
            confidence (str): Softmax probability formatted as a percentage with 2 decimal places (e.g. "99.95%")
            category (str): Functional category (Speed Limit, Warning, Prohibition, etc.)
            meaning (str): Human-readable driving significance
            recommended_action (str): Prescribed driver response
            gradcam_heatmap (numpy.ndarray): 2D float32 array in [0.0, 1.0] at original image resolution
            gradcam_overlay (numpy.ndarray): 3D float32 array (H, W, 3) in [0.0, 1.0] at original image resolution
    """
    if original_image is None and "original_image" in kwargs:
        original_image = kwargs["original_image"]

    # 1. Run inference & Grad-CAM (using original_image if provided)
    pred_class_id, conf_float, heatmap, overlay = generate_gradcam(image, original_image=original_image)

    # 2. Retrieve domain knowledge
    info = TRAFFIC_SIGN_KNOWLEDGE_BASE.get(pred_class_id, {
        "name": f"Unknown Class {pred_class_id}",
        "category": "Unknown",
        "meaning": "No meaning registered.",
        "recommended_action": "Exercise caution."
    })

    # 3. Format confidence as percentage with 2 decimal places
    confidence_formatted = f"{conf_float * 100:.2f}%"

    return {
        "predicted_class_id": pred_class_id,
        "predicted_class": info["name"],
        "confidence": confidence_formatted,
        "category": info["category"],
        "meaning": info["meaning"],
        "recommended_action": info["recommended_action"],
        "gradcam_heatmap": heatmap,
        "gradcam_overlay": overlay
    }


def main():
    """
    Command-line test harness for predict_traffic_sign.
    Tests multiple unseen test set images and saves formatted verification outputs.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    npz_data_path = os.path.join(project_root, "dataset", "processed", "gtsrb_processed.npz")
    labels_json_path = os.path.join(project_root, "dataset", "processed", "class_labels.json")
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 65)
    print("TRAFFIC SIGN RECOGNITION - PREDICTION INTELLIGENCE LAYER")
    print("=" * 65)

    # Load unseen test dataset
    dataset_npz = np.load(npz_data_path)
    X_test = dataset_npz["X_test"]
    y_test = dataset_npz["y_test"]

    with open(labels_json_path, "r", encoding="utf-8") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    # Select representative test indices covering diverse categories:
    # 1. Speed limit (e.g. 100 km/h)
    # 2. Stop sign (Prohibition / Regulatory)
    # 3. Yield (Priority)
    # 4. Keep right (Mandatory)
    # 5. Bumpy road (Warning)
    # 6. Beware of ice/snow (Warning - difficult class)
    test_indices = [
        ("Speed Limit", 11),     # Speed limit (100km/h)
        ("Stop", 93),            # Stop
        ("Yield", 2),            # Yield
        ("Keep Right", 8),       # Keep right
        ("Bumpy Road", 88),      # Bumpy road
        ("Beware Ice/Snow", 41)  # Beware of ice/snow
    ]

    output_lines = []
    header_text = (
        "================================================================================\n"
        "             TRAFFIC SIGN PREDICTION INTELLIGENCE LAYER - TEST OUTPUTS          \n"
        "Project: Traffic Sign Recognition Using Neural Networks\n"
        "Model:   models/best_traffic_sign_cnn_improved.pth\n"
        "================================================================================\n"
    )
    output_lines.append(header_text)
    print(header_text)

    for desc, idx in test_indices:
        img = X_test[idx]
        actual_class_id = int(y_test[idx])
        actual_sign_name = label_map[actual_class_id]

        # Call prediction intelligence function
        result = predict_traffic_sign(img)

        # Verification tests on outputs
        assert 0 <= result["predicted_class_id"] <= 42, f"Invalid class ID: {result['predicted_class_id']}"
        assert result["predicted_class"] in [v["name"] for v in TRAFFIC_SIGN_KNOWLEDGE_BASE.values()]
        assert "%" in result["confidence"], f"Confidence not formatted as percentage: {result['confidence']}"
        conf_val = float(result["confidence"].replace("%", ""))
        assert 0.0 <= conf_val <= 100.0, f"Confidence value out of bounds: {conf_val}"
        assert result["category"] != "", "Category missing"
        assert result["meaning"] != "", "Meaning missing"
        assert result["recommended_action"] != "", "Recommended action missing"
        assert result["gradcam_heatmap"].shape == (32, 32), f"Heatmap shape mismatch: {result['gradcam_heatmap'].shape}"
        assert result["gradcam_overlay"].shape == (32, 32, 3), f"Overlay shape mismatch: {result['gradcam_overlay'].shape}"
        assert not np.isnan(result["gradcam_heatmap"]).any(), "NaN found in heatmap"
        assert not np.isinf(result["gradcam_heatmap"]).any(), "Inf found in heatmap"

        # Format display card
        block = (
            "--------------------------------------------------\n"
            "TRAFFIC SIGN PREDICTION\n"
            "--------------------------------------------------\n"
            f"Actual Sign:               [{actual_class_id}] {actual_sign_name}\n"
            f"Predicted Sign:            [{result['predicted_class_id']}] {result['predicted_class']}\n"
            f"Confidence:                {result['confidence']}\n"
            f"Category:                  {result['category']}\n"
            f"Meaning:                   {result['meaning']}\n"
            f"Recommended Driver Action: {result['recommended_action']}\n"
            "Grad-CAM generated:        YES\n"
            "--------------------------------------------------"
        )
        print(block)
        output_lines.append(block + "\n")

    # Save to results/prediction_examples.txt
    examples_txt_path = os.path.join(results_dir, "prediction_examples.txt")
    with open(examples_txt_path, "w", encoding="utf-8") as f:
        f.writelines(output_lines)
    print(f"\nSaved prediction examples to: {examples_txt_path}")

    print("\nVerification Checks: ALL 6 TEST EXAMPLES PASSED INTEGRITY TESTS.")
    print("=" * 65)


if __name__ == "__main__":
    main()
