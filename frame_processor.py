import cv2
import torch
import numpy as np
from torchvision import transforms
from utils.plots import plot_one_box_kpt, colors, output_to_keypoint 
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utilities import calculate_angle  # Ensure this utility is properly imported or defined
import config

@torch.no_grad()
def process_frame(image, model, device, stride, names, hide_labels, hide_conf, line_thickness):
    

    # Assuming kpts is your keypoints array, with each keypoint having (x, y) or (x, y, confidence)
# and is structured as a flat list
    steps = 2 # Assuming each keypoint includes a confidence value

# Adjust indices for 0-based indexing
    right_hip_index = 8 * steps
    right_knee_index = 9 * steps
    right_ankle_index = 10 * steps
    right_wrist_index = 4 * steps

    

    orig_image = image  # Store frame
    frame_width = orig_image.shape[1]
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)  # Convert frame to RGB
    image = letterbox(image, (frame_width), stride=64, auto=True)[0]
    image_ = image.copy()
    image = transforms.ToTensor()(image)
    image = torch.tensor(np.array([image.numpy()]))

    image = image.to(device)  # Convert image data to device
    image = image.float()  # Convert image to float precision (cpu)

    output_data, _ = model(image)

    output_data = non_max_suppression_kpt(
        output_data,
        0.25,  # Conf. Threshold.
        0.65,  # IoU Threshold.
        nc=model.yaml['nc'],  # Number of classes.
        nkpt=model.yaml['nkpt'],  # Number of keypoints.
        kpt_label=True)

    output = output_to_keypoint(output_data)

    im0 = image[0].permute(1, 2, 0) * 255  # Change format [b, c, h, w] to [h, w, c] for displaying the image.
    im0 = im0.cpu().numpy().astype(np.uint8)

    im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)  # Reshape image format to (BGR)
    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # Normalization gain whwh

    for i, pose in enumerate(output_data):  # Detections per image

        if len(output_data):  # Check if no pose
            for c in pose[:, 5].unique():  # Print results
                n = (pose[:, 5] == c).sum()  # Detections per class
                print(f"No of Objects in Current Frame: {n}")

            for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:, :6])):  # Loop over poses for drawing on frame
                c = int(cls)  # Integer class
                kpts = pose[det_index, 6:]
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True),
                                 line_thickness=line_thickness, kpt_label=True, kpts=kpts, steps=3,
                                 orig_shape=im0.shape[:2])
                # Extract (x, y) coordinates for the right knee calculation
                #right_hip_index, right_knee_index, right_ankle_index = 8 * 3, 9 * 3, 10 * 3  # Adjust indices for keypoints including confidence

                right_hip = (kpts[right_hip_index], kpts[right_hip_index + 1])
                right_knee = (kpts[right_knee_index], kpts[right_knee_index + 1])
                right_ankle = (kpts[right_ankle_index], kpts[right_ankle_index + 1])

                # Calculate the angle at the right knee
#                angle_right_knee = calculate_angle(right_hip, right_knee, right_ankle)
#                angle_label = f"Right Knee: {int(angle_right_knee)}°"
#                cv2.putText(im0, angle_label, (int(right_knee[0]), int(right_knee[1] - 10)), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
#Wrist pose
    
#    for i, pose in enumerate(output_data):  # Detections per image
#        if len(pose):
#            # Extract right wrist position
#            right_wrist_pos = pose[:, right_wrist_index:right_wrist_index+2]
#            if right_wrist_pos.nelement() > 0:
#                right_wrist_y = right_wrist_pos[0][1].item()  # Get the Y position#
#
#                # Detect phase transitions based on wrist movement
#                if config.prev_wrist_y is not None:
#                    if right_wrist_y < config.prev_wrist_y:
#                        # Wrist moving upwards - potentially entering the Drive phase
#                        cv2.putText(im0, "Entering Drive", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
#                    elif right_wrist_y > config.prev_wrist_y:
#                        # Wrist moving downwards - potentially entering the Recovery phase
#                        cv2.putText(im0, "Entering Recovery", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#                
#                config.prev_wrist_y = right_wrist_y  # Update the tracked wrist position for the next frame





  #              print("The angle at the right knee is:", angle_right_knee, "degrees")

    return im0  # Return the processed frame
