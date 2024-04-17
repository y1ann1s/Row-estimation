import cv2
import torch
import numpy as np
from torchvision import transforms
from utils.plots import plot_one_box_kpt, colors, output_to_keypoint
from utils.datasets import letterbox
from utils.general import non_max_suppression_kpt
from utilities import calculate_angle
import config

@torch.no_grad()
def process_frame(image, model, device, names, hide_labels, hide_conf):
    try:
        # Prepare the image and apply the model
        orig_image = image
        frame_width = orig_image.shape[1]
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        image = letterbox(image, (frame_width), stride=config.IMAGE_STRIDE, auto=True)[0]
        image = transforms.ToTensor()(image)
        image = torch.tensor(np.array([image.numpy()]), device=device)

        image = image.to(device)
        image = image.float()

        output_data, _ = model(image)

        # Apply non-max suppression
        output_data = non_max_suppression_kpt(
            output_data,
            config.CONF_THRESHOLD,  # Conf threshold from config
            config.IOU_THRESHOLD,  # IoU threshold from config
            nc=model.yaml['nc'],  # Number of classes
            nkpt=model.yaml['nkpt'],  # Number of keypoints
            kpt_label=True
        )

        if output_data is None or len(output_data) == 0:
            raise ValueError("No valid poses were detected.")

        output = output_to_keypoint(output_data)

        # Convert tensor to numpy for visualization after moving to CPU
        im0 = image[0].cpu().permute(1, 2, 0) * 255
        im0 = im0.numpy().astype(np.uint8)
        im0 = cv2.cvtColor(im0, cv2.COLOR_RGB2BGR)

        # Plot results on image
        for i, pose in enumerate(output_data):
            if len(pose):
                for det_index, (*xyxy, conf, cls) in enumerate(reversed(pose[:, :6])):
                    c = int(cls)
                    kpts = pose[det_index, 6:].cpu().numpy()  # Move keypoints to CPU and convert to NumPy
                    label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                    plot_one_box_kpt(xyxy, im0, label=label, color=colors(c, True),
                                     line_thickness=config.LINE_THICKNESS, kpt_label=True, kpts=kpts, steps=config.KEYPOINT_STEPS,
                                     orig_shape=im0.shape[:2])
                  
                    # Calculate angles using keypoints for both left and right
                    right_hip = (kpts[config.RIGHT_HIP_INDEX], kpts[config.RIGHT_HIP_INDEX + 1])
                    right_knee = (kpts[config.RIGHT_KNEE_INDEX], kpts[config.RIGHT_KNEE_INDEX + 1])
                    right_ankle = (kpts[config.RIGHT_ANKLE_INDEX], kpts[config.RIGHT_ANKLE_INDEX + 1])
                    left_hip = (kpts[config.LEFT_HIP_INDEX], kpts[config.LEFT_HIP_INDEX + 1])
                    left_knee = (kpts[config.LEFT_KNEE_INDEX], kpts[config.LEFT_KNEE_INDEX + 1])
                    left_ankle = (kpts[config.LEFT_ANKLE_INDEX], kpts[config.LEFT_ANKLE_INDEX + 1])

                    angle_right_knee = calculate_angle(right_hip, right_knee, right_ankle)
                    angle_left_knee = calculate_angle(left_hip, left_knee, left_ankle)
                    average_angle = (angle_right_knee + angle_left_knee) / 2

                    angle_label = f"Avg Knee Angle: {int(average_angle)}°"
                    cv2.putText(im0, angle_label, (int((right_knee[0] + left_knee[0]) / 2), int((right_knee[1] + left_knee[1]) / 2 - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

                    # Draw keypoint indices
                   # for kpt_idx in range(0, len(kpts), config.KEYPOINT_STEPS):
                   #     x, y = int(kpts[kpt_idx]), int(kpts[kpt_idx + 1])
                   #     cv2.putText(im0, str(kpt_idx // config.KEYPOINT_STEPS), (x, y - 10), 
                   #                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                    # Wrist detection for phase transitions
                    right_wrist_y = kpts[config.RIGHT_WRIST_INDEX + 1]
                    if config.prev_wrist_y is not None:
                        if right_wrist_y < config.prev_wrist_y:
                            cv2.putText(im0, "Entering Drive", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                        elif right_wrist_y > config.prev_wrist_y:
                            cv2.putText(im0, "Entering Recovery", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    config.prev_wrist_y = right_wrist_y  # Update for next frame tracking
                            #calculate rail angle
        im0=detect_rail(im0)
        return im0  # Return the processed frame

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None  # Return None or appropriate error handling

def detect_rail(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is not None:
        for rho, theta in lines[0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(image, (x1, y1), (x2, y2), (0, 0, 0), 10)  # Draw thick black line
    return image  # Function modifies image in place, no need to return
