import cv2
from ultralytics import YOLO
import ollama
import tempfile
import numpy as np

def get_angle(p1, p2, p3):
    """Calculates the angle at p2 given points p1, p2, p3"""
    a = np.array(p1)
    b = np.array(p2) # Vertex
    c = np.array(p3)
    
    ba = a - b
    bc = c - b
    
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def analyze_squat(video_path):
    model = YOLO('yolo11n-pose.pt') 

    cap = cv2.VideoCapture(video_path)
    lowest_hip_y = 0
    best_frame = None
    biometrics = {}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        results = model(frame, verbose=False)[0]
        if results.keypoints:
            # YOLO COCO Keypoints: 11=L_Hip, 12=R_Hip, 13=L_Knee, 15=L_Ankle
            kpts = results.keypoints.xy[0].cpu().numpy()
            hip_y = (kpts[11][1] + kpts[12][1]) / 2 # Average hip height
            
            # Find the bottom of the squat
            if hip_y > lowest_hip_y:
                lowest_hip_y = hip_y
                best_frame = frame.copy()
                biometrics = {
                    "knee_angle": get_angle(kpts[11], kpts[13], kpts[15]),
                    "hip_depth_status": "Below Parallel" if kpts[11][1] > kpts[13][1] else "Above Parallel"
                }

    # 2. Save the "Bottom" frame for the Vision Model
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as temp_img:
        cv2.imwrite(temp_img.name, best_frame)
        
        # 3. Feed to LLM with Data
        response = ollama.chat(
            model='llama3.2-vision',
            messages=[{
                'role': 'user',
                'content': f"Analyze this squat frame. Measured Knee Angle: {biometrics['knee_angle']:.1f}°. "
                        f"Depth Status: {biometrics['hip_depth_status']}. Give 3 technical cues.",
                'images': [temp_img.name]
            }]
        )
    
    return response['message']['content']