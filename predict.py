import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained model
model = joblib.load("asl_model.pkl")

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

with HandLandmarker.create_from_options(options) as landmarker:

    camera = cv2.VideoCapture(0)

    while True:

        success, frame = camera.read()

        if not success:
            print("Could not access camera")
            break

        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=frame_rgb
        )

        result = landmarker.detect(mp_image)

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

            # Get 63 coordinates
            landmarks = []

            for landmark in hand:
                landmarks.append(landmark.x)
                landmarks.append(landmark.y)
                landmarks.append(landmark.z)

            # Convert to NumPy array
            input_data = np.array(landmarks).reshape(1, -1)

            # Predict
            prediction = model.predict(input_data)

            label = prediction[0]

            # Draw landmarks
            for landmark in hand:

                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(
                    frame,
                    (x, y),
                    5,
                    (0, 255, 0),
                    -1
                )

            # Display prediction
            cv2.putText(
                frame,
                f"Prediction: {label}",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        cv2.imshow("ASL Translator", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()