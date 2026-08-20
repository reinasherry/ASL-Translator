import cv2
import mediapipe as mp
import csv
import os

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Create data folder if it doesn't exist
os.makedirs("data", exist_ok=True)

# Open CSV file
file = open("data/asl_data.csv", "a", newline="")
writer = csv.writer(file)

# Create model options
options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

# Start hand detector
with HandLandmarker.create_from_options(options) as landmarker:

    camera = cv2.VideoCapture(0)

    print("Press A to collect an A sign")
    print("Press Q to quit")

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

        # Draw hand landmarks
        if result.hand_landmarks:

            hand = result.hand_landmarks[0]

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

        cv2.imshow("ASL Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF

        # Save A example
        if key == ord("a"):

            if result.hand_landmarks:

                hand = result.hand_landmarks[0]

                row = []

                for landmark in hand:
                    row.append(landmark.x)
                    row.append(landmark.y)
                    row.append(landmark.z)

                row.append("A")

                writer.writerow(row)

                print("Saved A example!")

            else:
                print("No hand detected!")

        # Quit
        if key == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
    file.close()

print("Data collection finished.")
