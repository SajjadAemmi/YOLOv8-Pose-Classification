import os
import argparse
import numpy as np
import tqdm
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from matplotlib import pyplot as plt

from src.full_body_pose_embedder import FullBodyPoseEmbedder
from src.pose_classifier import PoseClassifier
from src.ema_dict_smoothing import EMADictSmoothing
from src.repetition_counter import RepetitionCounter
from src.pose_classification_visualizer import PoseClassificationVisualizer


def show_image(img):
    plt.imshow(img)
    plt.pause(0.001)
    # plt.show()


parser = argparse.ArgumentParser()
parser.add_argument('--input-video-path', default="io/input/IMG_7685.mp4", type=str)
parser.add_argument('--class-name', default="ez_bar_preacher_curl_down", type=str)
parser.add_argument('--output-dir-path', default="io/output/", type=str)
args = parser.parse_args()

output_video_path = os.path.join(args.output_dir_path, os.path.basename(args.input_video_path))

video_cap = cv2.VideoCapture(args.input_video_path)

# Get some video parameters to generate output video with classification.
video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
video_fps = video_cap.get(cv2.CAP_PROP_FPS)
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Folder with pose class CSVs. That should be the same folder you're using while
# building classifier to output CSVs.
pose_samples_folder = 'fitness_poses_csvs_out'

# Initialize tracker.
pose_tracker = YOLO('yolov8x-pose.pt')

# Initialize embedder.
pose_embedder = FullBodyPoseEmbedder()

# Initialize classifier.
# Check that you are using the same parameters as during bootstrapping.
pose_classifier = PoseClassifier(
    pose_samples_folder=pose_samples_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)

# # Uncomment to validate target poses used by classifier and find outliers.
# outliers = pose_classifier.find_pose_sample_outliers()
# print('Number of pose sample outliers (consider removing them): ', len(outliers))

# Initialize EMA smoothing.
pose_classification_filter = EMADictSmoothing(
    window_size=10,
    alpha=0.2)

# Initialize counter.
repetition_counter = RepetitionCounter(
    class_name=args.class_name,
    enter_threshold=6,
    exit_threshold=4)

# Initialize renderer.
pose_classification_visualizer = PoseClassificationVisualizer(
    class_name=args.class_name,
    plot_x_max=video_n_frames,
    # Graphic looks nicer if it's the same as `top_n_by_mean_distance`.
    plot_y_max=10)

# Open output video.
out_video = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

frame_idx = 0
output_frame = None
plt.figure(figsize=(10, 10))
with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:
    while True:
        # Get next frame of the video.
        success, input_frame = video_cap.read()
        if not success:
            break

        # Run pose tracker.
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        results = pose_tracker.predict(input_frame, verbose=False, device=0)
        largest_area = 0
        for index, item in enumerate(results[0].boxes.xywh):
            area = item[2] * item[3]
            if area > largest_area:
                largest_area = area
                largest_bbox_index = index
                largest_item = item

        pose_landmarks = results[0].keypoints[largest_bbox_index].data

        # Draw pose prediction.
        output_frame = input_frame.copy()
        if pose_landmarks is not None:
            annotator = Annotator(output_frame)
            for k in reversed(pose_landmarks):
                annotator.kpts(k, (video_width, video_height))

            output_frame = annotator.result()

        if pose_landmarks is not None:
            # Get landmarks.
            frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
            
            pose_landmarks = pose_landmarks.cpu().numpy().squeeze(0)
            assert pose_landmarks.shape == (17, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

            # Classify the pose on the current frame.
            pose_classification = pose_classifier(pose_landmarks)
            print(pose_classification)

            # Smooth classification using EMA.
            pose_classification_filtered = pose_classification_filter(pose_classification)
            print(pose_classification_filtered)

            # Count repetitions.
            repetitions_count = repetition_counter(pose_classification_filtered)
        else:
            # No pose => no classification on current frame.
            pose_classification = None

            # Still add empty classification to the filter to maintaing correct smoothing for future frames.
            pose_classification_filtered = pose_classification_filter(dict())
            pose_classification_filtered = None

            # Don't update the counter presuming that person is 'frozen'. Just take the latest repetitions count.
            repetitions_count = repetition_counter.n_repeats

        # Draw classification plot and repetition counter.
        output_frame = pose_classification_visualizer(
            frame=output_frame,
            pose_classification=pose_classification,
            pose_classification_filtered=pose_classification_filtered,
            repetitions_count=repetitions_count)

        # Save the output frame.
        out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

        # Show intermediate frames of the video to track progress.
        if frame_idx % 10 == 0:
            show_image(output_frame)

        frame_idx += 1
        pbar.update()

# Close output video.
out_video.release()
