import datetime
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_option_menu import option_menu
import multiprocessing as mp
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


parser = argparse.ArgumentParser()
parser.add_argument('--input-video-path', default="io/input/IMG_7685.mp4", type=str)
parser.add_argument('--class-name', default="ez_bar_preacher_curl_down", type=str)
parser.add_argument('--output-dir-path', default="io/output/", type=str)
args = parser.parse_args()


st.set_page_config(
    page_title="PhysioAI",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)

st.markdown("<style>  h1, h2 {padding-top: 0;} p{margin-bottom: 0} </style>", unsafe_allow_html=True)


with st.sidebar:
    selected = option_menu("PhysioAI", 
                           ["Home", "Routines", "Workout", "Explore", "My achievements", "History", "Measures", 'Settings', "Support"], 
                           icons=['house', 'calendar', "play", 'compass', 'award', 'clock', 'person', 'gear', 'question'], 
                           menu_icon="cast", default_index=2, )


col1, col2 = st.columns(2)
with col1:
    pose_image = st.empty()
with col2:
    st.title("EZ-Bar Preacher Curl")
    st.markdown("""---""")
    col3, col4 = st.columns(2)
    with col3:
        st.write("Repetition")
        counter = st.empty()
    with col4:
        st.write("Set")
        st.markdown("# <span style='font-size:100px;'>2</span> :gray[ / 3 ]", unsafe_allow_html=True)

    notification = st.empty()

    col5, col6 = st.columns(2)
    with col5:
        plot = st.empty()
    with col6:
        st.image("EZ-Bar Preacher Curl.jpg")
    


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
    enter_threshold=7,
    exit_threshold=3)

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

pose_classification_history = []
pose_classification_filtered_history = []
end = False

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
                annotator.kpts(k, (video_width, video_height), radius=7)

            output_frame = annotator.result()

        if not end:
            if pose_landmarks is not None:
                # Get landmarks.
                frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
                
                pose_landmarks = pose_landmarks.cpu().numpy().squeeze(0)
                assert pose_landmarks.shape == (17, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

                # Classify the pose on the current frame.
                pose_classification = pose_classifier(pose_landmarks)

                # Smooth classification using EMA.
                pose_classification_filtered = pose_classification_filter(pose_classification)

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
        
        pose_classification_history.append(pose_classification)
        pose_classification_filtered_history.append(pose_classification_filtered)

        fig = go.Figure(layout_xaxis_range=[0, 1021], layout_yaxis_range=[0, 10])
        fig.update_layout(
            height=320,
            legend=dict(
                x=0.85,
                y=1,
                traceorder="normal",
            ),
             margin= {
                'l': 0,
                'r': 0,
                'b': 0,
                't': 0,
                'pad': 0
        	},
                xaxis=dict(
        title=None,
        showgrid=False,
        visible= False,
    ), 
    yaxis=dict(
        title=None,
        showgrid=False,
        visible= False,

    ),
        )
        
        for classification_history in [
            # pose_classification_history,
            pose_classification_filtered_history]:
            y = []
            for classification in classification_history:
                if classification is None:
                    y.append(None)
                elif args.class_name in classification:
                    y.append(classification[args.class_name])
                else:
                    y.append(0)
            
            df = pd.DataFrame(dict(
                x = list(range(len(classification_history))),
                y = y
            ))

            fig.add_trace(go.Scatter(x=list(range(len(classification_history))), y=y,
                    mode='lines',
                    name='Filtered' if classification_history == pose_classification_filtered_history else 'Raw'))

        plot.plotly_chart(fig, use_container_width=True, config = {'displayModeBar': False})
            
        frame_idx += 1
        counter.markdown(f"# :blue[<span style='font-size:100px;'>{repetitions_count}</span>] :gray[ / 12 ]", unsafe_allow_html=True)
        pose_image.image(output_frame)

        if repetitions_count == 12:
            notification.success('## ✅ Good Job Sajjad!')
            end = True

# Close output video.
out_video.release()
