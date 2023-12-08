# YOLOv8 Pose Classification

Pose classification and repetition counting with the k-NN algorithm

[Page](https://developers.google.com/ml-kit/vision/pose-detection/classifying-poses)


## Run

1. Use [notebook](Pose_classification_(extended).ipynb) to create and validate a training set for the k-NN classifier and save them as csv files.

2. Run classification on a video:

```commandline
python inference.py --input-video-path <path_to_input_video> --output-dir-path <path_to_output_dir> --class-name <name-of-exercise-class>
```
