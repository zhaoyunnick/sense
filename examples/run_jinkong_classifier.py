#!/usr/bin/env python
"""
Real time detection of 6 hand gesture events. Compared to `run_action_recognition`, the models used
in this script were trained to trigger the correct class only for a short period of time right after
the hand gesture occurred. This behavior policy makes it easier to quickly trigger multiple hand
gestures in a row.

Usage:
  run_gesture_control.py [--camera_id=CAMERA_ID]
                         [--path_in=FILENAME]
                         [--path_out=FILENAME]
                         [--title=TITLE]
                         [--model_name=NAME]
                         [--model_version=VERSION]
                         [--use_gpu]
  run_gesture_control.py (-h | --help)

Options:
  --path_in=FILENAME         Video file to stream from
  --path_out=FILENAME        Video file to stream to
  --title=TITLE              This adds a title to the window display
  --model_name=NAME          Name of the model to be used.
  --model_version=VERSION    Version of the model to be used.
  --use_gpu                  Whether to run inference on the GPU or not.
"""
from typing import Callable
from typing import Optional

from docopt import docopt

import sense.display
from sense.controller import Controller
from sense.downstream_tasks.nn_utils import LogisticRegression
from sense.downstream_tasks.nn_utils import Pipe
from sense.downstream_tasks.postprocess import AggregatedPostProcessors
from sense.downstream_tasks.postprocess import EventCounter
from sense.downstream_tasks.postprocess import PostprocessClassificationOutput
from sense.loading import get_relevant_weights
from sense.loading import build_backbone_network
from sense.loading import ModelConfig


SUPPORTED_MODEL_CONFIGURATIONS = [
    ModelConfig('StridedInflatedEfficientNet', 'pro', ['jk_gesture_control']),
]

LABEL2INT = {
    "rotation-ccw": 0,
    "rotation-cw": 1,
    "shrink": 2,
    "start": 3,
    "swipe-down": 4,
    "swipe-left": 5,
    "swipe-right": 6,
    "swipe-up": 7,
    "swipe-up-inv": 8,
    "zoom": 9
}
INT2LABEL = {value: key for key, value in LABEL2INT.items()}

ENABLE_LAB = [
    "rotation-ccw",
    "rotation-cw",
    "shrink",
    "start",
    "swipe-down",
    "swipe-left",
    "swipe-right",
    "swipe-up",
    "swipe-up-inv",
    "zoom"
]
LAB_THRESHOLDS = {key: 0.8 if key in ENABLE_LAB else 1. for key in LABEL2INT}

def run_jinkong(model_name: str,
                        model_version: str,
                        title: Optional[str] = None,
                        display_fn: Optional[Callable] = None,
                        **kwargs):
    """
    :param model_name:
        Model from backbone (StridedInflatedEfficientNet or StridedInflatedMobileNetV2).
    :param model_version:
        Model version (pro or lite)
    :param title:
        Title of the image frame on display.
    :param display_fn:
        Optional function to further process displayed image
    """
    # Load weights
    selected_config, weights = get_relevant_weights(
        SUPPORTED_MODEL_CONFIGURATIONS,
        model_name,
        model_version
    )

    # Load backbone network
    backbone_network = build_backbone_network(selected_config, weights['backbone'],
                                              weights_finetuned=weights['jk_gesture_control'])

    # Create a logistic regression classifier
    gesture_classifier = LogisticRegression(num_in=backbone_network.feature_dim,
                                            num_out=len(INT2LABEL))
    gesture_classifier.load_state_dict(weights['jk_gesture_control'])
    gesture_classifier.eval()

    # Concatenate backbone network and logistic regression
    net = Pipe(backbone_network, gesture_classifier)

    postprocessor = [
        PostprocessClassificationOutput(INT2LABEL, smoothing=1),
        AggregatedPostProcessors(
            post_processors=[
                EventCounter(key, LABEL2INT[key], LAB_THRESHOLDS[key]) for key in ENABLE_LAB
            ],
            out_key='counting',
        ),
    ]

    border_size_top = 0
    border_size_right = 500

    display_ops = [
        sense.display.DisplayFPS(expected_camera_fps=net.fps,
                                 expected_inference_fps=net.fps / net.step_size),
        sense.display.DisplayClassnameOverlay(thresholds=LAB_THRESHOLDS,
                                              duration=1,
                                              border_size_top=border_size_top if not title else border_size_top + 50,
                                              border_size_right=border_size_right),
        sense.display.DisplayPredictionBarGraph(ENABLE_LAB,
                                                LAB_THRESHOLDS,
                                                x_offset=900,
                                                y_offset=100,
                                                display_counts=True)
    ]
    display_results = sense.display.DisplayResults(title=title,
                                                   display_ops=display_ops,
                                                   border_size_top=border_size_top,
                                                   border_size_right=border_size_right,
                                                   display_fn=display_fn)

    # Run live inference
    controller = Controller(
        neural_network=net,
        post_processors=postprocessor,
        results_display=display_results,
        callbacks=[],
        **kwargs
    )
    controller.run_inference()


if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)

    run_jinkong(
        camera_id=int(args['--camera_id'] or 0),
        path_in=args['--path_in'] or None,
        path_out=args['--path_out'] or None,
        title=args['--title'] or None,
        model_name=args['--model_name'] or None,
        model_version=args['--model_version'] or None,
        use_gpu=args['--use_gpu'],
    )