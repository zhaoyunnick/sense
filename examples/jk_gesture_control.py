#!/usr/bin/env python
"""
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

import json
import os.path
from docopt import docopt

import sense.display
from sense.loading import load_weights_from_resources
from sense.controller import Controller
from sense.downstream_tasks.nn_utils import LogisticRegression
from sense.downstream_tasks.nn_utils import Pipe
from sense.downstream_tasks.postprocess import PostprocessClassificationOutput
from sense.downstream_tasks.postprocess import AggregatedPostProcessors
from sense.loading import build_backbone_network
from sense.loading import update_backbone_weights
from sense.downstream_tasks.postprocess import EventCounter
from sense.loading import ModelConfig


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
INT2LAB = {value: key for key, value in LABEL2INT.items()}

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
LAB_THRESHOLDS = {key: 0.6 if key in ENABLE_LAB else 1. for key in LABEL2INT}


if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)

    camera_id = int(args['--camera_id'] or 0)
    path_in = args['--path_in'] or None
    path_out = args['--path_out'] or None
    title = args['--title'] or None
    use_gpu = args['--use_gpu']

    # load backbone network
    backbone_model_config = ModelConfig('StridedInflatedEfficientNet', 'pro', [])
    backbone_model_weight = backbone_model_config.load_weights()['backbone']

    # load custom classifier
    classifier = load_weights_from_resources('jk_gesture_control/best_classifier.checkpoint')

    # update original weight in case some intermittent layers have been finned
    backbone_network = build_backbone_network(backbone_model_config, backbone_model_weight,
                                              weights_finetuned=classifier)

    # Create backbone network
    backbone_network = build_backbone_network(backbone_model_config, backbone_model_weight)

    # load label file
    '''
    with open(os.path.join(classiffier, 'label2int.json')) as file:
        class2label = json.load(file)
    INT2LABEL = {value: key for key, value in class2label.item()}
    '''
    gesture_classifier = LogisticRegression(num_in=backbone_network.feature_dim,
                                            num_out=len(INT2LAB))
    gesture_classifier.load_state_dict(classifier)
    gesture_classifier.eval()

    # Concatenate backbone network and logistic regression
    net = Pipe(backbone_network, gesture_classifier)

    postprocessor = [
        PostprocessClassificationOutput(INT2LAB, smoothing=4),
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
                                                   )
    '''
    display_ops = [
        sense.display.DisplayFPS(expected_camera_fps=net.fps,
                                 expected_inference_fps=net.fps / net.step_size),
        sense.display.DisplayTopKClassificationOutputs(top_k=1, threshold=0.1),
    ]
    display_results = sense.display.DisplayResults(title=title, display_ops=display_ops)
    '''

    # Run live inference
    controller = Controller(
        neural_network=net,
        post_processors=postprocessor,
        results_display=display_results,
        callbacks=[],
        camera_id=camera_id,
        path_in=path_in,
        path_out=path_out,
        use_gpu=use_gpu
    )
    controller.run_inference()






