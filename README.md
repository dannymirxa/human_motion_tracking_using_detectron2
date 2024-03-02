# human_motion_tracking_using_detectron2

This is part of my master research project title "Human Motion Tracking using ". It contains Jupyter notebooks of training and testing the keypoint RCNN model using Detectron2. These notebooks where run in Google collab but it can be used locally with some modifications.

## Objectives:

1. To implement and evaluate a human motion tracking model using Detectron2 on a video dataset.
2. To identify and discuss the challenges and limitations of human motion detection using Detectron2 such as accuracy, robustness, scalability and privacy.

## Prerequisites:

1. COCO dataset https://cocodataset.org/#download.
2. Detectron2 https://github.com/facebookresearch/detectron2.

## V3 vs V3 modified training parameters

### V3 parameters:

1. MODEL.PIXEL_MEAN = [103.530, 116.280, 123.675]
2. INPUT.MIN_SIZE_TRAIN = (1280, 720)
3. INPUT.MAX_SIZE_TRAIN = 1000
3. SOLVER.BASE_LR = 0.00025
4. SOLVER.MAX_ITER = 5000
5. MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 32
6. MODEL.RPN.NMS_THRESH = 0.9
7. MODEL.ROI_HEADS.NUM_CLASSES = 1
8. MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17
9. MODEL.KEYPOINT_ON = True
10. TEST.EVAL_PERIOD = 500

### V3 modified parameters:

1. SOLVER.MAX_ITER = 2500
2. SOLVER.GAMMA = 0.1
3. SOLVER.WARMUP_FACTOR = 1/1000
4. SOLVER.WARMUP_ITERS = 1000
5. MODEL.BACKBONE.FREEZE_AT = 30

## Reference

Paper: https://arxiv.org/abs/1703.06870