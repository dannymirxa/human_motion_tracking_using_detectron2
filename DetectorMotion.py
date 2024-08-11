from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances

import os
import cv2

class Detector:
    def __init__(self, model_type : str):
        register_coco_instances("keypoint_dataset_test", {}, r"/mnt/d/Master Data Science/Research Project 1/fiftyone/detections_keypoints_segmentations_test.json", r"/mnt/d/Master Data Science/Research Project 1/more_filtered_test/data")
        self.keypoint_names = [
            "nose", 
            "left_eye", "right_eye", 
            "left_ear", "right_ear", 
            "left_shoulder", "right_shoulder", 
            "left_elbow", "right_elbow", 
            "left_wrist", "right_wrist", 
            "left_hip", "right_hip", 
            "left_knee", "right_knee", 
            "left_ankle", "right_ankle" 
        ] 

        self.keypoint_flip_map = [
            ("left_eye", "right_eye"), 
            ("left_ear", "right_ear"), 
            ("left_shoulder", "right_shoulder"), 
            ("left_elbow", "right_elbow"), 
            ("left_wrist", "right_wrist"), 
            ("left_hip", "right_hip"), 
            ("left_knee", "right_knee"), 
            ("left_ankle", "right_ankle")
        ]      

        self.keypoint_connection_rules = [
            ['left_ankle', 'left_knee', (0, 0, 255)], ['left_knee', 'left_hip', (0, 0, 255)], ['right_ankle', 'right_knee', (0, 0, 255)],
            ['right_knee', 'right_hip', (0, 0, 255)], ['left_hip', 'right_hip', (0, 0, 255)], ['left_shoulder', 'left_hip', (0, 0, 255)],
            ['right_shoulder', 'right_hip', (0, 0, 255)], ['left_shoulder', 'right_shoulder', (0, 0, 255)], ['left_shoulder', 'left_elbow', (0, 0, 255)],
            ['right_shoulder', 'right_elbow', (0, 0, 255)], ['left_elbow', 'left_wrist', (0, 0, 255)], ['right_elbow', 'right_wrist', (0, 0, 255)],
            ['left_eye', 'right_eye', (0, 0, 255)], ['nose', 'left_eye', (0, 0, 255)], ['nose', 'right_eye', (0, 0, 255)], ['left_eye', 'left_ear', (0, 0, 255)],
            ['right_eye', 'right_ear', (0, 0, 255)], ['left_ear', 'left_shoulder', (0, 0, 255)], ['right_ear', 'right_shoulder', (0, 0, 255)]
        ]

        MetadataCatalog.get("keypoint_dataset_test").thing_classes = ["person"]
        MetadataCatalog.get("keypoint_dataset_test").keypoint_names = self.keypoint_names
        MetadataCatalog.get("keypoint_dataset_test").keypoint_flip_map = self.keypoint_flip_map
        MetadataCatalog.get("keypoint_dataset_test").keypoint_connection_rules = self.keypoint_connection_rules
        MetadataCatalog.get("keypoint_dataset_test").thing_dataset_id_to_contiguous_id = {1:0}
        MetadataCatalog.get("keypoint_dataset_test").evaluator_type="coco"

        self.keypoint_train_metadata = MetadataCatalog.get("keypoint_dataset_test")

        self.cfg = get_cfg()
        self.model_type = model_type

        if model_type == "OO":
            self.cfg.merge_from_file(model_zoo.get_config_file(r"COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(r"COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
        elif model_type == "IS":
            self.cfg.merge_from_file(model_zoo.get_config_file(r"COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(r"COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml")
        elif model_type == "KP":
            self.cfg.merge_from_file(model_zoo.get_config_file(r"COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml"))
            #self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(r"COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml")
            self.cfg.MODEL.WEIGHTS = r"D:\Master Data Science\Research Project 1\detectron2\model_final.pth"
            #self.cfg.MODEL.WEIGHTS = os.path.join(r"D:\Master Data Science\Research Project 1\Video Dataset\model_final.pth")
        elif model_type == "LVIS":
            self.cfg.merge_from_file(model_zoo.get_config_file(r"LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(r"LVISv0.5-InstanceSegmentation/mask_rcnn_R_101_FPN_1x.yaml")
        elif model_type == "PS":
            self.cfg.merge_from_file(model_zoo.get_config_file(r"COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(r"COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")

        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST= 0.8  # set threshold for this model
        self.cfg.MODEL.DEVICE= "cuda"

        self.predictor= DefaultPredictor(self.cfg)

    def onImage(self, imagePath : str):
  
        image = cv2.imread(imagePath)
        
        if self.model_type == "KP":
            predictions = self.predictor(image)
            """ v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode= ColorMode.IMAGE)
            output = v.draw_instance_predictions(predictions["instances"][predictions["instances"].pred_classes== 0].to("cpu")) """

            v = Visualizer(image[:, :, ::-1], metadata=self.keypoint_train_metadata, instance_mode= ColorMode.IMAGE)
            output = v.draw_instance_predictions(predictions["instances"].to("cpu"))

            #a = predictions["instances"][predictions["instances"].pred_classes== 0].pred_boxes.__iter__()
            #a = predictions["instances"][predictions["instances"].pred_classes== 0].pred_boxes
            #print(a.__next__().cpu().numpy())
        elif self.model_type == "PS":
            predictions, segmentInfo = self.predictor(image)["panoptic_seg"]
            v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
            output = v.draw_panoptic_seg_predictions(predictions.to("cpu"), segmentInfo)
            #print(predictions["instances"][predictions["instances"].pred_classes== 0].pred_boxes)
        else:
            predictions = self.predictor(image)
            v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode= ColorMode.IMAGE)
            #output = v.draw_instance_predictions(predictions["instances"][predictions["instances"].pred_classes== 0].to("cpu"))
            output = v.draw_instance_predictions(predictions["instances"].to("cpu"))
            #a = predictions["instances"][predictions["instances"].pred_classes== 0].pred_boxes.__iter__()
            #a = predictions["instances"][predictions["instances"].pred_classes== 0].pred_boxes
            #print(a.__next__().cpu().numpy())

        window_name = 'image'
        
        cv2.imshow(window_name, output.get_image()[:,:,::-1])
        
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()

    def onVideo(self, imagePath : str):
        cap = cv2.VideoCapture(imagePath)
        
        if (cap.isOpened()== False):
            print("Error opening video file")

        while(cap.isOpened()):
            ret,frame=cap.read()

            gray1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.GaussianBlur(gray1, (21, 21), 0)
            
            deltaframe=cv2.absdiff(gray1,gray2)
            threshold = cv2.threshold(deltaframe, 25, 255, cv2.THRESH_BINARY)[1]
            threshold = cv2.dilate(threshold,None)
            countour,heirarchy = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if self.model_type == "KP":
                predictions = self.predictor(frame)
                """ v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode= ColorMode.IMAGE)
                output = v.draw_instance_predictions(predictions["instances"][predictions["instances"].pred_classes== 0].to("cpu")) """
                v = Visualizer(frame[:, :, ::-1], metadata=self.keypoint_train_metadata, instance_mode= ColorMode.IMAGE)
                output = v.draw_instance_predictions(predictions["instances"].to("cpu"))
            elif self.model_type == "PS":
                predictions, segmentInfo = self.predictor(frame)["panoptic_seg"]
                v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                output = v.draw_panoptic_seg_predictions(predictions["instances"][predictions["instances"].pred_classes== 0].to("cpu"), segmentInfo)
                #print(predictions["instances"][predictions["instances"].pred_classes== 0].pred_boxes)
            else:
                predictions = self.predictor(frame)
                v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode= ColorMode.IMAGE)
                output = v.draw_instance_predictions(predictions["instances"][predictions["instances"].pred_classes== 0].to("cpu"))
            
            for i in countour:
                if cv2.contourArea(i) < 1000:
                    continue
            
            cv2.imshow('Frame', output.get_image()[:,:,::-1])
            
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        
        cv2.destroyAllWindows()
    
    def onCam(self):
        
        vid = cv2.VideoCapture(0)

        while(True):
        # Capture frame-by-frame
            ret, frame = vid.read()
            if self.model_type != "PS":
                predictions = self.predictor(frame)
                v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]), instance_mode= ColorMode.IMAGE)
                output = v.draw_instance_predictions(predictions["instances"][predictions["instances"].pred_classes== 0].to("cpu"))
                #print(predictions["instances"][predictions["instances"].pred_classes== 0].pred_boxes)
                
            else:
                predictions, segmentInfo = self.predictor(frame)["panoptic_seg"]
                v = Visualizer(frame[:, :, ::-1], MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]))
                output = v.draw_panoptic_seg_predictions(predictions["instances"][predictions["instances"].pred_classes== 0].to("cpu"), segmentInfo)
                #print(predictions["instances"][predictions["instances"].pred_classes== 0].pred_boxes)

            if ret == True:
            # Display the resulting frame
                cv2.imshow('Frame', output.get_image()[:,:,::-1])
                
            # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
        # When everything done, release
        # the video capture object
        
        # After the loop release the cap object
        vid.release()
        # Destroy all the windows
        cv2.destroyAllWindows()