#This will output vector of objects in the image using tfLite object detector
#https://www.tensorflow.org/lite/inference_with_metadata/task_library/object_detector#run_inference_in_python

#be sure to first use pip install tflite-support
import tflite_support as ts
#from tflite_support import task

#from tflite_support.task import vision
#from tflite_support.task import core
#from tflite_support.task import processor

# Initialization
base_options = ts.task.core.BaseOptions(file_name=model_path)
detection_options = ts.task.processor.DetectionOptions(max_results=2)
options = ts.taskvision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
detector = ts.task.vision.ObjectDetector.create_from_options(options)

# Alternatively, you can create an object detector in the following manner:
# detector = vision.ObjectDetector.create_from_file(model_path)

# Run inference
image = ts.task.vision.TensorImage.create_from_file(image_path)
detection_result = detector.detect(image)