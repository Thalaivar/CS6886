import cv2
import time
import json
import onnx
import onnxruntime
import numpy as np

MODEL_DIR = '../../data/A5/models/'
IMG_DIR = './'
MOBILENET = {   
        'input_name': 'input:0',
        'output_name': 'MobilenetV1/Predictions/Reshape_1:0',
        'input_size': (224, 224),
        'img_file': 'test.jpg',
        'model_name': 'mobilenet_v1_1.0_224.onnx'
    }

RESNET = {
        'input_name': 'input_tensor:0',
        'output_name': 'ArgMax:0',
        'input_size': (224, 224),
        'img_file': 'test.jpg',
        'model_name': 'resnet50_v1.onnx'
    }

SSD_MOBILENET = {
        'input_name': 'image_tensor:0',
        'output_name': 'num_detections:0',
        'input_size': (300, 300),
        'img_file': 'test.jpg',
        'model_name': 'ssd_mobilenet_v1_coco_2018_01_28.onnx'
    }

SSD_RESNET = {
        'input_name': 'image',
        'output_name': 'bboxes',
        'input_size': (1200, 1200),
        'img_file': 'test_large.jpg',
        'model_name': 'resnet34-ssd1200.onnx'
    }

def format_input_image(img_path, resize, model_name):
    img = cv2.imread(img_path).astype(np.float32)
    # convert BGR to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resize image to 224x224x3
    img = cv2.resize(img, dsize=resize, interpolation=cv2.INTER_AREA)

    if model_name == 'ssd_mobilenet_v1_coco_2018_01_28.onnx':
        img = img[np.newaxis, :]
        return img.astype(np.uint8)
    
    # normalize image
    mean = [123.68, 116.779, 103.939]
    stddev = [58.393, 57.12, 57.375]
    for i in range(3):
        img[:,:,i] -= mean[i]
        img[:,:,i] /= stddev[i]

    img -= img.min()
    img /= img.max()    
    img = np.transpose(img, axes=(2, 0, 1))
    img = img[np.newaxis, :]

    img = json.dumps({'data': img.tolist()})
    img = np.array(json.loads(img)['data']).astype('float32')
    
    return img

def get_inference_time(model_name, input_name, output_name, img_file, input_size, batch=False):
    N = 30

    model = MODEL_DIR + model_name
    img_file = IMG_DIR + img_file
    session = onnxruntime.InferenceSession(model)
    data = format_input_image(img_file, input_size, model_name)
    
    if batch:
        data = np.array([data for _ in range(N)]).squeeze()
        start_time = time.clock()
        session.run([output_name], {input_name: data})
        t_elapsed = time.clock() - start_time
    else:
        start_time = time.clock()
        for _ in range(N):
            session.run([output_name], {input_name: data})
        t_elapsed = time.clock() - start_time
    
    return t_elapsed/N

if __name__ == "__main__":
    resnet_50_t = get_inference_time(**RESNET)
    mobilenet_v1_t = get_inference_time(**MOBILENET)
    ss_mobilenet_300x300_t = get_inference_time(**SSD_MOBILENET)
    resnet_34_ssd1200_t = get_inference_time(**SSD_RESNET)

    print('\npython inference.py\nInference time for 30 runs:')
    print('     Image Classification:')
    print(f'        ResNet50 (v1.5): {round(resnet_50_t*1000, 2)} ms')
    print(f'        MobileNet (v1): {round(mobilenet_v1_t*1000, 2)} ms')
    print(f'    Object Detection:')
    print(f'        SSD-MobileNet 300x300: {round(ss_mobilenet_300x300_t*1000, 2)} ms')
    print(f'        SSD-Resnet34 1200x1200: {round(resnet_34_ssd1200_t*1000, 2)} ms')