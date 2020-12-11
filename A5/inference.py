import cv2
import time
import json
import onnx
import onnxruntime
import numpy as np


def input_format_for_resnet_50(img_path):
    img = cv2.imread(img_path).astype(np.float32)
    # convert BGR to RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # resize image to 224x224x3
    img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_AREA)
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

def get_inference_time(batch=False):
    N = 30
    model_dir = '../../data/A5/models/'
    model_name = 'resnet50_v1.onnx'
    img_file = 'ostrich.jpg'

    model = model_dir + model_name
    img_path = model_dir + img_file

    session = onnxruntime.InferenceSession(model)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    data = input_format_for_resnet_50(img_path)
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
    t = get_inference_time()
    print(f'Inference time for 30 runs: {round(t*1000), 2} ms')