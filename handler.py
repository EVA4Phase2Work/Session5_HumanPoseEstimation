try:
    import unzip_requirements
except ImportError:
    pass

from PIL import Image
import sys, time
import numpy as np
import boto3
import os
import io
import json
import base64
import cv2
import numpy as np
import math
import re
import os
import copy
import numpy as np
import onnxruntime

from PIL import Image


from requests_toolbelt.multipart import decoder
print("Import End...")

S3_BUCKET = os.environ['S3_BUCKET'] if 'S3_BUCKET' in os.environ else 'mdpbucket'
#MODEL_PATH = '/tmp/simple_pose_estimation.quantized.onnx'
MODEL_PATH = 'predictor/simple_pose_estimation.quantized.onnx'

print('Calling boto3.s3...')
s3 = boto3.client('s3')
print('Called boto3.s3...')

try:
    if os.path.isfile(MODEL_PATH) != True:
        print('ModelPath deos not exists...')
        #obj = s3.get_object(Bucket=S3_BUCKET, Key=MODEL_PATH)
        #print('Creating ByteStream')
        #bytestream = io.BytesIO(obj['Body'].read())
        #print("Loading Model")
        #newfile=open(MODEL_PATH,'wb')
        #newfile.write(bytestream)
        #print("Model Loaded...")
        
except Exception as e:
    print(repr(e))
    raise(e)
print('model. is ready..')

POSE_PAIRS = [[9, 8],[8, 7],[7, 6],[6, 2],[2, 1],[1, 0],[6, 3],[3, 4],[4, 5],[7, 12],[12, 11],[11, 10],[7, 13],[13, 14],[14, 15]]

from operator import itemgetter
get_keypoints = lambda pose_layers: map(itemgetter(1, 3), [cv2.minMaxLoc(pose_layer) for pose_layer in pose_layers])

JOINTS = ['0 - r ankle', '1 - r knee', '2 - r hip', '3 - l hip', '4 - l knee', '5 - l ankle', '6 - pelvis', '7 - thorax', '8 - upper neck', '9 - head top', '10 - r wrist', '11 - r elbow', '12 - r shoulder', '13 - l shoulder', '14 - l elbow', '15 - l wrist']
JOINTS = [re.sub(r'[0-9]+|-', '', joint).strip().replace(' ', '-') for joint in JOINTS]

def transform_image(image):
    train_mean = [0.485, 0.456, 0.406]
    train_std = [0.229, 0.224, 0.225]
    print(" Inside transform_image")
    resized_image = image.resize((256, 256)) 
    print(" resized_image: done")
    resized_image_np = np.asarray(resized_image)/255.0
    print(" resized_image_np: done: shape: ", resized_image_np.shape)
    normalize = lambda x: ((x - train_mean) / train_std).astype('float32')
    norm_img =  normalize(resized_image_np)
    print(" normalize: done")
    norm_img_exp_dim = np.expand_dims(norm_img, axis=0)
    print(" norm_img_exp_dim: done")
    return np.transpose(norm_img_exp_dim, [0, 3, 1, 2 ])



# get the human image with poses overlayued
def get_pose_estimation_image(img_pil, img_cv):
    # Detect faces in the image
    print("calling get_pose_estimation_image")
    tr_img = transform_image(img_pil)
    print("transform_image: Done")
    print(" get_pose_estimation_image: transpose_image called")
    ort_session = onnxruntime.InferenceSession(MODEL_PATH)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: tr_img}
    ort_outs = ort_session.run(None, ort_inputs)
    ort_outs = np.array(ort_outs[0][0]) # to get 16x64x64 output
    new_ort_outs = 255* ort_outs
    return get_human_with_pose_estimation(img_cv, new_ort_outs)
       

def get_human_with_pose_estimation(img, ort_outs):
    print("calling get_human_with_pose_estimation")
    THRESHOLD = 0.8
    OUT_HEIGHT = 64
    OUT_WIDTH = 64
    OUT_SHAPE = (OUT_HEIGHT, OUT_WIDTH)
    #image_p = cv2.imread(IMAGE_FILE)
    image_p = img
    pose_layers = copy.deepcopy(ort_outs)
    key_points = list(get_keypoints(pose_layers=pose_layers))
    is_joint_plotted = [False for i in range(len(JOINTS))]
    for pose_pair in POSE_PAIRS:
        from_j, to_j = pose_pair
        from_thr, (from_x_j, from_y_j) = key_points[from_j]
        to_thr, (to_x_j, to_y_j) = key_points[to_j]

        IMG_HEIGHT, IMG_WIDTH, _ = image_p.shape

        from_x_j, to_x_j = from_x_j * IMG_WIDTH / OUT_SHAPE[0], to_x_j * IMG_WIDTH / OUT_SHAPE[0]
        from_y_j, to_y_j = from_y_j * IMG_HEIGHT / OUT_SHAPE[1], to_y_j * IMG_HEIGHT / OUT_SHAPE[1]

        from_x_j, to_x_j = int(from_x_j), int(to_x_j)
        from_y_j, to_y_j = int(from_y_j), int(to_y_j)

        if from_thr > THRESHOLD and not is_joint_plotted[from_j]:
            # this is a joint
            cv2.ellipse(image_p, (from_x_j, from_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            is_joint_plotted[from_j] = True

        if to_thr > THRESHOLD and not is_joint_plotted[to_j]:
            # this is a joint
            cv2.ellipse(image_p, (to_x_j, to_y_j), (4, 4), 0, 0, 360, (255, 255, 255), cv2.FILLED)
            is_joint_plotted[to_j] = True

        if from_thr > THRESHOLD and to_thr > THRESHOLD:
            # this is a joint connection, plot a line
            cv2.line(image_p, (from_x_j, from_y_j), (to_x_j, to_y_j), (255, 74, 0), 3)
        
    print("End: get_human_with_pose_estimation")
    return Image.fromarray(cv2.cvtColor(image_p, cv2.COLOR_RGB2BGR))

def img_to_base64(img):
    #img = Image.fromarray(img, 'RGB') 
    buffer = io.BytesIO()
    img.save(buffer,format="JPEG")
    myimage = buffer.getvalue()                     
    img_str = f"data:image/jpeg;base64,{base64.b64encode(myimage).decode()}"
    return img_str

def get_pose_estimation(event, context):
    try:
        print('Body Pose Estimation')
        content_type_header = event['headers']['content-type']
        print("content loaded")
        body = base64.b64decode(event['body'])
        print('BODY LOADED')

        picture = decoder.MultipartDecoder(body, content_type_header).parts[0]
        im_arr = np.frombuffer(picture.content, dtype=np.uint8)
        print('im_arr shape:', im_arr.shape)
        img_cv = cv2.imdecode(im_arr, flags=cv2.IMREAD_COLOR)
        img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        print(" img_pil generated ")
        print('Image received')
        output_img=get_pose_estimation_image(img_pil, img_cv)
        print("output_img created")
        filename = picture.headers[b'Content-Disposition'].decode().split(';')[1].split('=')[1]
        if len(filename) < 4:
            filename = picture.headers[b'Content-Disposition'].decode().split(';')[2].split('=')[1]
        
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            "body": json.dumps({'alignedFaceImg': img_to_base64(output_img)})
        }
    except Exception as e:
        print(repr(e))
        return {
            "statusCode": 500,
            "headers": {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Credentials': True
            },
            'body': json.dumps({'error': repr(e)})
        }

