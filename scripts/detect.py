import os 
import cv2
import dlib

from tqdm import tqdm
from argparse import ArgumentParser

def parser():
    parser = ArgumentParser()

    parser.add_argument('--model',default='HOG')
    parser.add_argument('--input_dir', default='./image_datasets')

    return parser.parse_args()

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def dataset(image_dir):
    images_path = []

    for root, dirs, files in os.walk(image_dir):
        for fname in files:
            if is_image_file(fname):
                img_path = os.path.join(root, fname)
                images_path.append(img_path)

    images_path = sorted(images_path)
    
    print('Load {} images'.format(len(images_path)))

    return images_path

class face_detection:
    def __init__(self, model):
        if model == 'CNN':
            self.detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')
        elif model == 'HOG':
            self.detector = dlib.get_frontal_face_detector()
        else:
            print('Could load corresponded detection model!')
            exit()
    
    def detection(self, images_path):
        try:
            img_num = 0
            face_num = 0
            
            with tqdm(images_path, total=len(images_path)) as t:
                for img_path in t:
                    img = cv2.imread(img_path)
                    img_num += 1
                    
                    face = self.detector(img, 1)
                    face_num += len(face)
                return img_num, face_num
        
        except KeyboardInterrupt:
            t.close()
            raise

if __name__ == '__main__':
    args = parser()

    detector = face_detection(model=args.model)

    images_path = dataset(image_dir=args.input_dir)

    img_num, face_num = detector.detection(images_path=images_path)
    
    print('\nThe {}-based detected model finds {:.4f} percent faces \nfrom dataset: {}'.format(args.model, face_num/img_num, args.input_dir))
"""
img_path = './image_datasets/Privacy_Evaluation.png'
output_dir = './image_datasets'

img = cv2.imread(img_path)

cnn_detector = dlib.cnn_face_detection_model_v1('mmod_human_face_detector.dat')

faces_cnn = cnn_detector(img, 1)

print('{} faces are detected in this image'.format(len(faces_cnn)))

for face in faces_cnn:
  x = face.rect.left()
  y = face.rect.top()
  w = face.rect.right() - x
  h = face.rect.bottom() - y

  cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 2)
  
  cv2.imwrite(os.path.join(output_dir, 'cnn_face_detected.png'), img)
"""