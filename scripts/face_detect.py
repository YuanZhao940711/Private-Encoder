import os 
import cv2
import dlib

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
    
    print('\nThe percentage of {} detected faces is {:.2f}'.format(model, face_num/img_num))