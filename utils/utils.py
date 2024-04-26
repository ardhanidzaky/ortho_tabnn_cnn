import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import torchvision.transforms as T

class PrepareDataset:
    def __init__(self, image_path, subtask, data):
        self.image_path = image_path
        self.subtask = subtask
        self.dataframe, self.encoding_dict = self._get_dataframe(data, self.subtask)
        self.X, self.y = self._get_x_and_y()

    def _get_dataframe(self, data, subtask):
        encoder = OrdinalEncoder()
        unique_values = data[subtask].unique().reshape(-1, 1)
        encoder.fit(unique_values)

        before_val = data[subtask]
        data[subtask] = encoder.transform(data[subtask].values.reshape(-1, 1))
        encoding_dict = {original_class: encoded_value for original_class, encoded_value in zip(data[subtask], before_val)}
        encoding_dict = {v: k for k, v in encoding_dict.items()}
        
        return data, encoding_dict

    def _get_x_and_y(self):
        X = self.dataframe[self.image_path]
        y = self.dataframe[self.subtask]

        return X, y

    def get_train_test_split(self):
        X_train, X_val, y_train, y_val = train_test_split(
            self.X, self.y
            , stratify=self.y
            , test_size=0.2
            , random_state=42
        )

        return pd.concat([X_train, y_train], axis=1), pd.concat([X_val, y_val], axis=1)

def crop_face(image_path, width, height, ckp='artifacts/haarcascade_frontalface_default.xml'):
    image = cv2.imread(image_path)
    image_rb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Search for faces on the image.
    face_cascade = cv2.CascadeClassifier(ckp)
    faces = face_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5)
    x, y, w, h = faces[0]

    if width != height:
        exp_ratio = 3 / 4
        h = int(w / exp_ratio)

        # Adjust y, as a pre-caution if it 
        # being cropped below the forehead.
        y -= int((image.shape[0] / height) * 35)
        
        # Add padding for the height, as a pre-caution
        # if it being cropped below the forehead.
        if y + h > image.shape[0]:
            minus_y = y + h - image.shape[0]
            y -= minus_y

        image_cropped = image_rb[y:y+h, x:x+w]
        image_cropped_resized = cv2.resize(image_cropped, (width, height))
        
        resized_rgb = image_cropped_resized
        resized_gray = cv2.cvtColor(resized_rgb, cv2.COLOR_BGR2GRAY)
    else:
        image_cropped = image_rb[y:y+h, x:x+w]
        image_cropped_resized = cv2.resize(image_cropped, (width, height))
        
        resized_rgb = image_cropped_resized
        resized_gray = cv2.cvtColor(resized_rgb, cv2.COLOR_BGR2GRAY)

    return resized_rgb, resized_gray

def data_transform(desired_size):
    data_transform = T.Compose([
        T.Resize(desired_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Use the ImageNet mean and std
    ])

    return data_transform