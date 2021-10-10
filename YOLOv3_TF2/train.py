import numpy as np
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.utils import to_categorical

from utils import load_class_names, output_boxes, draw_outputs, resize_image
from yolov3 import YOLOv3Net


def train_model(x_train, y_train, model, nb_epochs, batch_size):

    l_rate = 1e-3 #What learning rate should we use?

    optimizer = tf.keras.optimizers.Adam(learning_rate=l_rate) # What optimizer should we use?

    model.compile(
        loss={"yolo_loss": lambda y_true, y_pred: y_pred}, 
        optimizer=optimizer, 
        metrics=[MeanIoU(num_classes=1)])
    print(model.summary()) 

    print("Start training")

    history = model.fit(
        x_train, y_train,
        batch_size=batch_size,
        epochs=nb_epochs,
        validation_split=0.1,
    )

    model.save('trained_models/test')

    return model

if __name__ == "__main__":
    x_train, y_train = #charger le dataset

    model_size = (416, 416, 3)
    num_classes = 1
    class_name = './data/massalia.names'
    max_output_size = 10
    max_output_size_per_class = 10
    iou_threshold = 0.5
    confidence_threshold = 0.5

    nb_epochs = 1
    batch_size= 32

    cfgfile = 'cfg/yolov3.cfg'

    model = YOLOv3Net(cfgfile='cfg/yolov3.cfg', model_size=nodel_size, num_classes=num_classes)

    train_model(x_train, y_train, model, nb_epochs, batch_size)