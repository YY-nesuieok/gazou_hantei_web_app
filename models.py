from django.db import models

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from PIL import Image
import io, base64

graph = tf.get_default_graph()
class Photo(models.Model):
    image = models.ImageField(upload_to='photos')
    
    IMAGE_SIZE = 224 #picture size
    MODEL_FILE_PATH = './carbike/ml_models/vgg16_transfer.h5' #model file
    
    # parameter initialize
    classes = ["car","motorbike"]
    num_classes = len(classes)
    
    # get img file from comand line
    def predict(self):
        model = None
        global graph
        with graph.as_default():
            model = load_model(self.MODEL_FILE_PATH)
            
            img_data = self.image.read()
            img_bin = io.BytesIO(img_data) #データをメモリ上に保持してファイルのようにアクセス
            
            image = Image.open(img_bin)
            image = image.convert("RGB") #make it rgb order
            image = image.resize((self.IMAGE_SIZE,self.IMAGE_SIZE)) #just in case there is an other size
            data = np.asarray(image)/255.0
            X = []#list
            X.append(data)#list
            X = np.array(X)#今回一つやからやらんでもよか
            
            result = model.predict([X])[0] # 1つめのデータの結果を返す。
            predicted = result.argmax()
            percentage = int(result[predicted]*100)
            
            #print(self.classes[predicted], percentage)
            return self.classes[predicted], percentage
    

    def image_src(self):
        with self.image.open() as img:
            base64_img = base64.b64encode(img.read()).decode()

            return 'data:' +img.file.content_type + ';base64,' +base64_img
