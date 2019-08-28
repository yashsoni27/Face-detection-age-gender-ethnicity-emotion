import os
import numpy as np
import json
from PIL import Image
import requests
import cv2
from tqdm import tqdm
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.utils import shuffle
from keras.preprocessing import image as img
from keras.models import load_model

import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

input ('\nPress Enter...')

os.chdir('C:\\Users\\Yash\\Desktop\\Yash\\Python\\face-recognition\\dataset')

input('Directory changed...')

with open('face_recognition.json','r') as infile:
	json_block = []

	for line in infile:
		json_block.append(line)

		if line.startswith('}'):
			json_dict = json.loads(''.join(json_block))
			print(json_dict)
			json_block = []

input('Json file loaded...\n')

length = len(json_block)

for i in tqdm(range(length+1)):
    try:
        js = json.loads(json_block[i].replace("'", "\""))
        image = Image.open(requests.get(js['content'], stream=True).raw)
        image = np.asarray(image)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        for d in js['annotation']:
            width = d['imageWidth']
            height = d['imageHeight']
            x1 = int(width*d['points'][0]['x'])
            y1 = int(height*d['points'][0]['y'])
            x2 = int(width*d['points'][1]['x'])
            y2 = int(height*d['points'][1]['y'])
            roi = gray[y1:y2,x1:x2]
            s = d['label'][0]
            for k in range(1,len(d['label'])):
                s = s+" "+ d['label'][k]
            cv2.imwrite(s+'.jpg',roi)
    except:
        continue

