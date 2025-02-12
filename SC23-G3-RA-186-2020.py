import sys
import numpy as np
import cv2 
import matplotlib
import matplotlib.pyplot as plt
import collections
import math
import pandas as pd
from scipy import ndimage
from sympy import sympify
matplotlib.rcParams['figure.figsize'] = 16,12
from tensorflow import keras
from sklearn.cluster import KMeans
import os
from sklearn.metrics import hamming_loss

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if filename.endswith(".jpg"):  
            img = os.path.join(folder, filename)
            if img is not None:
                images.append(img)
    return images

def load_image(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def image_gray(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def image_bin(image_gs):
    height, width = image_gs.shape[0:2]
    image_binary = np.ndarray((height, width), dtype=np.uint8)
    image_bin = cv2.adaptiveThreshold(image_gs, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 65, 5)
    return image_bin

def invert(image):
    return 255-image

def display_image(image, color=False):
    if color:
        plt.imshow(image)
    else:
        plt.imshow(image, 'gray')

def dilate(image):
    kernel = np.ones((3, 3)) 
    return cv2.dilate(image, kernel, iterations=1)

def erode(image):
    kernel = np.ones((3, 3)) 
    return cv2.erode(image, kernel, iterations=1)

def resize_region(region):
    return cv2.resize(region, (28, 28), interpolation=cv2.INTER_NEAREST)

def merge_contours(contours, threshold):
    merged_rectangles = []  
    skip_indices = set()    

    for i, cnt1 in enumerate(contours):
        if i in skip_indices:
            continue

        x1, y1, w1, h1 = cv2.boundingRect(cnt1)

        for j, cnt2 in enumerate(contours[i+1:], start=i+1):
            if j in skip_indices:
                continue

            x2, y2, w2, h2 = cv2.boundingRect(cnt2)

            if w1 + w2 < 30:
                
                merged_x = min(x1, x2)
                merged_y = min(y1, y2)
                merged_w = max(x1 + w1, x2 + w2) - merged_x
                merged_h = max(y1 + h1, y2 + h2) - merged_y
                
                merged_rectangles.append([merged_x, merged_y, merged_w, merged_h])
                skip_indices.add(i)
                skip_indices.add(j)

  
    for i, cnt in enumerate(contours):
        if i not in skip_indices:
            x, y, w, h = cv2.boundingRect(cnt)
            merged_rectangles.append([x, y, w, h])

    skip_indices.clear()
    final_rectangles = []  

    for i, rect1 in enumerate(merged_rectangles):
        if i in skip_indices:
            continue

        x1, y1, w1, h1 = rect1

        for j, rect2 in enumerate(merged_rectangles[i+1:], start=i+1):
            if j in skip_indices:
                continue

            x2, y2, w2, h2 = rect2

            if abs(y1 - y2) < threshold and (x2 < x1 + w1/2 < x2 + w2):
                merged_x = min(x1, x2)
                merged_y = min(y1, y2)
                merged_w = max(x1 + w1, x2 + w2) - merged_x
                merged_h = max(y1 + h1, y2 + h2) - merged_y
                final_rectangles.append([merged_x, merged_y, merged_w, merged_h])
                skip_indices.add(i)
                skip_indices.add(j)
                break

        if i not in skip_indices:
            final_rectangles.append(rect1)

    return final_rectangles


def select_roi(image_orig, image_bin):
    contours, _ = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []

   
    heights = [cv2.boundingRect(contour)[3] for contour in contours]
    threshold = np.mean(heights)
   
    merged_rectangles = merge_contours(contours, threshold)

    for rect in merged_rectangles:
        x, y, w, h = rect
        region = image_bin[y:y+h+1, x:x+w+1]
        regions_array.append([resize_region(region), (x, y, w, h)])
        cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    sorted_regions = [region[0] for region in regions_array]
    sorted_rectangles = [region[1] for region in regions_array]
    region_distances = []

    for index in range(len(sorted_rectangles) - 1):
        current = sorted_rectangles[index]
        next_rect = sorted_rectangles[index + 1]
        distance = next_rect[0] - (current[0] + current[2])
        region_distances.append(distance)

    return image_orig, sorted_regions, region_distances
    

def scale_to_range(image):
    return image/255

def matrix_to_vector(image):
    return image.flatten()

def prepare_for_ann(regions):
    ready_for_ann = []
    for region in regions:
        scale = scale_to_range(region)
        resized = resize_region(scale)
        vector = matrix_to_vector(resized)
        flattened = vector.flatten() 
        ready_for_ann.append(flattened)
    return ready_for_ann

def convert_output(alphabet):
    nn_outputs = []
    for index in range(len(alphabet)):
        output = np.zeros(len(alphabet))
        output[index] = 1
        nn_outputs.append(output)
    return np.array(nn_outputs)


def create_ann(output_size):
    ann = Sequential()
    ann.add(Dense(128, input_dim=784, activation='sigmoid'))
    ann.add(Dense(output_size, activation='sigmoid'))
    return ann

def train_ann(ann, X_train, y_train, epochs):
    X_train = np.array(X_train, np.float32) 
    y_train = np.array(y_train, np.float32) 
    
    sgd = SGD(learning_rate=0.01, momentum=0.9)
    ann.compile(loss='mean_squared_error', optimizer=sgd)
    ann.fit(X_train, y_train, epochs=epochs, batch_size=1, verbose=0, shuffle=False)
    return ann

def winner(output):
    return max(enumerate(output), key=lambda x: x[1])[0]

def display_result(outputs, alphabet, region_distances, threshold):
    result = alphabet[winner(outputs[0])]
    for idx, output in enumerate(outputs[1:, :]):
        if region_distances[idx] > threshold: 
            result += ' '
        result += alphabet[winner(output)]
    return result

def are_regions_similar(region1, region2, threshold=0.9):
    if region1.shape != region2.shape:
        return False

    similarity = np.sum(region1 == region2) / region1.size
    return similarity >= threshold


def select_roi_unique(image_orig, image_bin):
    contours, _ = cv2.findContours(image_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions_array = []
    heights = [cv2.boundingRect(contour)[3] for contour in contours]
    threshold = np.mean(heights)
    merged_rectangles = merge_contours(contours, threshold)  
    unique_regions = []  

    for rect in merged_rectangles:
        x, y, w, h = rect
        region = image_bin[y:y+h+1, x:x+w+1]
        is_unique = True

        for unique_region in unique_regions:
            if are_regions_similar(region, unique_region[0]):
                is_unique = False
                break

        if is_unique:
            unique_regions.append([region, (x, y, w, h)])
            cv2.rectangle(image_orig, (x, y), (x + w, y + h), (0, 255, 0), 2)

    regions_array = [[resize_region(region), rect] for region, rect in unique_regions]
    regions_array = sorted(regions_array, key=lambda x: x[1][0])
    sorted_regions = [region[0] for region in regions_array]

    return image_orig, sorted_regions


def process_images(image_paths):
    unique_regions_global = []  
    for path in image_paths:

        image_color = load_image(path)
        image_color1 = image_color[175:350, 250:825]  
        img = image_bin(image_gray(image_color1))

        selected_regions, letters = select_roi_unique(image_color1.copy(), img)

        for region in letters:
            is_unique = True
            for unique_region in unique_regions_global:
                if are_regions_similar(region, unique_region):
                    is_unique = False
                    break
            if is_unique:
                unique_regions_global.append(region)

    return unique_regions_global

directory=sys.argv[1]
folder_path = directory + '/pictures'
csv_path = directory + '/res.csv'
image_paths = load_images_from_folder(folder_path)
unique_regions = process_images(image_paths)

alphabet = ['с','у','ч','а','т','ь', 'х','р','о','ш','и','й','в','ъ','е','з','д','к','э','н','ц','ф','л','б','п','ю','щ','я','г','ё','ж']

inputs = prepare_for_ann(unique_regions)
outputs = convert_output(alphabet)
ann = create_ann(output_size=len(alphabet))
ann = train_ann(ann, inputs, outputs, epochs=1000)

predictions = []
for filename in os.listdir(folder_path):
        if filename.endswith('.jpg'):
            image_path = os.path.join(folder_path, filename)

            words = pd.read_csv(csv_path)[pd.read_csv(csv_path)['file'] == filename]['text'].values[0]
           
           
            test_color = load_image(image_path)
            test_color1= test_color[175:350, 250:825]
            test = image_bin(image_gray(test_color1))
            selected_test, test_letters, distances= select_roi(test_color1.copy(), test)

            distances = np.array(distances).reshape(len(distances), 1)

            test_inputs =  prepare_for_ann(test_letters)
            result = ann.predict(np.array(test_inputs, np.float32))
            threshold = 15
            res=display_result(result, alphabet, distances, threshold)
            predictions.append(res)
            print(f"{filename}-{words}-{res}")
            
            
true_values = pd.read_csv(csv_path, usecols=['text'])['text'].values

max_len = max(max(len(pred) for pred in predictions), max(len(true) for true in true_values))
predictions_padded = [pred.ljust(max_len) for pred in predictions]
true_values_padded = [true.ljust(max_len) for true in true_values]


average_hamming_loss = hamming_loss(true_values_padded, predictions_padded)

print(average_hamming_loss)
