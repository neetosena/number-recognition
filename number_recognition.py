import os
import re
import shutil
import time
from os import path

import cv2
import numpy as np
from PIL import Image

number_string = ""

# -----------Function to save the files and crop them --Using os to read the file + pillow to edit the image----------------


def reduction_save(ext):
    direct = os.chdir(
        "/Users/user/Desktop/Neto/Python/Number Recognition/test/JPG")
    box = (160, 150, 760, 450)
    size_200_100 = (200, 100)

    for f in os.listdir('.'):
        if f.endswith(ext):
            im = Image.open(f)
            fn, fext = os.path.splitext(f)

            region = im.crop(box)
            region.thumbnail(size_200_100)
            # region.show()
            region.save(
                "/Users/user/Desktop/Neto/Python/Number Recognition/test/reduction/{}{}".format(fn, fext))


# --------------------------------------------------------------------------------

#  -----------Function to read and to identify the number then convert to String --Using OpenCV2----------------
def read_matriz():
    digits = cv2.imread(
        "/Users/user/Desktop/Neto/Python/new_digits.jpg", cv2.IMREAD_GRAYSCALE)
    rows = np.vsplit(digits, 2)
    cells = []
    for row in rows:
        row_cells = np.hsplit(row, 5)
        for cell in row_cells:
            cell = cell.flatten()
            cells.append(cell)

    cells = np.array(cells, dtype=np.float32)
    return cells


def arange():
    k = np.arange(10)
    cells_labels = np.repeat(k, 1)
    return cells_labels


def read_numbers(path_cv2):
    test_digits = cv2.imread(
        "/Users/user/Desktop/Neto/Python/Number Recognition/test/reduction/" + path_cv2, cv2.IMREAD_GRAYSCALE)

    test_digits = np.hsplit(test_digits, 2)
    test_cells = []
    for d in test_digits:
        d = d.flatten()
        test_cells.append(d)

    test_cells = np.array(test_cells, dtype=np.float32)
    return test_cells


def number_recognition(cells, cells_labels, test_cells):
    # knn
    knn = cv2.ml.KNearest_create()
    knn.train(cells, cv2.ml.ROW_SAMPLE, cells_labels)
    ret, result, neighbours, dist = knn.findNearest(test_cells, k=1)
    return result


def convert_number_recognition_to_string(result):
    out_arr = np.array_str(result)
    numbers = re.findall('[0-9]+', out_arr)
    number_string = ""
    for i in numbers:
        number_string = number_string + i
    number_string = number_string + ".jpg"
    return number_string


# --------------------------------------------------------------------------------

def execute():
    direct = os.chdir(
        "/Users/user/Desktop/Neto/Python/Number Recognition/test/reduction/")
    for f in os.listdir('.'):
        if f.endswith('.jpg'):
            numberRecog = number_recognition(
                read_matriz(), arange(), read_numbers(f))
            numbConvString = convert_number_recognition_to_string(numberRecog)

            direct2 = os.chdir(
                "/Users/user/Desktop/Neto/Python/Number Recognition/test/JPG")
            if path.exists(f):
                src = path.realpath(f)
                os.rename(f, numbConvString)
                print("worked")


def listdir_no_hidden(path):
    for f in os.listdir(path):
        if not f.startswith('.DS_Store'):

            return f


direct = os.chdir(
    "/Users/user/Desktop/Neto/Python/Number Recognition/test/reduction/")
print(os.getcwd())
for d in os.listdir("."):

    if d.endswith('.jpg'):

        print(d)

    elif path.exists(d):
        reduction_save('.jpg')
        print("created")


execute()
