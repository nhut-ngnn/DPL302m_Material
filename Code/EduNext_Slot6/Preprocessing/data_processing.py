import shutil
import random 
import os
import time
from PIL import Image
import matplotlib.pyplot as plt
from cv2 import imread

    
def extract_corrupt_img_files(dir_path):
    """
    Gets corrupted image files within a directory
    
        Arguments:
            dir_path: a string representing the path for the target directory

        Returns:
            a list of the paths for the corrupted image files

    """ 
    i = 0
    beg = time.time()
    corrupted = []
    for filename in os.listdir(dir_path):
        i +=1
        if (i % 50 == 0):
            print(i, end =" ")
        if (i % 1000 == 0):
            print()
        try:
            img = Image.open(dir_path + '/' + filename)
        except:
            corrupted.append(filename)
            continue

    end = time.time()
    print()
    print('*' * 50) 
    print("\nTASK FINISHED IN " + str(end - beg) + " seconds ")
    print("{} corrupted files found in {}".format(len(corrupted), dir_path))
    print()
    print('*' * 50) 
    return corrupted

def copy_clean(src = '', dest ='', ignore = []):
    """
    Copies all the files from the source directory to the destination directory, ignoring the files specified in the ignore list.
    
    Parameters:
    src (str): The path of the source directory.
    dest (str): The path of the destination directory.
    ignore (list): A list of file names to ignore.
    
    Returns:
    None
    """
    beg = time.time()
    print("Copying file from " + src + " to " + dest)
    i = 0
    j = 0
    for filename in (os.listdir(src)):
        i += 1
        if filename not in ignore:
            shutil.copy(src + '/' + filename, dest + '/' + filename)
            j+=1
        if (i % 100 == 0):
            print(i, end = " ")
        if (i % 1000 == 0):
            print()
        
    end = time.time()
    print()
    print(j)
    print("Copying {} files finished in {} seconds ".format(len(os.listdir(dest)),int(end - beg)))
    

def display_imgs_from_path(path='', rows = 1, cols = 1):
    """
    Displays random rows * cols images from a directory
        
    Arguments:
        path: a string representing the path for the directory with the images to displat 
        rows: an integer representing the number of rows in the plots figure
        cols: an integer representing the number of columns in the plots figure

    Returns:
        None

    """
    fig = plt.figure(figsize=(8, 5))

    for i , img_name in enumerate(random.sample(os.listdir(path), rows * cols)):
        img = imread(path + '/' + img_name)
        fig.add_subplot(rows, cols, i+1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(img_name[:8])