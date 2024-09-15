import os
import shutil
import time
import random 
from utils.utils import original_validation_dir, validation_dir, test_dir, train_dir

def reset_directory(dir_path):
    if len(os.listdir(dir_path)) == 0 :
        print(dir_path + " is already empty")
        return

    beg = time.time()
    print("resetting "+ dir_path)
    shutil.rmtree(dir_path)

    os.makedirs(dir_path)
    print(dir_path + " is now empty")
    print("timing : " + str(time.time() - beg))
    
def test_validation_split(data_path, test_split):

    data_size = len(os.listdir(data_path))
    test_size = int(test_split * data_size)
    
    test_sample = []
    validation_sample = []
    
    test_sample = random.sample(os.listdir(data_path), test_size)
    
    remaining_data = set(os.listdir(data_path)) - set(test_sample)
    validation_sample = list(remaining_data)
    
    print('test size ' + str(len(test_sample)))
    print('validation size ' + str(len(validation_sample)))
    
    return test_sample, validation_sample
def split_data_to_dir(class_str = '',src_path ='', dest_path ='',samples ={}):

    src_path = src_path + '/' + class_str
    dest_path = dest_path + '/' + class_str
    beg = time.time()
    print(" Sending test samples to  " + dest_path)
    i = 0 
    for filename in samples[class_str]:
        shutil.copy(src_path + '/' + filename, dest_path + '/'+ filename)
        i+=1
        if (i % 25 == 0 ):
            print(i, end = " ")
        if (i % 500 == 0):
            print()

    endt = time.time()
    print("nb of test samples for {} is {}".format(class_str, str(i)))
    print("Sending {} test samples complete in {} seconds ".format(str(i),str(endt - beg)))

original_validation_dir = original_validation_dir()
validation_dir = validation_dir()
test_dir = test_dir()

dog_test_sample, dog_validation_sample = test_validation_split(data_path=original_validation_dir+'/dog', test_split=0.5)
cat_test_sample, cat_validation_sample = test_validation_split(data_path=original_validation_dir+'/cat', test_split=0.5)

test_samples = { 'cat': cat_test_sample,
                 'dog': dog_test_sample,
                }

validation_samples = { 'cat': cat_validation_sample,
                       'dog': dog_validation_sample,
                      }

reset_directory(test_dir +"/cat")
split_data_to_dir(class_str = "cat", src_path = original_validation_dir, dest_path = test_dir, samples = test_samples)
print(len(os.listdir(test_dir +"/cat")))

reset_directory(validation_dir +"/cat")
split_data_to_dir(class_str = "cat", src_path = original_validation_dir, dest_path = validation_dir, samples = validation_samples)
print(len(os.listdir(validation_dir +"/cat")))

reset_directory(test_dir +"/dog")
split_data_to_dir(class_str = "dog", src_path = original_validation_dir, dest_path = test_dir, samples = test_samples)
print(len(os.listdir(test_dir +"/dog")))

reset_directory(validation_dir +"/dog")
split_data_to_dir(class_str = "dog", src_path = original_validation_dir, dest_path = validation_dir, samples = validation_samples)
print(len(os.listdir(validation_dir +"/dog")))