import numpy as np
import os
import pickle
import subprocess
from PIL import Image

print("Downloading the CIFAR-10 data file.")
cifarFile = "cifar-10-python.tar.gz"
if not os.path.exists(cifarFile):
    fileUrl = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    subprocess.call("wget " + fileUrl, shell=True)

print("Extracting the file.")
extractDir = os.path.abspath("cifar-10-batches-py")
if not os.path.exists(extractDir):
    subprocess.call("tar -xvzf " + cifarFile, shell=True)

print("Reading the data.")
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict
data_batch_1 = unpickle(os.path.join(extractDir, "data_batch_1"))
data_batch_2 = unpickle(os.path.join(extractDir, "data_batch_2"))
data_batch_3 = unpickle(os.path.join(extractDir, "data_batch_3"))
data_batch_4 = unpickle(os.path.join(extractDir, "data_batch_4"))
data_batch_5 = unpickle(os.path.join(extractDir, "data_batch_5"))

data_batches = [data_batch_1, data_batch_2,
                data_batch_3, data_batch_4, data_batch_5]
test_batch   = unpickle(os.path.join(extractDir, "test_batch"  ))
batches_meta = unpickle(os.path.join(extractDir, "batches.meta"))

print("Making directories to store images.")
saveDir = os.path.abspath("cifar-images")
if not os.path.exists(saveDir):
    os.makedirs(saveDir)
for i in range(10):
    os.makedirs(os.path.join(saveDir, str(i)))
    os.makedirs(os.path.join(saveDir, 'test', str(i)))

print("Saving training files to corresponding directories.")
def getImage(datum, img):
    dim = 32
    for i in range(dim):
        for j in range(dim):
            i32 = i*32
            r, g, b = datum[i32+j], datum[i32+j+1024], datum[i32+j+2048]
            img[i][j] = [r, g, b]

def convertBatch(batch, saveDir, isTraining=True):
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    labels    = batch[b"labels"]
    data      = batch[b"data"]
    filenames = batch[b"filenames"]
    for i in range(len(labels)):
        label    = labels[i]
        datum    = data[i]
        filename = filenames[i]
        getImage(datum, img)
        pilImage = Image.fromarray(img)
        if isTraining:
            pilImage.save(os.path.join(saveDir, str(label), filename.decode('ascii')))
        else:
            pilImage.save(os.path.join(saveDir, "test", str(label), filename.decode('ascii')))

for batch in data_batches:
    convertBatch(batch, saveDir)

print("Saving test files to a corresponding directory.")
convertBatch(batch, saveDir, isTraining=False)

print("Finished!")

