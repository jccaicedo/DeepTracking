
# coding: utf-8

# In[1]:

import logging
import matplotlib.pyplot as PLT
import numpy as NP
import os
import pandas as PD
import sys
sys.path.append("/home/ubuntu/tracking/localization-agent/tracking/")
sys.path.append('/usr/local/lib/python2.7/site-packages')
import tracking.util.data.LogProcessor as LogProcessor
import tracking.util.data.Preprocess as Preprocess
import tracking.util.data.VotTool as VotTool
from keras.layers import Dense, Dropout, Flatten as FlattenK, Input, InputLayer, Merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.recurrent import LSTM, GRU
from keras.models import Sequential
from keras.optimizers import RMSprop, SGD
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
from PIL import ImageDraw
from tracking.model.keras.Beeper import Beeper
from tracking.model.keras.Cnn import Cnn
from tracking.model.keras.Inverter import Inverter
from tracking.model.keras.Rnn import Rnn
from tracking.model.keras.SquareAttention import SquareAttention
from tracking.model.keras.GaussianAttention import GaussianAttention
from tracking.model.keras.SpatialTransformer import SpatialTransformer
from tracking.model.keras.Tracker import Tracker
from tracking.model.keras.Transformer import Transformer
from tracking.model.keras.Regressor import Regressor
from tracking.model.keras.VggCnn import VggCnn
from tracking.model.core.DatasetTrainer import DatasetTrainer
from tracking.model.core.GeneratorTrainer import GeneratorTrainer
from tracking.model.core.OnlineTrainer import OnlineTrainer
from tracking.model.core.Validator import Validator
from tracking.sequence.Generator import Generator
#from tracking.sequence.LaplaceTM import LaplaceTM
from tracking.sequence.TransformerGenerator import TransformerGenerator
from tracking.sequence.Sequence import Sequence
from tracking.util.data.AroundSampler import AroundSampler
from tracking.util.data.CentroidDistance import CentroidDistance
from tracking.util.data.Cropper import Cropper
from tracking.util.data.CropperProcessor import CropperProcessor
from tracking.util.data.GeneralProcessor import GeneralProcessor
from tracking.util.data.Overlap import Overlap
from tracking.util.data.PositionSampler import PositionSampler
from tracking.util.data.Sampler import Sampler
from tracking.util.data.SequenceProcessor import SequenceProcessor
from tracking.util.data.TranslationSampler import TranslationSampler
from tracking.util.data.VggProcessor import VggProcessor
from tracking.util.model.CentroidPM import CentroidPM
from tracking.util.model.CentroidHWPM import CentroidHWPM
from tracking.util.model.TwoCornersPM import TwoCornersPM

#get_ipython().magic('matplotlib inline')


# # SERVER VARIABLES

# ## HAL

# In[2]:

votSeqPath = "/home/ubuntu/tracking/data/vot-challenge/sequences/"
imageDir = "/home/ubuntu/tracking/data/mscoco/"
summaryPath = "/home/ubuntu/tracking/data/vot-challenge/CocoSummaries/cocoTrainSummaryCategAndSideGt100SmplsAllCorrected.pkl"
trajectoryModelPath = "/home/ubuntu/tracking/data/gmmDenseAbsoluteNormalizedOOT.pkl"
#scenePathTemplate = "../../home/fhdiaze/Data/GrayScene/"
scenePathTemplate = "images/train2014"
objectPathTemplate = "images/train2014"


# ## LISI4

# votSeqPath = "/home/datasets/vot-challenge/vot2014/"
# imageDir = "/home/datasets/MSCOCO/"
# summaryPath = "/home/fhdiaze/Data/Tracking/cocoTestSummaryCategAndSideGt100SmplsAllCorrected.pkl"
# trajectoryModelPath = "/home/fhdiaze/Data/Tracking/gmmDenseAbsoluteNormalizedOOT.pkl"
# #scenePathTemplate = "../../home/fhdiaze/Data/GrayScene/"
# scenePathTemplate = "train2014"
# objectPathTemplate = "train2014"

# # GENERAL VARIABLES

# In[3]:

trackerName = "Tracker36"
batchSize = 128


# # LOGGING CONFIGURATION

# In[4]:

# LOGGING VARIABLES
logFilePath = "/home/ubuntu/tracking/data/log/" + trackerName + ".log"
logFormat = "%(asctime)s:%(levelname)s:%(funcName)s:%(lineno)d:%(message)s"
logDateFormat = "%H:%M:%S"
logLevel = logging.INFO

logger = logging.getLogger()
fileHandler = logging.FileHandler(logFilePath, mode='w')
formatter = logging.Formatter(logFormat)
fileHandler.setFormatter(formatter)
logger.handlers = []
logger.addHandler(fileHandler)
logger.setLevel(logLevel)


# # COMPONENTS CONFIGURATION

# In[5]:

# GENERATOR VARIABLES
frameDims = (3, 224, 224) # (channels, width, height)
genSeqLength = 2
trajectoryModelSpec = ['random', 'sine', 'stretched', 'gmm']
cameraTrajectoryModelSpec = ['random', 'sine', 'stretched', 'gmm']

# VALIDATOR VARIABLES
totalValSequences = 10
votValDataSetPath = "/home/ubuntu/tracking/data/votDataset10Fms50Seqs.pkl"


# In[6]:

generator = Generator(frameDims[1:3], objectPathTemplate, scenePathTemplate, genSeqLength, imageDir, summaryPath,
                      trajectoryModelPath, trajectoryModelSpec, cameraTrajectoryModelSpec)
#framesPath = os.path.join(imageDir, objectPathTemplate)
#lpTM = LaplaceTM(0.0, 1.0/5.0, 1.0, 1.0/15.0)
#generator = TransformerGenerator(frameDims, framesPath, genSeqLength, lpTM, summaryPath)
positionRepresentation = TwoCornersPM()
#positionRepresentation = CentroidHWPM()
#cropper = Cropper(frameDims, positionRepresentation, context, distortion, minSide, downsampleFactor)
processor = GeneralProcessor(frameDims[1:3], positionRepresentation)
#processor = VggProcessor(frameDims[1:3], positionRepresentation)
measure = Overlap()
#measure = CentroidDistance()
#valF, valP = generator.getBatch(totalValSequences)
#valF, valP = processor.preprocess(valF, valP)
valF, valP = VotTool.loadDataset(votValDataSetPath)
valF, valP = processor.preprocess(valF, valP)
validator = Validator([valF[:, 1:, ...]], valP, batchSize, measure)


# # TRACKER CONFIGURATION

# ## VARIABLES

# In[7]:

# ATT VARIABLES
alpha = 0.1
scale = 1.0
epsilon = 0.05

# CNN VARIABLES
convAct = "relu"
regAct = "tanh"
modelPath = "/home/ubuntu/tracking/vgg16_weights.h5"
layerKey = "fc7"

# RNN VARIABLES
targetDim = positionRepresentation.getTargetDim()
stateDim = 256

# TRACKER VARIABLES
lnr = 0.001
lnrdy = 0.1
momentum = 0.95
loss = "mae"
timeSize = 1

# CROPPER VARIABLES
context = 2.0
distortion = 0.0
minSide = 0.2
downsampleFactor = 1.0


# In[8]:

class Builder(object):
    
    def build(self, input, modules):
        out = modules["cropper"].getModel()([input[0], modules["beeper"].getModel()(input[1])])
        out = modules["cnn"].getModel()(out)
        out = modules["rnn"].getModel()(out)
        out = modules["regressor"].getModel()(out)

        return out

step = SequenceProcessor()
builder = Builder()

# ## TRACKER CONFIGURATION

# In[9]:

optimizer = RMSprop(lr=lnr)
#optimizer = SGD(lr=lnr, decay=lnrdy, momentum=momentum)

# INPUT CONFIGURATION
frameInp = Input(shape=(None, ) + frameDims)
positionInp = Input(shape=(None, 4))
thetaInp = Input(shape=(None, 3, 3))
inputs = [frameInp, positionInp]

# CROP CONFIGURATION
beeper = Beeper(positionInp, distortion, minSide, context)
cropper = SpatialTransformer([frameInp, thetaInp], downsampleFactor)

# ATTENTION CONFIGURATION
#att = SquareAttention(inputs, alpha, scale)
#att = SpatialTransformer(inputs, downsampleFactor=1)

# INVERTER CONFIGURATION
inverter = Inverter(thetaInp)

# TRANSFORMER CONFIGURATION
transformer = Transformer([positionInp, thetaInp])

# CNN CONFIGURATION
frame = Input(shape=frameDims)
conv0 = Convolution2D(16, 3, 3, subsample=(1, 1), activation=convAct)
conv1 = Convolution2D(32, 3, 3, subsample=(1, 1), activation=convAct)
conv2 = Convolution2D(64, 3, 3, subsample=(1, 1), activation=convAct)
conv3 = Convolution2D(128, 3, 3, subsample=(1, 1), activation=convAct)
conv4 = Convolution2D(256, 3, 3, subsample=(1, 1), activation=convAct)
conv5 = Convolution2D(512, 3, 3, subsample=(1, 1), activation=convAct)
mxp0 = MaxPooling2D((2,2), strides=(2,2))
mxp1 = MaxPooling2D((2,2), strides=(2,2))
mxp2 = MaxPooling2D((2,2), strides=(2,2))
mxp3 = MaxPooling2D((2,2), strides=(2,2))
mxp4 = MaxPooling2D((2,2), strides=(2,2))
flat = FlattenK()
cnnLays = [conv0, mxp0, conv1, mxp1, conv2, mxp2, conv3, mxp3, conv4, mxp4, conv5, flat]
cnn = Cnn(frame, cnnLays)

# MERGE
#merge = Merge(mode="concat", concat_axis=-1)
#merge = Rnn([Input(shape=cnn.getOutputShape()), Input(shape=cnn.getOutputShape())], [merge])

# RNN CONFIGURATION
rnnInputShape = cnn.getOutputShape()
print rnnInputShape
features = Input(shape=rnnInputShape)
rlay0 = LSTM(stateDim, return_sequences=True, input_shape=rnnInputShape)
rnnLays = [rlay0]
rnn = Rnn(features, rnnLays)

# REGRESSOR CONFIGURATION
regInputShape = rnn.getOutputShape()
print regInputShape
reg = Dense(targetDim, activation=regAct, input_dim=regInputShape[-1])
regressor = Regressor([reg])

# TRACKER CONFIGURATION
#modules = {"attention":att, "cnn":cnn, "rnn":rnn, "regressor":regressor}
modules = {"beeper":beeper, "cropper": cropper, "inverter":inverter, "cnn":cnn, "rnn":rnn, "regressor":regressor}
tracker = Tracker(inputs, modules, builder, optimizer, loss, step, timeSize)
tracker.build()


# # TRACKER OFFLINE TRAINING

# ## VARIABLES

# In[10]:

offEpochs = 1
offBatches = 3
trackerModelPath = "//home/ubuntu/tracking/data/Models/"+ trackerName + ".pkl"
votDataSetPaths = ["/home/ubuntu/tracking/data/Datasets/votDataset10Fms50Seqs.pkl", "/home/ubuntu/tracking/data/Datasets/votDataset10Fms100Seqs.pkl"]


# ## FUNCTIONS

# In[ ]:

def trainGenerator():
    while 1:
        frame, position = generator.getBatch(batchSize)
        frame, position = processor.preprocess(frame, position)
        cropPosition = NP.roll(position, 1, axis=1) # Shift the time
        cropPosition[:, 0, :] = cropPosition[:, 1, :] # First frame is ground truth
        #import cPickle as pickle
        #with open("/home/ubuntu/tracking/data/debug_batch.pkl", "wb") as out:
        #    pickle.dump([frame, cropPosition], out)
        
        yield [frame, cropPosition], position


# In[ ]:

#genTrainer = GeneratorTrainer(tracker, generator, processor, validator)
#Avoid maximum recursion limit exception when pickling by increasing limit from ~1000 by default
sys.setrecursionlimit(10000)
#genTrainer.train(7, 5, batchSize, lnr, lnrdy)
tracker.train(trainGenerator(), offEpochs, offBatches, batchSize, validator)
tracker.save(trackerModelPath)


# In[13]:

#tracker.load(trackerModelPath)
from keras.utils.visualize_util import plot as kplot
kplot(regressor.model.layer, to_file='/home/fhdiaze/model.png')


# # TRACKER TESTING

# ## VARIABLES

# In[14]:

fps = 15
outDir = "/home/fhdiaze/Data/Videos/"


# ## VOT TESTING

# In[15]:

tracker.setStateful(True, 1)


# In[16]:

# VOT VARIABLES
testVotSeqNames = ["ball", "car", "basketball", "david", "fish1", "motocross", "woman", "bolt"]
positionsFile = "groundtruth.txt"
extension = ".jpg"

seqs = VotTool.loadSequences(votSeqPath, testVotSeqNames, extension, positionsFile, frameDims[1:3])

def testGenerator(seqs):
    for name, frame, position in seqs:
        frame = NP.expand_dims(frame, axis=0)
        position = NP.expand_dims(position, axis=0)
        prepF, prepP = processor.preprocess(frame, position)
        
        # Generate name, input, label
        yield name, [prepF[:, 1:, ...]], prepP
    
measures = validator.test(tracker, testGenerator(seqs))
VotTool.plotResults(measures, measure.name)


# In[15]:




# ## LOADING TEST SEQUENCE

# In[18]:

testVotSeqName = "ball"
path = os.path.join(votSeqPath, testVotSeqName)
boxesPath = os.path.join(path, positionsFile)
testVotF, testVotP = VotTool.loadSequence(path, extension, positionsFile, frameDims[1:3])
testVotP = NP.expand_dims(testVotP, axis=0)
testVotF = NP.expand_dims(testVotF, axis=0)
testVotSeqLen = testVotP.shape[1]

# PREPROCESSING THE TEST SEQUENCE
prepTestVotF, prepTestVotP = processor.preprocess(testVotF, testVotP)
postTestVotF, postTestVotP = processor.postprocess(prepTestVotF, prepTestVotP)
print NP.sum(testVotP - postTestVotP)
print NP.sum(testVotF - postTestVotF)


# ## TRACKER ONLINE TRAINING

# In[22]:

import lasagne
# VARIABLES
onBatchSize = 2
onBatches = 10
transRange = 0.1
onLnr = 0.0000001

onlineTrainer = OnlineTrainer(tracker, processor, validator)
inLayer = lasagne.layers.InputLayer((onBatchSize, frameDims[0], frameDims[1], frameDims[2]))
paramsShape = (onBatchSize, 6)
transformer = lasagne.layers.TransformerLayer(inLayer, paramsShape, downsample_factor=1.0)

#sampler = Sampler(transformer, transRange)
#sampler = TranslationSampler(transformer, transRange)
#sampler = PositionSampler(transRange)
sampler = AroundSampler(transformer, transRange)

tracker.load(trackerModelPath)
tracker.reset()
#tracker.regressor.setTrainable(False)
#tracker.modules["rnn"].setTrainable(False)
#tracker.modules["cnn"].setTrainable(False)
#tracker.build()

onlineTrainer.fitFrame(sampler, onBatches, onBatchSize, onLnr, testVotF[0, 0, ...], testVotP[0, 0, ...])


# In[1]:

tracker.reset()
predTestVotP = tracker.forward([prepTestVotF], prepTestVotP[:, 0, :])
#predTestVotP = onlineTrainer.forwardOnline(sampler, prepTestVotF[:, ...], prepTestVotP[:, 0, :], 3, onSamples, onLnr)


# In[ ]:

postTestVotF, postPredTestVotP = processor.postprocess(prepTestVotF, predTestVotP)
postTestVotF, postTestVotP = processor.postprocess(prepTestVotF, prepTestVotP)
iou = measure.calculate(postTestVotP, postPredTestVotP)
iouMean = iou.mean()
print("iouFirstFrames =  " + str(iou[0, :5]))
PLT.plot(iou[0, ...])
PLT.suptitle("{}, {} {} = {}".format(testVotSeqName, measure.name, "Mean", iouMean))
PLT.xlabel("frame")
PLT.ylabel(measure.name)
PLT.show()


# ## GENERATING VOT VIDEO

# In[ ]:

gtPos = positionRepresentation.fromTwoCorners(testVotP)[0, ...].tolist()
predPos = positionRepresentation.fromTwoCorners(postPredTestVotP)[0, ...].tolist()
images = Preprocess.getFrames(testVotF[0, ...])
sequence = Sequence(images, positionRepresentation)
sequence.addBoxes(gtPos, "Red")
sequence.addBoxes(predPos, "Blue")
output = os.path.join(outDir, trackerName + testVotSeqName + ".mp4")
sequence.exportToVideo(fps, output)


# ## GENERATING ONLINE SAMPLES

# In[2]:

def generateSamples():
    sampledF, sampledP = sampler.generateSamples(testVotF[0, 100,...], testVotP[:, 100, :], onBatchSize)
    sampledF = NP.expand_dims(sampledF, axis=1)
    sampledP = NP.expand_dims(sampledP, axis=1)
    prepSampledF, prepSampledP = processor.preprocess(sampledF, sampledP)
    tracker.reset()
    predSampledP = tracker.forward(prepSampledF, prepSampledP[:, 0, :])
    postSampledF, postSampledP = processor.postprocess(prepSampledF, prepSampledP)
    postSampledF, postPredSampledP = processor.postprocess(prepSampledF, predSampledP)
    
    fig = PLT.figure(1, (30., 30.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(3, 3),  # creates 2x2 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                    )

    for i in range(onBatchSize):
        frame = postSampledF[i, 0, ...]
        frame = Image.fromarray(frame.astype(NP.uint8))
        drawCp = ImageDraw.Draw(frame)
        drawCp.rectangle(list(postSampledP[i, 0, :]), outline="red")
        drawCp.rectangle(list(postPredSampledP[i, 0, :]), outline="blue")
        grid[i].imshow(frame)

generateSamples()


# ## SYNTHETIC TESTING

# In[ ]:

testGenF, testGenP = generator.getBatch(batchSize)
prepTestGenF, prepTestGenP = processor.preprocess(testGenF, testGenP)
predtestGenP = tracker.forward(prepTestGenF, prepTestGenP[:, 0, :])
postTestGenF, postPredTestGenP = processor.postprocess(prepTestGenF, predtestGenP)

iou = measure.calculate(testGenP, postPredTestGenP)
iouMean = iou.mean()
logging.info("Test Overlap = %f", iouMean)
print("iouMean = " + str(iouMean))
print("iouFirstFrames =  " + str(iou[0, :5]))
PLT.plot(range(genSeqLength), iou[0, ...])
PLT.show()


# ## GENERATING SYNTHETIC VIDEO

# In[ ]:

images = Preprocess.getFrames(testGenF[0, ...])
sequence = Sequence(images, positionRepresentation)
gtPos = testGenP[0, ...].tolist()
predPos = postPredTestGenP[0, ...].tolist()

sequence.addBoxes(gtPos, "Red")
sequence.addBoxes(predPos, "Blue")
output = os.path.join(outDir, trackerName + "Synthetic.mp4")
sequence.exportToVideo(fps, output)


# # LOG PROCESSING

# In[ ]:

batchLosses, batchMeasures, epochMeasures = LogProcessor.getValues(logFilePath)
fig = plt.figure(1, (20., 10.))
plt.plot(batchMeasures, color="b")
plt.plot(batchLosses, color="r")
plt.plot(epochMeasures, color="g")
plt.show()


# In[ ]:



