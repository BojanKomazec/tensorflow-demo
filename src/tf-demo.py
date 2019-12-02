from keras.diabetes import diabetesDemo
from keras.mnist import mnistDemo
from keras.sequential import sequentialNNDemo
from misc.install import testTensorFlowInstallation
from misc.comp_graph import computationalGraphDemo
from opencv.webcam_obj_detection import webcamObjectDetectionDemo

def main():
	# computationalGraphDemo()
	# diabetesDemo()
	# mnistDemo()
	# sequentialNNDemo()
	# testTensorFlowInstallation()
	webcamObjectDetectionDemo()

if __name__ == "__main__":
	main()
