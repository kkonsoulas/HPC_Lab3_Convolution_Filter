all:
	nvcc -O4 Convolution2D.cu -o conv

measure:
	./run.sh
