M:=49152
N:=12288
K:=8192

all:


cuda: matrix.cu
	nvcc  matrix.cu -o matrix
	time ./matrix $(M) $(N) $(K)

cublas: cublas.cu
	nvcc  -lcublas cublas.cu  -o cublas
	time ./cublas $(M) $(N) $(K)
