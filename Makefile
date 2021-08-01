outputFile = finalProject
machineFile = mf
numProcs = 3

build:
	mpicxx -fopenmp -c main.c genetics.c genetics.h
	nvcc -I./inc -c geneCuda.cu -o geneCuda.o
	mpicxx -fopenmp -o $(outputFile)  main.o genetics.o geneCuda.o /usr/local/cuda-9.1/lib64/libcudart_static.a -ldl -lrt

clean:
	rm -f *.o *.h.gch ./$(outputFile) output.txt

run:
	mpiexec -np $(numProcs) ./$(outputFile)

runOnMultiple:
	mpiexec -np $(numProcs) -machinefile  $(machineFile)  -map-by  node  ./$(outputFile)

