#Test Program

# Default plateform level :
ifeq ($(TGPU),)
        #$(error GPU PLATFORM IS UNDEFINED. Use make TGPU=<platform code> where <platform code> is a two-digits code (e.g. 35))
	TGPU = 35
endif

EXEC		= kartetPerformanceTest
INCLUDES	= -I../../include/ -I/usr/local/cuda/include
NVCCFLAGS	= -arch=compute_$(TGPU) -code=sm_$(TGPU) -ftz=false -prec-div=true -prec-sqrt=true -O3 -Xcompiler -m64
NVCCLDFLAGS	= -L/usr/local/cuda/lib64 -lcufft -lcublas -lcurand

all : CuTimer.o main.o
	@echo 'Linking (platform code $(TGPU))...'
	@nvcc -o $(EXEC) $^ $(INCLUDES) $(NVCCLDFLAGS)

main.o : main.cu
	@echo 'Compiling $^ ...'
	@nvcc -o $@ -c $^ $(INCLUDES) $(NVCCFLAGS)

CuTimer.o : CuTimer.cu
	@echo 'Compiling $^ ...'
	@nvcc -o $@ -c $^ $(INCLUDES) $(NVCCFLAGS)

clean:
	@rm -f CuTimer.o main.o 

mrproper: clean
	@rm -f $(EXEC)

