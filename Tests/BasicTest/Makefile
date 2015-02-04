#Test Program

# Default plateform level :
ifeq ($(TGPU),)
        #$(error GPU PLATFORM IS UNDEFINED. Use make TGPU=<platform code> where <platform code> is a two-digits code (e.g. 35))
	TGPU = 35
endif

EXEC		= kartetBasicTest
INCLUDES	= -I../../include/ -I/usr/local/cuda/include
NVCCFLAGS	= -arch=compute_$(TGPU) -code=sm_$(TGPU) -ftz=false -prec-div=true -prec-sqrt=true -O3 -Xcompiler -m64
NVCCLDFLAGS	= -L/usr/local/cuda/lib64 -lcufft -lcublas -lcurand

all : main.o
	@echo 'Linking (platform code $(TGPU))...'
	@nvcc -o $(EXEC) $^ $(INCLUDES) $(NVCCLDFLAGS)

main.o : main.cu
	@echo 'Compiling main.cu ...'
	@nvcc -o $@ -c $^ $(INCLUDES) $(NVCCFLAGS)

clean:
	@rm -f main.o

mrproper: clean
	@rm -f $(EXEC)

