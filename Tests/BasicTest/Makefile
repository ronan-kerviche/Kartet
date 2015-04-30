# Basic Program

# Default plateform level, change with :
# make TGPU=<platform code>
ifeq ($(TGPU),)
	TGPU = 35
endif

EXEC		= kartetBasicTest
INCLUDES	= -I../../include/ -I/usr/local/cuda/include
NVCCFLAGS	= -arch=compute_$(TGPU) -code=sm_$(TGPU) -ftz=false -prec-div=true -prec-sqrt=true -O3 -Xcompiler -m64
NVCCLDFLAGS	= -L/usr/local/cuda/lib64 -lcufft -lcublas -lcurand -lcblas # You can add -latlas to link against atlas

all : main.o
	@echo 'Linking (platform code $(TGPU))...'
	@nvcc -o $(EXEC) $^ $(INCLUDES) $(NVCCLDFLAGS)

main.o : main.cu
	@echo 'Compiling $^ ...'
	@nvcc -o $@ -c $^ $(INCLUDES) $(NVCCFLAGS)

clean:
	@rm -f main.o 

mrproper: clean
	@rm -f $(EXEC)

