# Basic Program

ifeq ($(TGPU),)
	TGPU = 35
#$(warning Setting default platform code : $(TGPU))
endif

EXEC_DEVICE	= kartetBasicTestDevice
EXEC_HOST	= kartetBasicTestHost
FILES		= main.cu
OBJECTS		= $(patsubst %.cu,%.o,$(filter %.cu,$(FILES))) $(patsubst %.cpp,%.o,$(filter %.cpp,$(FILES))) $(patsubst %.c,%.o,$(filter %.c,$(FILES)))
DEVICE_OBJECTS	= $(patsubst %.o,%.device.o,$(OBJECTS))
HOST_OBJECTS	= $(patsubst %.o,%.host.o,$(OBJECTS))
GCC_INCLUDES	= -I../../include/
CUDA_INCLUDES	= -I/usr/local/cuda/include $(GCC_INCLUDES)
GCC_CCFLAGS	= -m64 -O3
CUDA_CCFLAGS 	= -arch=compute_$(TGPU) -code=sm_$(TGPU) -ftz=false -prec-div=true -prec-sqrt=true -Xcompiler $(GCC_CCFLAGS)
GCC_CCLDFLAGS	= -llapack -lf77blas -lcblas -latlas 
CUDA_CCLDFLAGS	= -L/usr/local/cuda/lib64 -lcufft -lcublas -lcurand $(GCC_CCLDFLAGS)
	# You can also use the multithreaded versions : -lptlapack -lptf77blas -lptcblas -latlas	
	# Make sure that Atlas was compile for the number of core you are targeting.
	# See http://math-atlas.sourceforge.net/faq.html#tnum
KARTETOPTIONS	= -D KARTET_USE_ATLAS 
	# Change the default location : -D KARTET_DEFAULT_LOCATION=HostSide

all : deviceBinary hostBinary
	@echo '[ ALL  ] Done.'

deviceBinary : $(SOURCES) $(DEVICE_OBJECTS)
	@echo '[DEVICE] Linking (Platform code $(TGPU))...'
	@nvcc -o $(EXEC_DEVICE) $^ $(CUDA_CCFLAGS) $(CUDA_CCLDFLAGS)

hostBinary : $(SOURCES) $(HOST_OBJECTS)
	@echo '[ HOST ] Linking...'
	@g++ -o $(EXEC_HOST) $^ $(GCC_CCFLAGS) $(GCC_CCLDFLAGS)

%.device.o : %.cu
	@echo '[DEVICE] Compiling $^ ...'	
	@nvcc -o $@ -c $^ $(CUDA_INCLUDES) $(KARTETOPTIONS) $(CUDA_CCFLAGS)

%.device.o : %.cpp
	@echo '[DEVICE] Compiling $^ ...'	
	@g++ -o $@ -c $^ $(GCC_INCLUDES) $(KARTETOPTIONS) $(GCC_CCFLAGS)

%.device.o : %.c
	@echo '[DEVICE] Compiling $^ ...'	
	@g++ -o $@ -c $^ $(GCC_INCLUDES) $(KARTETOPTIONS) $(GCC_CCFLAGS)

%.host.o : %.cu
	@echo '[ HOST ] Compiling $^ ...'
	@ln -si $^ $(patsubst %.cu,%.cpp,$^)
	@g++ -o $@ -c $(patsubst %.cu,%.cpp,$^) $(GCC_INCLUDES) $(KARTETOPTIONS) $(GCC_CCFLAGS)
	@rm $(patsubst %.cu,%.cpp,$^)

%.host.o : %.cpp
	@echo '[ HOST ] Compiling $^ ...'
	@g++ -o $@ -c $^ $(GCC_INCLUDES) $(KARTETOPTIONS) $(GCC_CCFLAGS)

%.host.o : %.c
	@echo '[ HOST ] Compiling $^ ...'
	@g++ -o $@ -c $^ $(GCC_INCLUDES) $(KARTETOPTIONS) $(GCC_CCFLAGS)

clean:
	@rm -f $(DEVICE_OBJECTS) $(HOST_OBJECTS)

mrproper: clean
	@rm -f $(EXEC_DEVICE) $(EXEC_HOST)

