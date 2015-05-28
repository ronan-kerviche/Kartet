# Basic Program

ifeq ($(TGPU),)
	TGPU = 35
#$(warning Setting default platform code : $(TGPU))
endif

EXEC_DEVICE	= kartetBasicTestDevice
EXEC_HOST	= kartetBasicTestHost
FILES		= main.cu
KARTET_OPTIONS	= -D KARTET_USE_ATLAS
# Use ATLAS (requires -llapack -lf77blas -lcblas -latlas) : -D KARTET_USE_ATLAS
# Use CBLAS only (requires -lcblas, ignored if KARTET_USE_ATLAS) : -D KARTET_USE_CBLAS 
# Use FFTW (v3, requires -lfftw3 -lfftw3f) : -D KARTET_USE_FFTW
# Change the default location : -D KARTET_DEFAULT_LOCATION=HostSide
#
# Note : 
# You can also use the multithreaded versions of ATLAS by replacing with : -lptlapack -lptf77blas -lptcblas -latlas	
# Make sure that Atlas was compile for the number of core you are targeting.
# See http://math-atlas.sourceforge.net/faq.html#tnum

OBJECTS		= $(patsubst %.cu,%.o,$(filter %.cu,$(FILES))) $(patsubst %.cpp,%.o,$(filter %.cpp,$(FILES))) $(patsubst %.c,%.o,$(filter %.c,$(FILES)))
DEVICE_OBJECTS	= $(patsubst %.o,%.device.o,$(OBJECTS))
HOST_OBJECTS	= $(patsubst %.o,%.host.o,$(OBJECTS))
GCC_INCLUDES	= -I../../include/
CUDA_INCLUDES	= -I/usr/local/cuda/include $(GCC_INCLUDES)
GCC_CCFLAGS	= -m64 -O3
CUDA_CCFLAGS 	= -arch=compute_$(TGPU) -code=sm_$(TGPU) -ftz=false -prec-div=true -prec-sqrt=true -Xcompiler $(GCC_CCFLAGS)
GCC_CCLDFLAGS	= -lm	$(subst KARTET_USE_ATLAS, -llapack -lf77blas -lcblas -latlas, $(findstring KARTET_USE_ATLAS,$(KARTET_OPTIONS))) \
			$(subst KARTET_USE_CBLAS, -lcblas, $(findstring KARTET_USE_CBLAS,$(KARTET_OPTIONS))) \
			$(subst KARTET_USE_FFTW, -lfftw3 -lfftw3f, $(findstring KARTET_USE_FFTW,$(KARTET_OPTIONS)))
CUDA_CCLDFLAGS	= -L/usr/local/cuda/lib64 -lcufft -lcublas -lcurand $(GCC_CCLDFLAGS)

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
	@nvcc -o $@ -c $^ $(CUDA_INCLUDES) $(KARTET_OPTIONS) $(CUDA_CCFLAGS)

%.device.o : %.cpp
	@echo '[DEVICE] Compiling $^ ...'	
	@g++ -o $@ -c $^ $(GCC_INCLUDES) $(KARTET_OPTIONS) $(GCC_CCFLAGS)

%.device.o : %.c
	@echo '[DEVICE] Compiling $^ ...'	
	@g++ -o $@ -c $^ $(GCC_INCLUDES) $(KARTET_OPTIONS) $(GCC_CCFLAGS)

%.host.o : %.cu
	@echo '[ HOST ] Compiling $^ ...'
	@ln -si $^ $(patsubst %.cu,%.cpp,$^)
	@g++ -o $@ -c $(patsubst %.cu,%.cpp,$^) $(GCC_INCLUDES) $(KARTET_OPTIONS) $(GCC_CCFLAGS)
	@rm $(patsubst %.cu,%.cpp,$^)

%.host.o : %.cpp
	@echo '[ HOST ] Compiling $^ ...'
	@g++ -o $@ -c $^ $(GCC_INCLUDES) $(KARTET_OPTIONS) $(GCC_CCFLAGS)

%.host.o : %.c
	@echo '[ HOST ] Compiling $^ ...'
	@g++ -o $@ -c $^ $(GCC_INCLUDES) $(KARTET_OPTIONS) $(GCC_CCFLAGS)

clean:
	@rm -f $(DEVICE_OBJECTS) $(HOST_OBJECTS)

mrproper: clean
	@rm -f $(EXEC_DEVICE) $(EXEC_HOST)

