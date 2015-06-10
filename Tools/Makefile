# GENERIC MAKEFILE TOOL FOR KARTET
# Options :
# KARTET_PATH		: Path to Kartet directory
# EXEC_DEVICE		: Name of the executable to generate for the device platform.
# EXEC_HOST		: Name of the executable to generate for the host platform.
# FILES			: Files to compile (*.cu, *.cpp)
# DEVICE_CODE		: Target device platform code (20, 35, ...)
# CC_EXTRA_FLAGS	: Extra flags for the compilers.
# CC_EXTRA_LINK		: Extra flags for the linkers.
# HSCC			: Host side compiler.
# NVCC			: Device side compiler.
# DEFAULT_TARGET	: Default compilation target amon deviceBinary and hostBinary.
# KARTET_OPTIONS	: Options for the library
#     -D KARTET_USE_ATLAS			Use ATLAS (requires -llapack -lf77blas -lcblas -latlas).
#     -D KARTET_USE_CBLAS			Use CBLAS only (requires -lcblas, ignored if KARTET_USE_ATLAS).
#     -D KARTET_USE_FFTW			Use FFTW (v3, requires -lfftw3 -lfftw3f).
#     -D KARTET_DEFAULT_LOCATION=newLocation	Change the default location (HostSide or DeviceSide).
#
#     Note : 
#     You can also use the multithreaded versions of ATLAS by replacing with : -lptlapack -lptf77blas -lptcblas -latlas	
#     Make sure that Atlas was compile for the number of core you are targeting.
#     See http://math-atlas.sourceforge.net/faq.html#tnum

# Example of client Makefile :
#export EXEC_DEVICE	= execNameDevice
#export EXEC_HOST	= execNameTestHost
#export FILES		= main.cu
#export KARTET_OPTIONS	= -D KARTET_USE_ATLAS
##export CC_EXTRA_LINK	= -shared # For a shared object/dynamic library
#all :
#% :
#	@$(MAKE) -s -f $(KARTET_PATH)/Tools/Makefile $@

# First, test :
ifeq ($(KARTET_PATH),)
#$(error Undefined variable KARTET_PATH)
	KARTET_PATH = /usr/include/
endif

ifeq ($(EXEC_DEVICE),)
$(error Undefined variable EXEC_DEVICE)
endif

ifeq ($(EXEC_HOST),)
$(error Undefined variable EXEC_HOST)
endif

ifeq ($(FILES),)
$(error Undefined variable FILES)
endif

ifeq ($(DEVICE_CODE),)
	DEVICE_CODE = 35
#$(info Setting default platform code : $(DEVICE_CODE).)
endif

ifeq ($(HSCC),)
	HSCC = g++
#$(info Using $(HSCC) as default host compiler.)
endif

ifeq ($(NVCC),)
	NVCC = nvcc
#$(info Using $(NVCC) as default device compiler.)
endif

ifeq ($(DEFAULT_TARGET),)
	DEFAULT_TARGET = deviceBinary hostBinary
endif

# Generate :
OBJECTS		= $(patsubst %.cu,%.o,$(filter %.cu,$(FILES))) $(patsubst %.cpp,%.o,$(filter %.cpp,$(FILES))) $(patsubst %.c,%.o,$(filter %.c,$(FILES)))
DEVICE_OBJECTS	= $(patsubst %.o,%.device.o,$(OBJECTS))
HOST_OBJECTS	= $(patsubst %.o,%.host.o,$(OBJECTS))
CC_INCLUDE	= -I$(KARTET_PATH)/include/
NVCC_INCLUDE	= -I/usr/local/cuda/include $(CC_INCLUDE)
CC_FLAGS	= -m64 -O3 -Wall -Wextra $(CC_EXTRA_FLAGS)
NVCC_FLAGS 	= -arch=compute_$(DEVICE_CODE) -code=sm_$(DEVICE_CODE) -ftz=false -prec-div=true -prec-sqrt=true -O3 -Xcompiler "\"$(CC_FLAGS)\""
CC_LINK		= -lm	$(subst KARTET_USE_ATLAS, -llapack -lf77blas -lcblas -latlas, $(findstring KARTET_USE_ATLAS,$(KARTET_OPTIONS))) \
			$(subst KARTET_USE_CBLAS, -lcblas, $(findstring KARTET_USE_CBLAS,$(KARTET_OPTIONS))) \
			$(subst KARTET_USE_FFTW, -lfftw3 -lfftw3f, $(findstring KARTET_USE_FFTW,$(KARTET_OPTIONS))) \
		  $(CC_EXTRA_LINK)	
NVCC_LINK	= -L/usr/local/cuda/lib64 -lcufft -lcublas -lcurand $(CC_LINK)

all : $(DEFAULT_TARGET)
	@echo '[ ALL  ] Done.'

deviceBinary : $(SOURCES) $(DEVICE_OBJECTS)
	@echo '[DEVICE] Linking (Platform code $(DEVICE_CODE))...'
	@$(NVCC) -o $(EXEC_DEVICE) $^ $(NVCC_FLAGS) $(NVCC_LINK)

hostBinary : $(SOURCES) $(HOST_OBJECTS)
	@echo '[ HOST ] Linking...'
	@$(HSCC) -o $(EXEC_HOST) $^ $(CC_FLAGS) $(CC_LINK)

%.device.o : %.cu
	@echo '[DEVICE] Compiling $^ ...'	
	@$(NVCC) -o $@ -c $^ $(NVCC_INCLUDE) $(KARTET_OPTIONS) $(NVCC_FLAGS)

%.device.o : %.cpp
	@echo '[DEVICE] Compiling $^ ...'	
	@$(HSCC) -o $@ -c $^ $(CC_INCLUDE) $(KARTET_OPTIONS) $(CC_FLAGS)

%.device.o : %.c
	@echo '[DEVICE] Compiling $^ ...'	
	@$(HSCC) -o $@ -c $^ $(CC_INCLUDE) $(KARTET_OPTIONS) $(CC_FLAGS)

%.host.o : %.cu
	@echo '[ HOST ] Compiling $^ ...'
	@ln -si $^ $(patsubst %.cu,%.cpp,$^)
	@$(HSCC) -o $@ -c $(patsubst %.cu,%.cpp,$^) $(CC_INCLUDE) $(KARTET_OPTIONS) $(CC_FLAGS)
	@rm $(patsubst %.cu,%.cpp,$^)

%.host.o : %.cpp
	@echo '[ HOST ] Compiling $^ ...'
	@$(HSCC) -o $@ -c $^ $(CC_INCLUDE) $(KARTET_OPTIONS) $(CC_FLAGS)

%.host.o : %.c
	@echo '[ HOST ] Compiling $^ ...'
	@$(HSCC) -o $@ -c $^ $(CC_INCLUDE) $(KARTET_OPTIONS) $(CC_FLAGS)

clean:
	@rm -f $(DEVICE_OBJECTS) $(HOST_OBJECTS)

mrproper: clean
	@rm -f $(EXEC_DEVICE) $(EXEC_HOST)

