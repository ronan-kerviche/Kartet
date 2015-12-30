# GENERIC MAKEFILE TOOL FOR KARTET
# Options :
# KARTET_PATH		: Path to Kartet directory
# EXEC_DEVICE		: Name of the executable to generate for the device platform.
# EXEC_HOST		: Name of the executable to generate for the host platform.
# FILES			: Files to compile (*.cu, *.cpp)
# DEVICE_CODE		: Target device platform code (20, 35, ...)
# CC_EXTRA_FLAGS	: Extra flags for the compilers.
# CC_EXTRA_LIB		: Extra linrary flags for the linkers.
# HSCC			: Host side compiler.
# NVCC			: Device side compiler.
# KARTET_DEFAULT_TARGET	: Default compilation target amon deviceBinary and hostBinary.
# SOURCES_DIR		: Directory for the source files.
# OBJ_DIR		: Directory for the object files.
# CUDA_INCLUDE_PATH	: Path to CUDA headers directory.
# CUDA_LIB_PATH		: Path to CUDA libraries.
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
##export CC_EXTRA_LIB	= -shared # For a shared object/dynamic library
#all :
#% :
#	@$(MAKE) -s -f $(KARTET_PATH)/Tools/Makefile $@

# Note on NVidia Architecture :
# -arch=sm_XX is equivalent to -gencode=arch=compute_XX,code=sm_XX -gencode=arch=compute_XX,code=compute_XX
# NVCC generates intermediate PTX (VM specified by -arch) and closer SASS (-code) from the sources. 
# PTX can be transformed into SASS by the driver without having the CUDA SDK installed.

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

ifeq ($(KARTET_DEFAULT_TARGET),)
	KARTET_DEFAULT_TARGET = deviceBinary hostBinary
endif

ifeq ($(CUDA_INCLUDE_PATH),)
	CUDA_INCLUDE_PATH = /usr/local/cuda/include
endif

ifeq ($(CUDA_LIB_PATH),)
	CUDA_LIB_PATH = /usr/local/cuda/lib64
endif

# Generate :
FILES_LIST	= $(wildcard $(SOURCES_DIR)$(FILES))
OBJECTS		= $(patsubst %.cu,%.o,$(filter %.cu,$(FILES_LIST))) $(patsubst %.cpp,%.o,$(filter %.cpp,$(FILES_LIST))) $(patsubst %.c,%.o,$(filter %.c,$(FILES_LIST)))
DEVICE_OBJECTS	= $(patsubst %.o,%.device.o,$(OBJECTS))
HOST_OBJECTS	= $(patsubst %.o,%.host.o,$(OBJECTS))
CC_INCLUDE	= -I$(KARTET_PATH)/include/ $(subst KARTET_USE_ATLAS, -L/usr/include/atlas, $(findstring KARTET_USE_ATLAS,$(KARTET_OPTIONS)))
NVCC_INCLUDE	= -I$(CUDA_INCLUDE_PATH) $(CC_INCLUDE)
CC_FLAGS	= -fPIC -m64 -O3 -march=native -Wall -Wextra $(CC_EXTRA_FLAGS)
NVCC_FLAGS 	= -arch=sm_$(DEVICE_CODE) -ftz=false -prec-div=true -prec-sqrt=true -O3 -Xcompiler "$(CC_FLAGS)" 
CC_LIB		= -lm	$(subst KARTET_USE_ATLAS, -L/usr/lib64/atlas/ -llapack -lf77blas -lcblas -latlas, $(findstring KARTET_USE_ATLAS,$(KARTET_OPTIONS))) \
			$(subst KARTET_USE_CBLAS, -lcblas, $(findstring KARTET_USE_CBLAS,$(KARTET_OPTIONS))) \
			$(subst KARTET_USE_LAPACK, -llapack, $(findstring KARTET_USE_LAPACK,$(KARTET_OPTIONS))) \
			$(subst KARTET_USE_FFTW, -lfftw3 -lfftw3f, $(findstring KARTET_USE_FFTW,$(KARTET_OPTIONS))) \
		  $(CC_EXTRA_LIB)	
NVCC_LIB	= -L$(CUDA_LIB_PATH) -lcufft -lcublas -lcurand $(CC_LIB)

all : $(KARTET_DEFAULT_TARGET)
	@echo '[ ALL  ] Done.'

deviceBinary : $(DEVICE_OBJECTS)
	@echo '[DEVICE] Linking (Platform code $(DEVICE_CODE))...'
	@$(NVCC) -o $(EXEC_DEVICE) $(OBJ_DIR)$(notdir $^) $(NVCC_FLAGS) $(NVCC_LIB)

hostBinary : $(HOST_OBJECTS)
	@echo '[ HOST ] Linking...'
	@$(HSCC) -o $(EXEC_HOST) $(OBJ_DIR)$(notdir $^) $(CC_FLAGS) $(CC_LIB)

%.device.o : $(SOURCES_DIR)%.cu
	@echo '[DEVICE] Compiling $^ ...'	
	@$(NVCC) -o $(OBJ_DIR)$(notdir $@) -c $^ $(NVCC_INCLUDE) $(KARTET_OPTIONS) $(NVCC_FLAGS)

%.device.o : $(SOURCES_DIR)%.cpp
	@echo '[DEVICE] Compiling $^ ...'	
	@$(HSCC) -o $(OBJ_DIR)$(notdir $@) -c $^ $(CC_INCLUDE) $(KARTET_OPTIONS) $(CC_FLAGS)

%.device.o : $(SOURCES_DIR)%.c
	@echo '[DEVICE] Compiling $^ ...'	
	@$(HSCC) -o $(OBJ_DIR)$(notdir $@) -c $^ $(CC_INCLUDE) $(KARTET_OPTIONS) $(CC_FLAGS)

%.host.o : $(SOURCES_DIR)%.cu
	@echo '[ HOST ] Compiling $^ ...'
	@ln -sf $(notdir $^) $(patsubst %.cu,%.cpp,$^)
	@$(HSCC) -o $(OBJ_DIR)$(notdir $@) -c $(patsubst %.cu,%.cpp,$^) $(CC_INCLUDE) $(KARTET_OPTIONS) $(CC_FLAGS)
	@rm $(patsubst %.cu,%.cpp,$^)

%.host.o : $(SOURCES_DIR)%.cpp
	@echo '[ HOST ] Compiling $^ ...'
	@$(HSCC) -o $(OBJ_DIR)$(notdir $@) -c $^ $(CC_INCLUDE) $(KARTET_OPTIONS) $(CC_FLAGS)

%.host.o : $(SOURCES_DIR)%.c
	@echo '[ HOST ] Compiling $^ ...'
	@$(HSCC) -o $(OBJ_DIR)$(notdir $@) -c $^ $(CC_INCLUDE) $(KARTET_OPTIONS) $(CC_FLAGS)

clean:
	@rm -f $(OBJ_DIR)$(notdir $(DEVICE_OBJECTS)) $(OBJ_DIR)$(notdir $(HOST_OBJECTS))

mrproper: clean
	@rm -f $(EXEC_DEVICE) $(EXEC_HOST)

test:
	@echo "Options : "
	@echo "  KARTET_PATH           : $(KARTET_PATH)"
	@echo "  EXEC_DEVICE           : $(EXEC_DEVICE)"
	@echo "  EXEC_HOST             : $(EXEC_HOST)"
	@echo "  FILES                 : $(FILES)"
	@echo "  DEVICE_CODE           : $(DEVICE_CODE)"
	@echo "  CC_EXTRA_FLAGS        : $(CC_EXTRA_FLAGS)"
	@echo "  CC_EXTRA_LIB          : $(CC_EXTRA_LIB)"
	@echo "  HSCC                  : $(HSCC)"
	@echo "  NVCC                  : $(NVCC)"
	@echo "  KARTET_DEFAULT_TARGET : $(DEFAULT_TARGET)"
	@echo "  KARTET_OPTIONS        : $(KARTET_OPTIONS)"
	@echo "  SOURCES_DIR           : $(SOURCES_DIR)"
	@echo "  OBJ_DIR               : $(OBJ_DIR)"
	@echo "  CUDA_INCLUDE_PATH     : $(CUDA_INCLUDE_PATH)"
	@echo "  CUDA_LIB_PATH         : $(CUDA_LIB_PATH)"
	@echo "Variables : "
	@echo "  FILES_LIST            : $(FILES_LIST)"
	@echo "  OBJECTS               : $(OBJECTS)"
	@echo "  DEVICE_OBJECTS        : $(DEVICE_OBJECTS)"
	@echo "  HOST_OBJECTS          : $(HOST_OBJECTS)"
	@echo "  CC_INCLUDE            : $(CC_INCLUDE)"
	@echo "  NVCC_INCLUDE          : $(NVCC_INCLUDE)"
	@echo "  CC_FLAGS              : $(CC_FLAGS)"
	@echo "  NVCC_FLAGS            : $(NVCC_FLAGS)"
	@echo "  CC_LIB                : $(CC_LIB)"
	@echo "  NVCC_LIB              : $(NVCC_LIB)"

