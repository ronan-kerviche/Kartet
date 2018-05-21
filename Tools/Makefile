# GENERIC MAKEFILE TOOL FOR KARTET
# Options :
# KARTET_PATH		: Path to Kartet directory
# EXEC_DEVICE		: Name of the executable to generate for the device platform.
# EXEC_HOST		: Name of the executable to generate for the host platform.
# FILES			: Files to compile (*.cu, *.cpp)
# DEVICE_CODE		: Target device platform code (20 30 35 37 50 52 60...), can be a list of space-separated codes.
# CC_EXTRA_FLAGS	: Extra flags for the compilers.
# CC_EXTRA_INCLUDE	: Extra flags for the compilers.
# CC_EXTRA_LIB		: Extra flags for the linkers.
# HSCC			: Host side compiler.
# NVCC			: Device side compiler.
# KARTET_DEFAULT_TARGET	: Default compilation target amon deviceBinary and hostBinary.
# SOURCES_DIR		: Directory for the source files.
# OBJ_DIR		: Directory for the object files.
# CUDA_INCLUDE_PATH	: Path to CUDA headers directory.
# CUDA_LIB_PATH		: Path to CUDA libraries.
# CUDA_FAST_MATH	: Enable fast math.
# KARTET_OPTIONS	: Options for the library
#     -D KARTET_USE_64BITS_INDEXING		Use 64 bits indexing, slower operation (2x) but enable large arrays (>2e9 elements).
#     -D KARTET_USE_ATLAS			Use ATLAS (requires -llapack -lf77blas -lcblas -latlas).
#     -D KARTET_USE_SATLAS			Use SATLAS (serial).
#     -D KARTET_USE_TATLAS			Use TATLAS (threaded).
#     -D KARTET_USE_OPENBLAS			USe OpenBLAS.
#     -D KARTET_USE_CBLAS			Use CBLAS only (requires -lcblas, ignored if KARTET_USE_ATLAS).
#     -D KARTET_USE_FFTW			Use FFTW (v3, requires -lfftw3 -lfftw3f).
#     -D KARTET_USE_OMP				Enable OpenMP (preliminary, on a few constructs only).
#     -D KARTET_DEFAULT_LOCATION=newLocation	Change the default location (HostSide or DeviceSide).
#
#     Note : 
#     You can also use the older multithreaded versions of ATLAS by replacing with : -lptlapack -lptf77blas -lptcblas -latlas
#     Make sure that Atlas was compile for the number of core you are targeting.
#     See http://math-atlas.sourceforge.net/faq.html#tnum
#
# Example of client Makefile :
#	export EXEC_DEVICE	= execNameDevice
#	export EXEC_HOST	= execNameTestHost
#	export FILES		= main.cu
#	export KARTET_OPTIONS	+= -D KARTET_USE_ATLAS
#	#export CC_EXTRA_LIB	+= -shared # For a shared object/dynamic library
#	all :
#	% :
#		@$(MAKE) -s -f $(KARTET_PATH)/Tools/Makefile $@
#
# Test :
# make test

# Note on NVidia Architecture :
# -arch=sm_XX is equivalent to -gencode=arch=compute_XX,code=sm_XX -gencode=arch=compute_XX,code=compute_XX
# NVCC generates intermediate PTX (VM specified by -arch) and closer SASS (-code) from the sources. 
# PTX can be transformed into SASS by the driver without having the CUDA SDK installed.

# Note on FFTW3's error: identifier "__float128" is undefined
# Passing -D__INTEL_COMPILER solve the problem for NVCC.

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
	DEVICE_CODE = 30 35 37 50 52 60
#$(info Using default platform codes : $(DEVICE_CODE))
#$(info To speed up compilation, specify the environment variable DEVICE_CODE.)
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
	#CUDA_INCLUDE_PATH = /usr/local/cuda/include # Usually
	CUDA_INCLUDE_PATH = $(subst /bin/nvcc,/include,$(shell which $(NVCC)))
endif

ifeq ($(CUDA_LIB_PATH),)
	#CUDA_LIB_PATH = /usr/local/cuda/lib64 # Usually
	CUDA_LIB_PATH = $(subst /bin/nvcc,/lib64,$(shell which $(NVCC)))
endif

ifeq ($(CUDA_FAST_MATH),)
	NVCC_PERFORMANCE_FLAGS = -fmad=true -ftz=false -prec-div=true -prec-sqrt=true
else
	NVCC_PERFORMANCE_FLAGS = -fmad=true -ftz=true -prec-div=false -prec-sqrt=false -use_fast_math
endif

# Gencodes :
ifeq ($(CUDA_GENCODES),)
$(foreach dc,$(DEVICE_CODE),$(eval CUDA_GENCODES += -gencode arch=compute_$(dc),code=sm_$(dc)))
	CUDA_HIGHEST_DEVICECODE := $(lastword $(sort $(DEVICE_CODE)))
	ifneq ($(CUDA_HIGHEST_DEVICECODE),)
		CUDA_GENCODES += -gencode arch=compute_$(CUDA_HIGHEST_DEVICECODE),code=compute_$(CUDA_HIGHEST_DEVICECODE)
	endif
endif

# Generate :
FILES_LIST	= $(wildcard $(SOURCES_DIR)$(FILES))
OBJECTS		= $(patsubst %.cu,%.o,$(filter %.cu,$(FILES_LIST))) $(patsubst %.cpp,%.o,$(filter %.cpp,$(FILES_LIST))) $(patsubst %.c,%.o,$(filter %.c,$(FILES_LIST)))
DEVICE_OBJECTS	= $(patsubst %.o,%.device.o,$(OBJECTS))
HOST_OBJECTS	= $(patsubst %.o,%.host.o,$(OBJECTS))
CC_INCLUDE	= -I$(KARTET_PATH)/include/ \
			$(subst KARTET_USE_ATLAS, -I/usr/include/atlas, $(findstring KARTET_USE_ATLAS,$(KARTET_OPTIONS))) \
			$(subst KARTET_USE_SATLAS, -I/usr/include/atlas, $(findstring KARTET_USE_SATLAS,$(KARTET_OPTIONS))) \
			$(subst KARTET_USE_TATLAS, -I/usr/include/atlas, $(findstring KARTET_USE_TATLAS,$(KARTET_OPTIONS))) \
			$(subst KARTET_USE_OPENBLAS, -I/usr/local/opt/openblas/include, $(findstring KARTET_USE_OPENBLAS,$(KARTET_OPTIONS))) \
			$(CC_EXTRA_INCLUDE)
NVCC_INCLUDE	= -I$(CUDA_INCLUDE_PATH) $(CC_INCLUDE)
CC_FLAGS	= -fPIC -m64 -O3 -march=native -Wall -Wextra -Wno-unused-local-typedefs \
			$(subst KARTET_USE_OMP, -fopenmp, $(findstring KARTET_USE_OMP,$(KARTET_OPTIONS))) \
		   $(CC_EXTRA_FLAGS)
NVCC_FLAGS	= $(CUDA_GENCODES) $(NVCC_PERFORMANCE_FLAGS) -O3 -D__INTEL_COMPILER -Xcompiler "$(CC_FLAGS)" -ccbin=$(HSCC)
CC_LIB		= -lm	$(subst KARTET_USE_ATLAS, -L/usr/lib64/atlas/ -llapack -lf77blas -lcblas -latlas, $(findstring KARTET_USE_ATLAS,$(KARTET_OPTIONS))) \
			$(subst KARTET_USE_SATLAS, -L/usr/lib64/atlas/ -lsatlas, $(findstring KARTET_USE_SATLAS,$(KARTET_OPTIONS))) \
			$(subst KARTET_USE_TATLAS, -L/usr/lib64/atlas/ -ltatlas, $(findstring KARTET_USE_TATLAS,$(KARTET_OPTIONS))) \
			$(subst KARTET_USE_OPENBLAS, -L/usr/local/opt/openblas/lib/ -lblas -llapack -lopenblas, $(findstring KARTET_USE_OPENBLAS,$(KARTET_OPTIONS))) \
			$(subst KARTET_USE_CBLAS, -lcblas, $(findstring KARTET_USE_CBLAS,$(KARTET_OPTIONS))) \
			$(subst KARTET_USE_LAPACK, -llapack, $(findstring KARTET_USE_LAPACK,$(KARTET_OPTIONS))) \
			$(subst KARTET_USE_FFTW, -lfftw3 -lfftw3f, $(findstring KARTET_USE_FFTW,$(KARTET_OPTIONS))) \
		   $(CC_EXTRA_LIB)	
NVCC_LIB	= -L$(CUDA_LIB_PATH) -lcufft -lcublas -lcurand $(CC_LIB)

all : $(KARTET_DEFAULT_TARGET)
	@echo '[ ALL  ] Done.'

deviceBinary : $(DEVICE_OBJECTS)
	@echo '[DEVICE] Linking (Platform code(s) : $(DEVICE_CODE)) ...'
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
	@ln -sf $(notdir $^) $(patsubst %.cu,%.lnk.cpp,$^)
	@$(HSCC) -o $(OBJ_DIR)$(notdir $@) -c $(patsubst %.cu,%.lnk.cpp,$^) $(CC_INCLUDE) $(KARTET_OPTIONS) $(CC_FLAGS)
	@rm $(patsubst %.cu,%.lnk.cpp,$^)

%.host.o : $(SOURCES_DIR)%.cpp
	@echo '[ HOST ] Compiling $^ ...'
	@$(HSCC) -o $(OBJ_DIR)$(notdir $@) -c $^ $(CC_INCLUDE) $(KARTET_OPTIONS) $(CC_FLAGS)

%.host.o : $(SOURCES_DIR)%.c
	@echo '[ HOST ] Compiling $^ ...'
	@$(HSCC) -o $(OBJ_DIR)$(notdir $@) -c $^ $(CC_INCLUDE) $(KARTET_OPTIONS) $(CC_FLAGS)

clean:
	@rm -f $(OBJ_DIR)$(notdir $(DEVICE_OBJECTS)) $(OBJ_DIR)$(notdir $(HOST_OBJECTS))

mrclean: clean
	@rm -f $(EXEC_DEVICE) $(EXEC_HOST)

help:
	@head -n 43 $(KARTET_PATH)/Tools/Makefile

test:
	@echo "Options : "
	@echo "  KARTET_PATH           : $(KARTET_PATH)"
	@echo "  EXEC_DEVICE           : $(EXEC_DEVICE)"
	@echo "  EXEC_HOST             : $(EXEC_HOST)"
	@echo "  FILES                 : $(FILES)"
	@echo "  DEVICE_CODE           : $(DEVICE_CODE)"
	@echo "  CC_EXTRA_FLAGS        : $(CC_EXTRA_FLAGS)"
	@echo "  CC_EXTRA_INCLUDE      : $(CC_EXTRA_INCLUDE)"
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
	@echo "  CUDA_GENCODES         : $(CUDA_GENCODES)"

