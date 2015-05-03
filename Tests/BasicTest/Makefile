# Basic Program

# Default plateform level, change with :
# make TGPU=<platform code>
ifeq ($(TGPU),)
	TGPU = 35
endif

EXEC		= kartetBasicTest
INCLUDES	= -I../../include/ -I/usr/local/cuda/include
NVCCFLAGS	= -arch=compute_$(TGPU) -code=sm_$(TGPU) -ftz=false -prec-div=true -prec-sqrt=true -O3 -Xcompiler -m64
NVCCLDFLAGS	= -L/usr/local/cuda/lib64 -lcufft -lcublas -lcurand -llapack -lf77blas -lcblas -latlas 
			# You can also use the multithreaded versions : -lptlapack -lptf77blas -lptcblas -latlas	
			# Make sure that Atlas was compile for the number of core you are targeting.
			# See http://math-atlas.sourceforge.net/faq.html#tnum
KARTETOPTIONS	= -D KARTET_USE_ATLAS 
	# Change the default location : -D KARTET_DEFAULT_LOCATION=HostSide

all : main.o
	@echo 'Linking (platform code $(TGPU))...'
	@nvcc -o $(EXEC) $^ $(INCLUDES) $(NVCCLDFLAGS)

main.o : main.cu
	@echo 'Compiling $^ ...'
	@nvcc -o $@ -c $^ $(INCLUDES) $(KARTETOPTIONS) $(NVCCFLAGS)

clean:
	@rm -f main.o 

mrproper: clean
	@rm -f $(EXEC)

