export EXEC_DEVICE		= kartetPerformanceTestDevice
export EXEC_HOST		= kartetPerformanceTestHost
export FILES			= main.cu CuTimer.cu ../Common/GetArgs.cpp
export KARTET_DEFAULT_TARGET	= deviceBinary
export KARTET_OPTIONS		+= -D KARTET_USE_ATLAS
export CC_EXTRA_INCLUDE		+= -I../Common/
all :
% :
	@$(MAKE) -s -f $(KARTET_PATH)/Tools/Makefile $@
