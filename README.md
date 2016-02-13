# Kartet

A minimal C++ array library for CPU & GPU (CUDA) computing.

* Header-only code.
* Expression Templates with operators and functions.
* Transparent support for complex numbers.
* CuBLAS, CuRAND and CuFFT interfaces.
* Optional Host-Only binary generation.
* Optional CBLAS, ATLAS and FFTW(3) interfaces.

## Installation :
  Set the environment variable KARTET_PATH
to the path of the library.
  (Optional) Set the environment variable
KARTET_DEFAULT_TARGET to either 'deviceBinary'
or 'hostBinary' to generate only one binary.

## Options : 
* -D KARTET_USE_64BITS_INDEXING			Use 64 bits indexing, slower operation (2x) but enable large arrays (>2e9 elements).
* -D KARTET_USE_ATLAS				Use ATLAS (requires -llapack -lf77blas -lcblas -latlas).
* -D KARTET_USE_CBLAS				Use CBLAS only (requires -lcblas, ignored if KARTET_USE_ATLAS).
* -D KARTET_USE_FFTW				Use FFTW (v3, requires -lfftw3 -lfftw3f).
* -D KARTET_USE_OPENMP				Use OpenMP (preliminary, only a few constructs will benefit).
* -D KARTET_DEFAULT_LOCATION=newLocation	Change the default location (HostSide or DeviceSide).

## Examples :
  See Tests/BasicTest for simple operations.

