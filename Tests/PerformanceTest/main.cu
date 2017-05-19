/* ************************************************************************************************************* */
/*                                                                                                               */
/*     Kartet                                                                                                    */
/*     A Simple C++ Array Library for CUDA                                                                       */
/*                                                                                                               */
/*     LICENSE : The MIT License                                                                                 */
/*     Copyright (c) 2015 Ronan Kerviche                                                                         */
/*                                                                                                               */
/*     Permission is hereby granted, free of charge, to any person obtaining a copy                              */
/*     of this software and associated documentation files (the "Software"), to deal                             */
/*     in the Software without restriction, including without limitation the rights                              */
/*     to use, copy, modify, merge, publish, distribute, sublicense, and/or sell                                 */
/*     copies of the Software, and to permit persons to whom the Software is                                     */
/*     furnished to do so, subject to the following conditions:                                                  */
/*                                                                                                               */
/*     The above copyright notice and this permission notice shall be included in                                */
/*     all copies or substantial portions of the Software.                                                       */
/*                                                                                                               */
/*     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR                                */
/*     IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,                                  */
/*     FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE                               */
/*     AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER                                    */
/*     LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,                             */
/*     OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN                                 */
/*     THE SOFTWARE.                                                                                             */
/*                                                                                                               */
/* ************************************************************************************************************* */

#include <iostream>
#include "GetArgs.hpp"
#include "Kartet.hpp"
#include "CuTimer.hpp"
#include <thrust/version.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

namespace thrust
{
	#define THRUST_UNARY_FUNCTION(NAME, RESULT) \
		template<typename T> \
		struct NAME \
		{ \
			__host__ __device__ T operator()(T x) { return RESULT ; } \
		};

	THRUST_UNARY_FUNCTION(usqrt,	sqrt(x))
	THRUST_UNARY_FUNCTION(uexp,	exp(x))
	THRUST_UNARY_FUNCTION(ulog,	log(x))
	THRUST_UNARY_FUNCTION(ucos,	cos(x))
	THRUST_UNARY_FUNCTION(usin,	sin(x))
	THRUST_UNARY_FUNCTION(utan,	tan(x))
	THRUST_UNARY_FUNCTION(uacos,	acos(x))
	THRUST_UNARY_FUNCTION(uasin,	asin(x))
	THRUST_UNARY_FUNCTION(uatan,	atan(x))
	THRUST_UNARY_FUNCTION(uextra,	sqrt(exp(tan(cos(sin(x))))))

	#undef THRUST_UNARY_FUNCTION

	#define THRUST_BINARY_FUNCTION(NAME, RESULT) \
		template<typename T> \
		struct NAME \
		{ \
			__host__ __device__ T operator()(T x, T y) { return RESULT ; } \
		};

	THRUST_BINARY_FUNCTION(bsqrt,	sqrt(x+y))
	THRUST_BINARY_FUNCTION(bexp,	exp(x+y))
	THRUST_BINARY_FUNCTION(blog,	log(x+y))
	THRUST_BINARY_FUNCTION(bcos,	cos(x+y))
	THRUST_BINARY_FUNCTION(bsin,	sin(x+y))
	THRUST_BINARY_FUNCTION(btan,	tan(x+y))
	THRUST_BINARY_FUNCTION(bacos,	acos(x+y))
	THRUST_BINARY_FUNCTION(basin,	asin(x+y))
	THRUST_BINARY_FUNCTION(batan,	atan(x+y))
	THRUST_BINARY_FUNCTION(bextra,	sqrt(exp(tan(cos(sin(x+y))))))

	#undef THRUST_BINARY_FUNCTION
}

STANDARD_UNARY_OPERATOR_DEFINITION(mySinObj, mySin, return sin(a); )

template<typename T>
__global__ void mySinFun(const Kartet::Accessor<T> acc)
{
	const Kartet::index_t k = threadIdx.x;//acc.getIndex();
	acc.data(k) = sin(acc.data(k));
}

__global__ void mySinFun2(float* data)
{
	data[threadIdx.x] = sin(data[threadIdx.x]);
}

int main(int argc, char** argv)
{
	int returnCode = 0;
	std::cout << "================================" << std::endl;
	std::cout << "    Kartet Performance Tests    " << std::endl;
	std::cout << "================================" << std::endl;
	std::cout << "Build : " << __DATE__ << ' ' << __TIME__ << std::endl;
	srand(time(NULL));

	try
	{
		// Select the device from the arguments ?
		int 	deviceId = 0,
			numSamples = 32;
		std::map<const char, int*> arguments;
		std::map<const char, std::string> argumentsHelp;
		arguments['d'] = &deviceId;
		argumentsHelp['d'] = "Target device Id, per nvidia-smi indexing.";
		arguments['l'] = &numSamples;
		argumentsHelp['l'] = "Number of samples, for the average.";
		if(!getArgs(argc, argv, arguments, argumentsHelp))
			throw Kartet::InvalidArgument;
		#ifdef __CUDACC__
			// Only executed if compiled by NVCC :
			std::cout << "Target device Id : " << deviceId << std::endl;
			cudaError_t cudaError = cudaSetDevice(deviceId);
			if(cudaError!=0)
				throw static_cast<Kartet::Exception>(Kartet::CudaExceptionsOffset + cudaError);
		#else
			// Otherwise :
			std::cout << "[Warning] Ignoring device selection in Host binary (deviceId=" << deviceId << ") ..." << std::endl;
		#endif

		// Start here :
		Kartet::initialize();
		Kartet::Layout::StaticContainer<void>::numThreads = 512;

		CuTimer timer;
		Kartet::BLASContext blasContext;
		Kartet::ReduceContext reduceContext;
	
		// Setup :
		typedef float T;
		if(Kartet::IsSame<T,float>::value)
			std::cout << "In single precision." << std::endl;
		else
			std::cout << "In double precision." << std::endl;

		const int M = 8192, N = 8192, L = std::max(1, numSamples);
		const double GB = 1024.0*1024.0*1024.0;
		double t = 0.0, v = 0.0;
		Kartet::Array<T> A(M, N), B(M, N), C(1, N), V(M),
				 ones(M, N);
		thrust::device_ptr<T> 	devPtrA( A.dataPtr() ),
					devPtrB( B.dataPtr() ),
					devPtrV( V.dataPtr() );
		ones = 1;
		A = Kartet::IndexI();
		B = Kartet::IndexJ();

		std::cout << "Number of samples : " << L << std::endl;
		// Force the syncrhonization :
		timer.start();
		timer.stop();

		/*// Test :
		timer.start();
		for(int l=0; l<L; l++)
			A = A+B;
		timer.stop();
		t = timer.getElapsedTime_s();
		std::cout << "Addition : " << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*3.0*A.size())/(GB*t) << " GB/s" << std::endl;
		std::cout << std::endl;

		// Against Thrust :
		timer.start();
		for(int l=0; l<L; l++)
			thrust::transform(devPtrA, devPtrA + A.numElements(), devPtrB, devPtrB, thrust::plus<T>());
			//thrust::transform(devPtrA, devPtrA + A.numElements(), devPtrA, thrust::tsin<T>());
		timer.stop();
		t = timer.getElapsedTime_s();
		std::cout << "[THRUST] Addition : " << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*3.0*A.size())/(GB*t) << " GB/s" << std::endl;
		std::cout << std::endl;*/

		// Tests :
		#define TEST_UNARY_EXPRESSION(TITLE, EXPRESSION, THRUST_EXPRESSION) \
			A = Kartet::IndexI(); \
			timer.start(); \
			for(int l=0; l<L; l++) \
			{ \
				EXPRESSION ;\
			} \
			timer.stop(); \
			t = timer.getElapsedTime_s(); \
			std::cout << TITLE << std::endl; \
			std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl; \
			std::cout << "Cumulative bandwidth : " << (L*2.0*A.size())/(GB*t) << " GB/s" << std::endl; \
			A = Kartet::IndexI(); \
			timer.start(); \
			for(int l=0; l<L; l++) \
			{ \
				thrust::transform(devPtrA, devPtrA + A.numElements(), devPtrA, thrust:: THRUST_EXPRESSION <T>() ); \
			} \
			timer.stop(); \
			t = timer.getElapsedTime_s(); \
			std::cout << "/ Thrust : " << std::endl; \
			std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl; \
			std::cout << "Cumulative bandwidth : " << (L*2.0*A.size())/(GB*t) << " GB/s" << std::endl; \
			std::cout << std::endl;

		TEST_UNARY_EXPRESSION("[Unary] Sqrt(x) : ",	A = sqrt(A),	usqrt)
		TEST_UNARY_EXPRESSION("[Unary] Exp(x) : ",	A = exp(A),	uexp)
		TEST_UNARY_EXPRESSION("[Unary] Log(x) : ",	A = log(A),	ulog)
		TEST_UNARY_EXPRESSION("[Unary] Cos(x) : ",	A = cos(A),	ucos)
		TEST_UNARY_EXPRESSION("[Unary] Sin(x) : ",	A = sin(A),	usin)
		TEST_UNARY_EXPRESSION("[Unary] Tan(x) : ",	A = tan(A),	utan)
		TEST_UNARY_EXPRESSION("[Unary] Acos(x) : ",	A = acos(A),	uacos)
		TEST_UNARY_EXPRESSION("[Unary] Asin(x) : ",	A = asin(A),	uasin)
		TEST_UNARY_EXPRESSION("[Unary] Atan(x) : ",	A = atan(A),	uatan)
		TEST_UNARY_EXPRESSION("[Unary] Extra(x) : ",	A = sqrt(exp(tan(cos(sin(A))))), uextra)

		/*TEST_UNARY_EXPRESSION("TEST #1 : ", A = mySin(B), usin)
		TEST_UNARY_EXPRESSION("TEST #2 : ", mySinFun COMPUTE_LAYOUT(A) (A), usin)
		TEST_UNARY_EXPRESSION("TEST #3 : ", mySinFun2 COMPUTE_LAYOUT(A) (A.dataPtr()), usin)*/

		#undef TEST_UNARY_EXPRESSION

		#define TEST_BINARY_EXPRESSION(TITLE, EXPRESSION, THRUST_EXPRESSION) \
			A = Kartet::IndexI(); \
			timer.start(); \
			for(int l=0; l<L; l++) \
			{ \
				EXPRESSION ;\
			} \
			timer.stop(); \
			t = timer.getElapsedTime_s(); \
			std::cout << TITLE << std::endl; \
			std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl; \
			std::cout << "Cumulative bandwidth : " << (L*3.0*A.size())/(GB*t) << " GB/s" << std::endl; \
			A = Kartet::IndexI(); \
			timer.start(); \
			for(int l=0; l<L; l++) \
			{ \
				thrust::transform(devPtrA, devPtrA + A.numElements(), devPtrB, devPtrB, thrust:: THRUST_EXPRESSION <T>() ); \
			} \
			timer.stop(); \
			t = timer.getElapsedTime_s(); \
			std::cout << "/ Thrust : " << std::endl; \
			std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl; \
			std::cout << "Cumulative bandwidth : " << (L*3.0*A.size())/(GB*t) << " GB/s" << std::endl; \
			std::cout << std::endl;

		TEST_BINARY_EXPRESSION("[Binary] Sqrt(x+y) : ",	A = sqrt(A+B),	bsqrt)
		TEST_BINARY_EXPRESSION("[Binary] Exp(x+y) : ",	A = exp(A+B),	bexp)
		TEST_BINARY_EXPRESSION("[Binary] Log(x+y) : ",	A = log(A+B),	blog)
		TEST_BINARY_EXPRESSION("[Binary] Cos(x+y) : ",	A = cos(A+B),	bcos)
		TEST_BINARY_EXPRESSION("[Binary] Sin(x+y) : ",	A = sin(A+B),	bsin)
		TEST_BINARY_EXPRESSION("[Binary] Tan(x+y) : ",	A = tan(A+B),	btan)
		TEST_BINARY_EXPRESSION("[Binary] Acos(x+y) : ",	A = acos(A+B),	bacos)
		TEST_BINARY_EXPRESSION("[Binary] Asin(x+y) : ",	A = asin(A+B),	basin)
		TEST_BINARY_EXPRESSION("[Binary] Atan(x+y) : ",	A = atan(A+B),	batan)
		TEST_BINARY_EXPRESSION("[Binary] Sin(x+y) : ",	A = mySin(A+B),	bsin)
		TEST_BINARY_EXPRESSION("[Binary] Extra(x+y) : ",A = sqrt(exp(tan(cos(sin(A+B))))), bextra)

		#undef TEST_BINARY_EXPRESSION

		// Test :
		//timer.start();
		//for(int l=0; l<L; l++)
		//	A = exp(cos(sin(A*l+B)));
		//timer.stop();
		//t = timer.getElapsedTime_s();
		//std::cout << "Expression : " << std::endl;
		//std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		//std::cout << "Cumulative bandwidth : " << (L*3.0*A.size())/(GB*t) << " GB/s" << std::endl;
		//std::cout << std::endl;

		// Test :	
		/*v = 0.0;
		A = 1;
		timer.start();
		for(int l=0; l<L; l++)
			v += reduceContext.sum(A);
		timer.stop();
		t = timer.getElapsedTime_s();
		std::cout << "Reduction sum : " << v << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*A.size())/(GB*t) << " GB/s" << std::endl;
		std::cout << std::endl;

		// Against Thrust :
		v = 0.0;
		timer.start();
		for(int l=0; l<L; l++)
			v += thrust::reduce(devPtrA, devPtrA + A.numElements(), 0.0, thrust::plus<T>());
		timer.stop();
		t = timer.getElapsedTime_s();
		std::cout << "[THRUST] Reduction sum : " << v << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*A.size())/(GB*t) << " GB/s" << std::endl;
		std::cout << std::endl;

		// Test :
		v = 0.0;
		timer.start();
		for(int l=0; l<L; l++)
			v += reduceContext.sum(A.layout(), A*l);
		timer.stop();
		t = timer.getElapsedTime_s();
		std::cout << "Expression Reduction sum : " << v << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*A.size())/(GB*t) << " GB/s" << std::endl;
		std::cout << std::endl;

		// Against Thrust :
		v = 0.0;
		timer.start();
		for(int l=0; l<L; l++)
		{
			B = A*l;
			v += thrust::reduce(devPtrB, devPtrB + B.numElements(), 0.0, thrust::plus<T>());
		}
		timer.stop();
		t = timer.getElapsedTime_s();
		std::cout << "[THRUST] Expression Reduction sum : " << v << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*3.0*A.size())/(GB*t) << " GB/s (but uses two more I/O operations)" << std::endl;
		std::cout << std::endl;

		// Test :
		timer.start();
		for(int l=0; l<L; l++)
			reduceContext.sumBlock(A.layout(), A*A, C);
		timer.stop();
		t = timer.getElapsedTime_s();
		std::cout << "Expression Reduction multi-sum : " << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*A.size())/(GB*t) << " GB/s" << std::endl;
		std::cout << std::endl;

		// Against BLAS (1) :
		timer.start();
		for(int l=0; l<L; l++)
		{
			for(unsigned int k=0; k<A.numColumns(); k++)
				v += blasContext.nrm2(A.column(k));
		}
		timer.stop();
		t = timer.getElapsedTime_s();
		std::cout << "[BLAS] (1) Expression Reduction multi-sum : " << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*1.0*A.size())/(GB*t) << " GB/s" << std::endl;
		std::cout << std::endl;

		// Against BLAS (2) :
		timer.start();
		for(int l=0; l<L; l++)
		{
			B = A*A;
			blasContext.gemm(ones.column(0), Kartet::OpTr, B, Kartet::OpNo, C);
		}
		timer.stop();
		t = timer.getElapsedTime_s();
		std::cout << "[BLAS] (2) Expression Reduction multi-sum : " << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*3.0*A.size())/(GB*t) << " GB/s" << std::endl;
		std::cout << std::endl;

		// Against Thrust :
		timer.start();
		for(int l=0; l<L; l++)
		{
			for(unsigned int k=0; k<A.numColumns(); k++)
			{
				V = A.column(k)*A.column(k);
				v += thrust::reduce(devPtrV, devPtrV + V.numElements(), 0.0, thrust::plus<T>());
			}
		}
		timer.stop();
		t = timer.getElapsedTime_s();
		std::cout << "[THRUST] Expression Reduction multi-sum : " << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*3.0*A.size())/(GB*t) << " GB/s" << std::endl;
		std::cout << std::endl;*/
	}
	catch(const Kartet::Exception& e)
	{
		std::cerr << "Exception : " << e << std::endl;
		returnCode = -1;
	}

	return returnCode;
}

