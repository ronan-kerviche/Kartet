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
#include "Kartet.hpp"
#include "CuTimer.hpp"
#include <thrust/version.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

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
		Kartet::initialize();

		CuTimer timer;
		Kartet::BLASContext blasContext;
		Kartet::ReduceContext reduceContext;
	
		// Speed test:
		typedef float T;
		const int M = 4096, N = 4096, L = 20;
		double t = 0.0, v = 0.0;
		Kartet::Array<T> A(M, N), B(M, N), C(1, N), V(M),
				 ones(M, N);
		thrust::device_ptr<T> 	devPtrA( A.dataPtr() ),
					devPtrB( B.dataPtr() ),
					devPtrV( V.dataPtr() );
		// Setup :
		ones = 1;
		A = Kartet::IndexI();
		B = Kartet::IndexJ();

		// Test :
		timer.start();
		for(int l=0; l<L; l++)
			A = A+B;
		timer.stop();
		t = timer.getElapsedTime_s();
		std::cout << "Addition                                  : " << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*3.0*A.size())/(1024.0*1024.0*1024.0*t) << " GB/s" << std::endl;
		std::cout << std::endl;

		// Against Thrust :
		timer.start();
		for(int l=0; l<L; l++)
			thrust::transform(devPtrA, devPtrA + A.numElements(), devPtrB, devPtrB, thrust::plus<T>());
		timer.stop();
		t = timer.getElapsedTime_s();
		std::cout << "[THRUST] Addition                         : " << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*3.0*A.size())/(1024.0*1024.0*1024.0*t) << " GB/s" << std::endl;
		std::cout << std::endl;

		// Test :
		timer.start();
		for(int l=0; l<L; l++)
			A = A*l+B;
		timer.stop();
		t = timer.getElapsedTime_s();
		std::cout << "Affine Expression                         : " << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*3.0*A.size())/(1024.0*1024.0*1024.0*t) << " GB/s" << std::endl;
		std::cout << std::endl;

		// Test :
		timer.start();
		for(int l=0; l<L; l++)
			A = sin(A*l+B);
		timer.stop();
		t = timer.getElapsedTime_s();
		std::cout << "Expression                                : " << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*3.0*A.size())/(1024.0*1024.0*1024.0*t) << " GB/s" << std::endl;
		std::cout << std::endl;

		// Test :	
		v = 0.0;
		timer.start();
		for(int l=0; l<L; l++)
			v += reduceContext.sum(A);
		timer.stop();
		t = timer.getElapsedTime_s();
		std::cout << "Reduction sum                             : " << v << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*A.size())/(1024.0*1024.0*1024.0*t) << " GB/s" << std::endl;
		std::cout << std::endl;

		// Against Thrust :
		v = 0.0;
		timer.start();
		for(int l=0; l<L; l++)
			v += thrust::reduce(devPtrA, devPtrA + A.numElements(), 0.0, thrust::plus<T>());
		timer.stop();
		t = timer.getElapsedTime_s();
		std::cout << "[THRUST] Reduction sum                    : " << v << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*A.size())/(1024.0*1024.0*1024.0*t) << " GB/s" << std::endl;
		std::cout << std::endl;

		// Test :
		v = 0.0;
		timer.start();
		for(int l=0; l<L; l++)
			v += reduceContext.sum(A.layout(), A*l);
		timer.stop();
		t = timer.getElapsedTime_s();
		std::cout << "Expression Reduction sum                  : " << v << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*A.size())/(1024.0*1024.0*1024.0*t) << " GB/s" << std::endl;
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
		std::cout << "[THRUST] Expression Reduction sum         : " << v << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*3.0*A.size())/(1024.0*1024.0*1024.0*t) << " GB/s (but uses two more I/O operations)" << std::endl;
		std::cout << std::endl;

		// Test :
		timer.start();
		for(int l=0; l<L; l++)
			reduceContext.sumBlock(A.layout(), A*A, C);
		timer.stop();
		t = timer.getElapsedTime_s();
		std::cout << "Expression Reduction multi-sum            : " << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*A.size())/(1024.0*1024.0*1024.0*t) << " GB/s" << std::endl;
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
		std::cout << "Cumulative bandwidth : " << (L*1.0*A.size())/(1024.0*1024.0*1024.0*t) << " GB/s" << std::endl;
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
		std::cout << "Cumulative bandwidth : " << (L*3.0*A.size())/(1024.0*1024.0*1024.0*t) << " GB/s" << std::endl;
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
		std::cout << "[THRUST] Expression Reduction multi-sum   : " << std::endl;
		std::cout << "Elapsed time         : " << (t/L) << " second(s)" << std::endl;
		std::cout << "Cumulative bandwidth : " << (L*3.0*A.size())/(1024.0*1024.0*1024.0*t) << " GB/s" << std::endl;
		std::cout << std::endl;
	}
	catch(const Kartet::Exception& e)
	{
		std::cerr << "Exception : " << e << std::endl;
		returnCode = -1;
	}

	return returnCode;
}

