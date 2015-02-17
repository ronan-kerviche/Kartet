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

	__global__ void testPrint(const Kartet::Layout layout)
	{
		const Kartet::index_t	i = layout.getI(), 
					j = layout.getJ(), 
					k = layout.getK(),
					p = layout.getIndex();
		printf("  Hi, from block : (%d; %d; %d) Thread : (%d; %d; %d) => (i=%d; j=%d; k=%d):p=%d\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z, (int)i, (int)j, (int)k, (int)p);
	}

int main(int argc, char** argv)
{
	int returnCode = 0;
	std::cout << "================================" << std::endl;
	std::cout << "       Kartet Syntax Tests      " << std::endl;
	std::cout << "================================" << std::endl;
	std::cout << "Build : " << __DATE__ << ' ' << __TIME__ << std::endl;
	srand(time(NULL));

	try
	{
		// Testing expressions :
		Kartet::Array<double> A(5, 3), B(5, 3);

		A = Kartet::IndexI();
		B = Kartet::IndexJ();
		std::cout << "A = " << A << std::endl;
		std::cout << "B = " << B << std::endl;
		A = A+B;
		std::cout << "S = " << A << std::endl;
		
		std::cout << "Layout : " << A.getLayout() << std::endl;
		A = 1.23;
		B = 2.56 * Kartet::IndexJ();
		A = cos(A + exp(2.0 * B));
		std::cout << "A = " << A << std::endl;

		cudaDeviceSynchronize();
		std::cout << "Test kernel : " << std::endl;
		testPrint COMPUTE_LAYOUT(A) (A.getLayout());
		cudaDeviceSynchronize();
		std::cout << "Done." << std::endl;

		A = min(A, B);
		std::cout << "A = " << A << std::endl;

		Kartet::Array<double> C(16);
		C = Kartet::IndexI();
		std::cout << "C = " << C << std::endl;

		C = Kartet::cast<unsigned char>(Kartet::IndexI()*64);
		std::cout << "C = " << C << std::endl;

		// BLAS :
		Kartet::BLASContext blas;
		C = 4 - absSq(Kartet::IndexI()-13) / 100.0;
		std::cout << "C = " << C << std::endl;
		int idx = blas.amax(C);
		std::cout << "Index of the absolute maximum : " << idx << std::endl;
		std::cout << std::endl;

		const int w1 = 256, w2 = 512, w3 = 348;
		Kartet::Array<float> X(w1, w3), Y(w3, w2), Z(w1, w2);
		X = Kartet::IndexI();
		Y = Kartet::IndexJ();
		Z = 0.0;
		//gemm(1.0, X, CUBLAS_OP_N, Y, CUBLAS_OP_N, 0.0, Z);
		blas.gemm(X, Y, Z);

		// Generate random numbers :
		Kartet::RandomSourceContext randomSourceContext;
		randomSourceContext.setSeed();
		Kartet::UniformSource uniformSource;
		uniformSource >> A;
		std::cout << "A = " << A << std::endl;

		// Accessing data :
		A = Kartet::IndexI() + Kartet::IndexJ();
		// Select 2 vectors starting at 0 with a step of 2 (1st and 3rd vectors).
		Kartet::Accessor<double> S = A.vectors(0, 2, 2);
		uniformSource >> S;
		std::cout << "S = " << S << std::endl;
		std::cout << "A = " << A << std::endl;
		A.subArray(1,0,3,3) = -23.0;
		std::cout << A << std::endl;
	
		// More manipulations with accessors :
		Kartet::Array<float> D(4, 4, 3);
		Kartet::NormalSource normalSource(2.0, 10.0);
		normalSource >> D;
		std::cout << "D = " << D << std::endl;
		D.vectors(0, 2, 2) = 0;
		std::cout << "D = " << D << std::endl;
		D.vectors(0, 2, 2).slice(1) = 1.0;
		std::cout << "D = " << D << std::endl;

		// Computing on complex numbers without storing :
		D = real(piAngleToComplex(Kartet::IndexI() + Kartet::IndexJ()));
		std::cout << "D = " << D << std::endl;
		std::cout << "Layout of D.slices(0, 2, 2) : " << D.slices(0, 2, 2).getLayout() << std::endl;
		D.slice(0) = D.slice(0)*D.slice(1);
		std::cout << "D.slice(0) = " << D.slice(0) << std::endl;
		D.slice(1) = (D.slice(1) + D.slice(0))/2.0;
		std::cout << "D = " << D << std::endl;

		// Complex :
		Kartet::Array<cuDoubleComplex> CxA(4, 4);
		CxA = 1.0 + Kartet::IndexJ();
		CxA = angleToComplex(real(CxA));
		std::cout << "CxA = " << CxA << std::endl;
		Kartet::Array<double> CxAbs(CxA.getLayout());
		CxAbs = abs(CxA) - real(CxA);
		std::cout << "CxAbs = " << CxAbs << std::endl;
		CxAbs = angle(CxA);
		std::cout << "CxAbs = " << CxAbs << std::endl;

		// File I/O :
		Kartet::Array<int> U(8, 8);
		U = Kartet::IndexI() + Kartet::IndexJ();
		U.writeToFile("tmp.dat");
		std::cout << "U = " << U << std::endl;
		Kartet::Array<float> V(8, 8);
		V.readFromFile("tmp.dat");
		std::cout << "V = " << V << std::endl;
		Kartet::Array<cuDoubleComplex> W("tmp.dat");
		std::cout << "W = " << W << std::endl;

		// Reduction :
		Kartet::ReduceContext reduceContext;
		const Kartet::Layout l(4661,7965);
		std::cout << "Layout : " << l << std::endl;
		int sum1 = reduceContext.sum(l, 1);
		std::cout << "Sum(1) : " << sum1 << " == " << l.getNumElements() << ", test : " << (sum1==l.getNumElements()) << std::endl;

		double  sum2 = reduceContext.sum(l, Kartet::cast<double>(Kartet::IndexI()+Kartet::IndexJ())),
			res2 = static_cast<double>(l.getNumRows()+l.getNumColumns()-2)*static_cast<double>(l.getNumRows()*l.getNumColumns())/2.0;
		std::cout << "Sum(I()+J()) : " << sum2 << " == " << res2 << ", diff : " << std::abs(sum2-res2)/res2 << std::endl;
		
		int testReduction = reduceContext.sum(U);
		std::cout << "Sum(U) : " << testReduction << std::endl;
		testReduction = reduceContext.sum(U.getLayout(), Kartet::IndexI()*Kartet::IndexJ());
		std::cout << "Sum(I()*J()) = " << testReduction << " (==784 on 8x8)" << std::endl;
		std::cout << "Sum(sqrt(I()*J())) = " << reduceContext.sum(Kartet::Layout(16, 16), sqrt(Kartet::cast<double>(Kartet::IndexI()*Kartet::IndexJ()))) << " (\\approx 1637.755873, for 16x16)" << std::endl;
	}
	catch(Kartet::Exception& e)
	{
		std::cout << "Exception : " << e << std::endl;
		returnCode = -1;
	}

	return returnCode;
}

