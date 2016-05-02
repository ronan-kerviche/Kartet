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

int main(int argc, char** argv)
{
	UNUSED_PARAMETER(argc)
	UNUSED_PARAMETER(argv)

	int returnCode = 0;
	std::cout << "================================" << std::endl;
	std::cout << "      Kartet Syntax Example     " << std::endl;
	std::cout << "================================" << std::endl;
	std::cout << "Build : " << __DATE__ << ' ' << __TIME__ << std::endl;
	srand(time(NULL));

	try
	{
		Kartet::initialize();

		// Creating a Layout (size of an array) :
		const Kartet::Layout layout(5, 3);

		// Creating arrays :
		Kartet::Array<float> 	A(layout),
					B(layout),
					C(layout);

		Kartet::Array< Kartet::Complex<double>, Kartet::HostSide> Cx(layout);

		// Initialization, fill with ones :
		A = 1.0f;
		std::cout << "A = " << A << std::endl;

		// Initialization with the row indices :
		B = Kartet::IndexI();
		std::cout << "B = " << B << std::endl;

		Cx = Kartet::complex(Kartet::IndexI(), Kartet::IndexJ());
		std::cout << "Cx = " << Cx << std::endl;

		// Compute an expression :
		C = 2.0f*(B-A);
		std::cout << "C = " << C << std::endl;

		Cx = Cx * Cx - 1.0;
		std::cout << "Cx = " << Cx << std::endl;

		// Computing over parts of the array :
		C.column(0) = C.column(2) - C.column(1);
		std::cout << "C = " << C << std::endl;
		
		// Explicit manipulations :
		Kartet::Array<int> block(3, 3);
		block = Kartet::cast<float>(Kartet::IndexI()!=1 && Kartet::IndexJ()!=1);
		Kartet::Array<int> large(3*block.numRows(), 3*block.numColumns());
		large = repeat(block);
		std::cout << "Block = " << block << std::endl;
		std::cout << "Large (repeat block) = " << large << std::endl;
		large = expand(block);
		std::cout << "Large (expand block) = " << large << std::endl;

		// Random sources :
		Kartet::UniformSource<> uniformSource;
		uniformSource.setSeed();
		uniformSource >> A;
		uniformSource >> B;
		std::cout << "A = " << A << std::endl;
		std::cout << "B = " << B << std::endl;
		B = yFlip(xFlip(A)); // We cannot use in place flip here, these operations will be slow when performed on device for large arrays (naive/non-coalesced memory access).
		std::cout << "A (flipped) = " << B << std::endl;

		// Reduction example :
		Kartet::ReduceContext reduceContext;
		const float mA = reduceContext.mean(A);
		std::cout << "Mean of A : " << mA << std::endl;
		const float count1 = reduceContext.sum(A.layout(), Kartet::cast<int>(A>=B));
		std::cout << "Number of elements of A larger than B : " << count1 << std::endl;
		
		const float sum1 = reduceContext.sum(A.layout(), abs(A-B));
		std::cout << "Sum over |A-B| : " << sum1 << std::endl;

		// BLAS Example :
		const Kartet::Layout matrixLayout(4,4);
		Kartet::Array<double> M1(matrixLayout), M2(matrixLayout), M3(matrixLayout);
		M1 = Kartet::IndexI();
		M2 = (Kartet::IndexI() + Kartet::IndexJ())/2.0;
		M3 = 0.0;

		Kartet::BLASContext blas;
		blas.gemm(M1,M2,M3);
		std::cout << "M1 = " << M1 << std::endl;
		std::cout << "M2 = " << M2 << std::endl;
		std::cout << "M3 = " << M3 << std::endl;
	}
	catch(const Kartet::Exception& e)
	{
		std::cerr << "Exception : " << e << std::endl;
		returnCode = -1;
	}

	return returnCode;
}

