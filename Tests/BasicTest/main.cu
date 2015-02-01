#include <iostream>
#include "Kartet.hpp"

int main(int argc, char** argv)
{
	int returnCode = 0;
	std::cout << "Kartet Test" << std::endl;
	std::cout << "Build : " << __DATE__ << ' ' << __TIME__ << std::endl;

	cuFloatComplex c = Kartet::toComplex<float>(1.0);

	try
	{
		Kartet::Array<double> A(5, 3), B(5, 3);
		A = 1.23;
		B = 2.56 * Kartet::IndexJ();
		A = cos(A + exp(2.0 * B));
		std::cout << A << std::endl;

		Kartet::Array<double> C(16);
		C = Kartet::IndexI();
		std::cout << C << std::endl;

		C = Kartet::cast<unsigned char>( Kartet::IndexI()*64 );
		std::cout << C << std::endl;

		Kartet::BLAS Context;
		C = 4 - absSq(Kartet::IndexI()-13) / 100.0;
		std::cout << C << std::endl;
		int idx = Kartet::amax(C);
		std::cout << "Index of the absolute maximum : " << idx << std::endl;
		std::cout << std::endl;

		const int w1 = 256, w2 = 512, w3 = 348;
		Kartet::Array<float> X(w1, w3), Y(w3, w2), Z(w1, w2);
		X = Kartet::IndexI();
		Y = Kartet::IndexJ();
		Z = 0.0;
		//gemm(1.0, X, CUBLAS_OP_N, Y, CUBLAS_OP_N, 0.0, Z);
		gemm(X, Y, Z);

		Kartet::RandomSourceContext randomSourceContext;
		Kartet::UniformSource uniformSource;
		uniformSource >> A;
		std::cout << A << std::endl;

		A = Kartet::IndexI() + Kartet::IndexJ();
		// Select the vectors from 0 to 2 with a step of 2 (1st and 3rd vectors).
		Kartet::Accessor<double> S = A.vectors(0, 2, 2);
		uniformSource >> S;
		std::cout << "S is : " << S << std::endl;
		std::cout << "A is : " << A << std::endl;

		Kartet::Accessor<double> T = A.subArray(1,0,3,3);
		std::cout << T << std::endl;
		T = -23.0;
		std::cout << A << std::endl;

		Kartet::Array<float> D(4, 4, 3);
		Kartet::NormalSource normalSource(2.0, 10.0);
		normalSource >> D;
		std::cout << D << std::endl;
		Kartet::Accessor<float> E = D.vectors(0, 2, 2);
		E = 0;
		std::cout << D << std::endl;
		Kartet::Accessor<float> F = E.slice(1);
		F = 1.0;
		std::cout << D << std::endl;

		D = real(piAngleToComplex(Kartet::IndexI() + Kartet::IndexJ()));
		std::cout << D << std::endl;
		D.slice(0) = D.slice(0)*D.slice(1);
		std::cout << D.slice(0) << std::endl;
		D.slice(1) = (D.slice(1) + D.slice(0))/2.0;
		std::cout << D << std::endl;
	}
	catch(Kartet::Exception& e)
	{
		std::cout << "Exception : " << e << std::endl;
		returnCode = -1;
	}	
	return returnCode;
}

