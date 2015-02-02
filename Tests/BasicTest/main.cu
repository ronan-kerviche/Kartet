#include <iostream>
#include "Kartet.hpp"

int main(int argc, char** argv)
{
	int returnCode = 0;
	std::cout << "Kartet Test" << std::endl;
	std::cout << "Build : " << __DATE__ << ' ' << __TIME__ << std::endl;

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

		C = Kartet::cast<unsigned char>(Kartet::IndexI()*64);
		std::cout << C << std::endl;

		Kartet::BLASContext Context;
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
		// Select 2 vectors starting at 0 with a step of 2 (1st and 3rd vectors).
		Kartet::Accessor<double> S = A.vectors(0, 2, 2);
		uniformSource >> S;
		std::cout << "S = " << S << std::endl;
		std::cout << "A = " << A << std::endl;

		A.subArray(1,0,3,3) = -23.0;
		std::cout << A << std::endl;

		Kartet::Array<float> D(4, 4, 3);
		Kartet::NormalSource normalSource(2.0, 10.0);
		normalSource >> D;
		std::cout << D << std::endl;
		D.vectors(0, 2, 2) = 0;
		std::cout << D << std::endl;
		D.vectors(0, 2, 2).slice(1) = 1.0;
		std::cout << D << std::endl;

		D = real(piAngleToComplex(Kartet::IndexI() + Kartet::IndexJ()));
		std::cout << D << std::endl;
		D.slice(0) = D.slice(0)*D.slice(1);
		std::cout << D.slice(0) << std::endl;
		D.slice(1) = (D.slice(1) + D.slice(0))/2.0;
		std::cout << D << std::endl;

		Kartet::Array<cuDoubleComplex> CxA(4, 4);
		CxA = 1.0 + Kartet::IndexJ();
		CxA = angleToComplex(real(CxA));
		std::cout << CxA << std::endl;
		Kartet::Array<double> CxAbs(CxA.layout());
		CxAbs = abs(CxA) - real(CxA);
		std::cout << CxAbs << std::endl;
		CxAbs = angle(CxA);
		std::cout << CxAbs << std::endl;
	}
	catch(Kartet::Exception& e)
	{
		std::cout << "Exception : " << e << std::endl;
		returnCode = -1;
	}	
	return returnCode;
}

