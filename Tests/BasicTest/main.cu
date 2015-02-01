#include <iostream>
#include "Kartet.hpp"

template<typename T>
void print(const Kartet::Accessor<T>& a)
{
	Kartet::Layout l = a.getSolidLayout();
	double* tmp = a.getData();
	std::cout << "Array (" << a.getNumRows() << ", " << a.getNumColumns() << ", " << a.getNumSlices() << ") : " << std::endl;

	for(int k=0; k<l.getNumSlices(); k++)
	{
		if(l.getNumSlices()>1)
			std::cout << "Slice " << k << std::endl;
	
		for(int i=0; i<l.getNumRows(); i++)
		{
			for(int j=0; j<(l.getNumColumns()-1); j++)
				std::cout << tmp[l.getIndex(i,j,k)] << ", ";
			std::cout << tmp[l.getIndex(i,l.getNumColumns()-1,k)] << std::endl;
		}
		std::cout << std::endl;
	}

	delete[] tmp;
}

int main(int argc, char** argv)
{
	std::cout << "Kartet Test" << std::endl;
	std::cout << "Build : " << __DATE__ << ' ' << __TIME__ << std::endl;

	cuFloatComplex c = Kartet::toComplex<float>(1.0);

	try
	{
		Kartet::Array<double> A(5, 3), B(5, 3);
		A = 1.23;
		B = 2.56 * Kartet::IndexJ();
		A = cos(A + exp(2.0 * B));
		print(A);

		Kartet::Array<double> C(16);
		C = Kartet::IndexI();
		print(C);	

		C = Kartet::Cast<unsigned char>( Kartet::IndexI()*64 );
		print(C);

		Kartet::BLAS Context;
		C = 1.0f / square( Kartet::IndexI()-13 );
		print(C);
		int idx = Kartet::amax(C);
		std::cout << "Index of the maximum : " << idx << std::endl;
		std::cout << std::endl;

		const int w1 = 256, w2 = 512, w3 = 348;
		Kartet::Array<float> X(w1, w3), Y(w3, w2), Z(w1, w2);
		X = Kartet::IndexI();
		Y = Kartet::IndexJ();
		Z = 0.0;
		gemm(1.0, X, CUBLAS_OP_N, Y, CUBLAS_OP_N, 0.0, Z);

		Kartet::RandomSourceContext randomSourceContext;
		Kartet::UniformSource uniformSource;
		uniformSource >> A;
		print(A);

		A = Kartet::IndexI() + Kartet::IndexJ();
		// Select the vectors from 0 to 2 in slice 0, with a step of 2 (1st and 3rd vectors).
		Kartet::Accessor<double> S = A.vectors(0,2,0,2);
		std::cout << S.getNumRows() << 'x' << S.getNumColumns() << 'x' << S.getNumSlices() << ';' << S.getLeadingColumns() << std::endl;
		uniformSource >> S;
		print(S);
		print(A);

		Kartet::Accessor<double> T = A.subArray(1,0,3,2);
		print(T);
		T = -23.0;
		print(A);
	}
	catch(Kartet::Exception& e)
	{
		std::cout << "Exception : " << e << std::endl;
	}
	
	return 0;
}
