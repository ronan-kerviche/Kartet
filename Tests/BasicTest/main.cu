#include <iostream>
#include "Kartet.hpp"

int main(int argc, char** argv)
{
	std::cout << "Kartet Test" << std::endl;
	std::cout << "Build : " << __DATE__ << ' ' << __TIME__ << std::endl;

	try
	{
		Kartet::Array<double> A(5, 3), B(5, 3);
		A = 1.23;
		B = 2.56 * Kartet::IndexJ();
		A = cos(A + exp(2.0 * B));
		double* tmp = A.getData();
		std::cout << "Result : " << std::endl;
		for(int k=0; k<A.getNumElements()-1; k++)
			std::cout << tmp[k] << ", ";
		std::cout << tmp[A.getNumElements()-1] << std::endl;
		delete[] tmp;
		tmp = NULL;

		Kartet::Array<double> C(512);
		C = Kartet::IndexI();
		tmp = C.getData();
		std::cout << "Result : " << std::endl;
		for(int k=0; k<C.getNumElements()-1; k++)
			std::cout << tmp[k] << ", ";
		std::cout << tmp[C.getNumElements()-1] << std::endl;

		C = Kartet::Cast<unsigned char>( Kartet::IndexI() );
		C.getData(tmp);
		std::cout << "Result : " << std::endl;
		for(int k=0; k<C.getNumElements()-1; k++)
			std::cout << tmp[k] << ", ";
		std::cout << tmp[C.getNumElements()-1] << std::endl;
		delete[] tmp;
		tmp = NULL;
	}
	catch(Kartet::Exception& e)
	{
		std::cout << "Exception : " << e << std::endl;
	}
	
	return 0;
}
