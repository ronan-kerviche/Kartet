/* ************************************************************************************************************* */
/*                                                                                                               */
/*     Kartet                                                                                                    */
/*     A Simple C++ Array Library for CUDA                                                                       */
/*                                                                                                               */
/*     LICENSE : The MIT License                                                                                 */
/*     Copyright (c) 2015-2017 Ronan Kerviche                                                                    */
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

	if(argc!=2)
	{
		std::cerr << "Error : missing argument." << std::endl;
		std::cerr << argv[0] << " filename" << std::endl;
	}
	else
	{
		try
		{
			std::ifstream file(argv[1], std::ios::in | std::ios::binary);
			bool isComplex = true;
			if(!file.is_open())
			{
				std::cerr << "Cannot open file : " << argv[1] << "." << std::endl;
				return -1;
			}
			file.seekg(0);
			Kartet::Layout::readFromStream(file, NULL, &isComplex);
			file.seekg(0);

			if(isComplex)
			{
				Kartet::Array<Kartet::Complex<double>, Kartet::HostSide> A(file);
				std::cout << A;
			}
			else
			{
				Kartet::Array<double, Kartet::HostSide> A(file);
				std::cout << A; 
			}
			return 0;
		}
		catch(Kartet::Exception& e)
		{
			std::cerr << "Error : exception caught : " << e << std::endl;
			return -static_cast<int>(e);
		}
	}
}

