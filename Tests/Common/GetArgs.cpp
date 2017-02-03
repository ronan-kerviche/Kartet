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
#include <sstream>
#include "GetArgs.hpp"

	bool getArgs(const int& argc, char const* const* argv, const std::map<const char, int*>& arguments, const std::map<const char, std::string>& argumentsHelp)
	{
		for(int k=1; k<argc; k++)
		{
			const std::string arg = argv[k];
			if(arg[0]!='-' || arg.size()!=2)
			{
				std::cerr << "Unknown argument : " << arg << std::endl;
				return false;
			}
			if(arg[1]=='h')
			{
				std::cout << "Help : " << std::endl;
				std::cout << argv[0] << " [Arguments...]" << std::endl;
				if(!argumentsHelp.empty())
				{
					std::cout << "Arguments : " << std::endl;
					for(std::map<const char, std::string>::const_iterator it=argumentsHelp.begin(); it!=argumentsHelp.end(); it++)
						std::cout << "  -" << it->first << "  : " << it->second << std::endl;
				}
				return false;
			}
			std::map<const char, int*>::const_iterator it=arguments.find(arg[1]);
			if(it==arguments.end())
			{
				std::cerr << "Invalid argument : " << arg << std::endl;
				return false;
			}
			k++;
			if(k>=argc)
			{
				std::cerr << "Missing value for argument : " << arg << std::endl;
				return false;
			}
			if(it->second!=NULL)
			{
				const std::string value = argv[k];
				std::istringstream iss(value);
				if(!static_cast<bool>(iss >> (*it->second)))
				{
					std::cerr << "Could not convert value of argument " << arg << " : " << value << std::endl;
					return false;
				}
			}
		}
		return true;
	}

