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

/**
	\file    GetArgs.hpp
	\brief   Mini arguments tools.
	\author  R. Kerviche
	\date    January 1st 2017
**/

#ifndef __GET_ARGS__
#define __GET_ARGS__

	#include <map>
	#include <string>

/**
\brief Parse arguments to values.
\param argc Number of arguments to be parsed.
\param argv Arguments.
\param arguments List of integer arguments to be collected.
\param argumentsHelp Help of the arguments to be collected.

Example :
\code
int 	deviceId = 0,
	numThreads = 512,
	maxBlockRepetition = 64;;
std::map<const char, int*> arguments;
std::map<const char, std::string> argumentsHelp;
arguments['d'] = &deviceId;
argumentsHelp['d'] = "Target device Id, per nvidia-smi indexing.";
arguments['t'] = &numThreads;
argumentsHelp['t'] = "Number of threads per block (~512).";
arguments['r'] = &maxBlockRepetition;
argumentsHelp['r'] = "Number of block repetition (~64).";
if(!getArgs(argc, argv, arguments, argumentsHelp))
	throw Kartet::InvalidArgument;
\endcode
**/
	bool getArgs(const int& argc, char const* const* argv, const std::map<const char, int*>& arguments, const std::map<const char, std::string>& argumentsHelp=std::map<const char, std::string>());
#endif

