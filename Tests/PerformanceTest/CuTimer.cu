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

#include "CuTimer.hpp"

	CuTimer::CuTimer(void)
	 : elapsedtime_ms(0.0f)
	{
		cudaEventCreate(&startEvent);
		cudaEventCreate(&stopEvent); 
	}

	CuTimer::~CuTimer(void)
	{
		cudaEventDestroy(startEvent);
		cudaEventDestroy(stopEvent);
	}

	void CuTimer::start(void)
	{
		cudaEventRecord(startEvent, 0);
	}

	void CuTimer::stop(void)
	{
		cudaDeviceSynchronize();
		cudaEventRecord(stopEvent, 0);
		cudaEventSynchronize(stopEvent); 
		cudaEventElapsedTime(&elapsedtime_ms, startEvent, stopEvent);
	}
	
	float CuTimer::getElapsedTime_s(void) const
	{
		return elapsedtime_ms*1e-3;
	}

