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

/**
	\file    TemplateSharedMemory.hpp
	\brief   Shared memory access.
	\author  R. Kerviche
	\date    November 1st 2009
**/

#ifndef __KARTET_TEMPLATE_SHARED_MEMORY__
#define __KARTET_TEMPLATE_SHARED_MEMORY__

namespace Kartet
{
// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
	template<class T>
	struct SharedMemory
	{
		__device__ inline operator T*()
		{
			extern __shared__ int __smem[];
			return reinterpret_cast<T*>(__smem);
		}

		__device__ inline operator const T*() const
		{
			extern __shared__ int __smem[];
			return reinterpret_cast<T*>(__smem);
		}
	};

// MACRO-template : 
	#define SharedMemoryInterface(type, sufix) \
		template<> \
		struct SharedMemory<type> \
		{ \
			__device__ inline operator type*() \
			{ \
				extern __shared__ type __smem_##sufix []; \
				return (type*)__smem_##sufix ; \
			} \
			 \
			__device__ inline operator const type*() const \
			{ \
				extern __shared__ type __smem_##sufix []; \
				return (type*)__smem_##sufix ; \
			} \
		};

// For the types : 
	SharedMemoryInterface(int, 		i)
	SharedMemoryInterface(unsigned int, 	ui)
	SharedMemoryInterface(short,		s)
	SharedMemoryInterface(unsigned short,	us)
	SharedMemoryInterface(long,		l)
	SharedMemoryInterface(unsigned long,	ul)
	SharedMemoryInterface(float, 		f)
	SharedMemoryInterface(double,		d)

	#undef SharedMemoryInterface

} // namespace Kartet

#endif

