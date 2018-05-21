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
	\file    Function.hpp
	\brief   Functions tools.
	\author  R. Kerviche
	\date    September 12th 2017
**/

#ifndef __KARTET_FUNCTION_TOOLS__
#define __KARTET_FUNCTION_TOOLS__

	#include "FunctionSupportMacros.hpp"
	#define KARTET_FUNCTION_COMPUTE_LAYOUT COMPUTE_LAYOUT

	#ifdef __CUDACC__
		// Standard :
		#define KARTET_FUNCTION(functionName, layout, idx, jdx, kdx, ...) \
			__host__ __device__ void functionName##__kernel (const Layout& layout, const index_t& idx, const index_t& jdx, const index_t& kdx, ARGLIST(const, &, __VA_ARGS__)); \
			 \
			__global__ void functionName##__global(const Layout layout, ARGLIST(const, , __VA_ARGS__)) \
			{ \
				functionName##__kernel (layout, layout.getI(), layout.getJ(), layout.getK(), VARLIST(__VA_ARGS__)); \
			} \
			 \
			template<Location location> \
			void functionName (const Layout& layout, ARGLIST(const, &, __VA_ARGS__)) \
			{ \
				if(location==DeviceSide) \
				{ \
					functionName##__global KARTET_FUNCTION_COMPUTE_LAYOUT(layout) (layout, VARLIST(__VA_ARGS__)); \
					cudaError_t err = cudaGetLastError(); \
					if(err!=cudaSuccess) \
						throw static_cast<Exception>(CudaExceptionsOffset + err); \
				} \
				else \
				{ \
					const index_t 	R = layout.numRows(), \
							Z = layout.numColumns() * layout.numSlices(); \
					OMP_PARALLEL_STATIC \
					for(index_t q=0; q<Z; q++) \
					{ \
						index_t dummy, jdx, kdx; \
						layout.unpackIndex(q*R, dummy, jdx, kdx); \
						for(index_t idx=0; idx<R; idx++) \
							functionName##__kernel (layout, idx, jdx, kdx, VARLIST(__VA_ARGS__)); \
					} \
				} \
			} \
			 \
			__host__ __device__ void functionName##__kernel (const Layout& layout, const index_t& idx, const index_t& jdx, const index_t& kdx, ARGLIST(const, &, __VA_ARGS__))

		// Template :
		#define KARTET_TEMPLATE_FUNCTION(functionName, templates, layout, idx, jdx, kdx, ...) \
			template<Location location, MACRO_EXPAND(MACRO_BREAK, templates) > \
			__host__ __device__ void functionName##__kernel (const Layout& layout, const index_t& idx, const index_t& jdx, const index_t& kdx, ARGLIST(const, &, __VA_ARGS__)); \
			 \
			template<Location location, MACRO_EXPAND(MACRO_BREAK, templates) > \
			__global__ void functionName##__global(const Layout layout, ARGLIST(const, , __VA_ARGS__)) \
			{ \
				functionName##__kernel (layout, layout.getI(), layout.getJ(), layout.getK(), VARLIST(__VA_ARGS__)); \
			} \
			 \
			template<Location location, MACRO_EXPAND(MACRO_BREAK, templates) > \
			void functionName (const Layout& layout, ARGLIST(const, &, __VA_ARGS__)) \
			{ \
				if(location==DeviceSide) \
				{ \
					functionName##__global KARTET_FUNCTION_COMPUTE_LAYOUT(layout) (layout, VARLIST(__VA_ARGS__)); \
					cudaError_t err = cudaGetLastError(); \
					if(err!=cudaSuccess) \
						throw static_cast<Exception>(CudaExceptionsOffset + err); \
				} \
				else \
				{ \
					const index_t 	R = layout.numRows(), \
							Z = layout.numColumns() * layout.numSlices(); \
					OMP_PARALLEL_STATIC \
					for(index_t q=0; q<Z; q++) \
					{ \
						index_t dummy, jdx, kdx; \
						layout.unpackIndex(q*R, dummy, jdx, kdx); \
						for(index_t idx=0; idx<R; idx++) \
							functionName##__kernel (layout, idx, jdx, kdx, VARLIST(__VA_ARGS__)); \
					} \
				} \
			} \
			 \
			template<Location location, MACRO_EXPAND(MACRO_BREAK, templates) > \
			__host__ __device__ void functionName##__kernel (const Layout& layout, const index_t& idx, const index_t& jdx, const index_t& kdx, ARGLIST(const, &, __VA_ARGS__))

	#else
		// Standard :
		#define KARTET_FUNCTION(functionName, layout, idx, jdx, kdx, ...) \
			__host__ __device__ void functionName##__kernel (const Layout& layout, const index_t& idx, const index_t& jdx, const index_t& kdx, ARGLIST(const, &, __VA_ARGS__)); \
			 \
			template<Location location> \
			void functionName (const Layout& layout, ARGLIST(const, &, __VA_ARGS__)) \
			{ \
				if(location==DeviceSide) \
					throw NotSupported; \
				else \
				{ \
					const index_t 	R = layout.numRows(), \
							Z = layout.numColumns() * layout.numSlices(); \
					OMP_PARALLEL_STATIC \
					for(index_t q=0; q<Z; q++) \
					{ \
						index_t dummy, jdx, kdx; \
						layout.unpackIndex(q*R, dummy, jdx, kdx); \
						for(index_t idx=0; idx<R; idx++) \
							functionName##__kernel (layout, idx, jdx, kdx, VARLIST(__VA_ARGS__)); \
					} \
				} \
			} \
			 \
			__host__ __device__ void functionName##__kernel (const Layout& layout, const index_t& idx, const index_t& jdx, const index_t& kdx, ARGLIST(const, &, __VA_ARGS__))

		// Template :
		#define KARTET_TEMPLATE_FUNCTION(functionName, templates, layout, idx, jdx, kdx, ...) \
			template<Location location, MACRO_EXPAND(MACRO_BREAK, templates) > \
			__host__ __device__ void functionName##__kernel (const Layout& layout, const index_t& idx, const index_t& jdx, const index_t& kdx, ARGLIST(const, &, __VA_ARGS__)); \
			 \
			template<Location location, MACRO_EXPAND(MACRO_BREAK, templates) > \
			void functionName (const Layout& layout, ARGLIST(const, &, __VA_ARGS__)) \
			{ \
				if(location==DeviceSide) \
					throw NotSupported; \
				else \
				{ \
					const index_t 	R = layout.numRows(), \
							Z = layout.numColumns() * layout.numSlices(); \
					OMP_PARALLEL_STATIC \
					for(index_t q=0; q<Z; q++) \
					{ \
						index_t dummy, jdx, kdx; \
						layout.unpackIndex(q*R, dummy, jdx, kdx); \
						for(index_t idx=0; idx<R; idx++) \
							functionName##__kernel (layout, idx, jdx, kdx, VARLIST(__VA_ARGS__)); \
					} \
				} \
			} \
			 \
			template<Location location, MACRO_EXPAND(MACRO_BREAK, templates) > \
			__host__ __device__ void functionName##__kernel (const Layout& layout, const index_t& idx, const index_t& jdx, const index_t& kdx, ARGLIST(const, &, __VA_ARGS__))
	#endif

/**
\ingroup FunctionsGroup
\def KARTET_FUNCTION
\brief Define a variadic function which can run either on host or device (see Kartet::Location).
\param functionName Name of the function.
\param layout Name of the layout variable.
\param idx Name of the I index variable.
\param jdx Name of the J index variable.
\param kdx Name of the K index variable.
\param ... List of parameters as parentheses-delimited groups of types and variables.

The function cannot be used in a template expression. However, it provides a simple mechanism to write domain-agnostic code.

Define a function :
\code
// Always to be protected inside the Kartet namespace :
namespace Kartet
{
	// Define the function, the first lists the indexing parameters :
	KARTET_FUNCTION(myFunction, i, j, k, layout,
	// The second line lists the input parameters with the following constraints :
	// - the types and the variables are grouped inside parentheses.
	// - the types can be fixed templates.
	// - the type should be the base type or a pointer, the referencing mechanism is handled by the macro.
	// - replace Array type arguments by the corresponding Accessor types.
		(Accessor<float>, a), (MyObject, obj))
	{
		// The body of the function :
		UNUSED_PARAMETER(k)
		UNUSED_PARAMETER(layout)
		if(a.isInside(i,j))
			a.data(i,j) = obj.performSomeConstantOperation(i, j, a.data(i,j));
	}
}
\endcode

Use the previous function :
\code
using namespace Kartet;
// ...

Array<float> a(128, 128);
Array<double> b(128, 128);
MyDataStructure myData;
// ...

// Default execution, might be executed on device or host depending on the default domain :
myFunction<KARTET_DEFAULT_LOCATION>(a.layout(), a, myData);
\endcode
**/

/**
\ingroup FunctionsGroup
\def KARTET_TEMPLATE_FUNCTION
\brief Define a variadic function which can run either on host or device (see Kartet::Location).
\param functionName Name of the function.
\param templates List of the templates, contained in parentheses.
\param layout Name of the layout variable.
\param idx Name of the I index variable.
\param jdx Name of the J index variable.
\param kdx Name of the K index variable.
\param ... List of parameters as parentheses-delimited groups of types and variables.

The function cannot be used in a template expression. However, it provides a simple mechanism to write domain-agnostic code.

Define a template function :
\code
// Always to be protected inside the Kartet namespace :
namespace Kartet
{
	// To define template functions use :
	KARTET_TEMPLATE_FUNCTION(myTemplateFunction, 
		(typename TA, typename TB)
		i, j, k, layout,
	// - "location" is a special template parameter which is already defined by the macro and can be infered from the arguments via :
		(Accessor<TA, location>, a), (Accessor<TB, location>, b))
	{
		UNUSED_PARAMETER(k)
		UNUSED_PARAMETER(layout)
		if(a.isInside(i,j))
			a.data(i,j) = static_cast<T>(i+j)*b.data(i,j);
	}
}
\endcode

Use the previous function :
\code
using namespace Kartet;
// ...

Array<float> a(128, 128);
Array<double> b(128, 128);
MyDataStructure myData;
// ...

// Infer the location from the arguments :
myTemplateFunction(a.layout(), a, b);
\endcode
**/

#endif

