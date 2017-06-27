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
	\file    Meta.hpp
	\brief   Meta tools.
	\author  R. Kerviche
	\date    November 1st 2009
**/

#ifndef __KARTET_META__
#define __KARTET_META__

namespace Kartet
{
	// Null type :
	struct Void
	{ };

	// Static assertion :
	template<bool test>
	struct StaticAssert;

	template<>
	struct StaticAssert<true>
	{ };

	// Static assertion with messages :
	template<bool test>
	struct StaticAssertVerbose;

	template<>
	struct StaticAssertVerbose<true>
	{
		enum
		{
			KARTET_STATIC_ERROR___RHS_IS_COMPLEX_BUT_LHS_IS_NOT,
			KARTET_STATIC_ERROR___LHS_AND_RHS_HAVE_INCOMPATIBLE_LOCATIONS,
			KARTET_STATIC_ERROR___ARGUMENTS_HAVE_INCOMPATIBLE_LOCATIONS,
			KARTET_STATIC_ERROR___STATIC_CONTAINER_MUST_HAVE_VOID_PARAMETER,
			KARTET_STATIC_ERROR___INVALID_LOCATION,
			KARTET_STATIC_ERROR___TYPE_NOT_SUPPORTED,
			KARTET_STATIC_ERROR___TYPE_MUST_BE_REAL,
			KARTET_STATIC_ERROR___TYPE_MUST_BE_COMPLEX,
			KARTET_STATIC_ERROR___LOCATION_NOT_SUPPORTED,
			KARTET_STATIC_ERROR___UNSUPPORTED_TYPE_SIZE,
			KARTET_STATIC_ERROR___INVALID_ARGUMENTS_LAYOUT,
			KARTET_STATIC_ERROR___ARGUMENTS_CONFLICT,
			KARTET_STATIC_ERROR___INVALID_DIMENSION
		};
	};
	
	template<>
	struct StaticAssertVerbose<false>
	{ };

	template<int>
	struct StaticAssertCapsule
	{ };

	#define _JOIN(x, y) x ## y
	#define JOIN(x, y) _JOIN(x, y)
	#define STATIC_ASSERT_VERBOSE(test, message) typedef Kartet::StaticAssertCapsule<Kartet::StaticAssertVerbose<(test)>::KARTET_STATIC_ERROR___##message> JOIN(StaticAssertTest, __LINE__);

	// Static tests :
	template<bool test, typename TrueResult, typename FalseResult>
	struct StaticIf;

	template<typename TrueResult, typename FalseResult>
	struct StaticIf<true, TrueResult, FalseResult>
	{
		typedef TrueResult Type;
	};

	template<typename TrueResult, typename FalseResult>
	struct StaticIf<false, TrueResult, FalseResult>
	{
		typedef FalseResult Type;
	};

	// Type test :
	template<typename T1, typename T2>
	struct IsSame
	{
		static const bool value = false;
	};

	template<typename T>
	struct IsSame<T,T>
	{
		static const bool value = true;
	};

		// Loop Unrolling via metaprogramming :
	template<int d, int k=0>
	struct ForLoop
	{
		template<typename TOp>
		__host__ __device__ static inline void run(TOp& op)
		{
			op.apply(k);
			ForLoop<d, k+1>::run(op);
		}

		template<typename TOp>
		__host__ __device__ static inline typename TOp::TResult sum(TOp& op)
		{
			return op.applySoft(k) + ForLoop<d, k+1>::sum(op);
		}
	};

	template<int d>
	struct ForLoop<d, d> // Stop condition.
	{
		template<typename TOp>
		__host__ __device__ static inline void run(TOp&)
		{ }

		template<typename TOp>
		__host__ __device__ static inline typename TOp::TResult sum(TOp&)
		{
			return static_cast<typename TOp::TResult>(0);
		}
	};

	#define META_UNARY_OPERATOR(NAME, ...) \
		template<typename B, typename A> \
		struct MetaUnary##NAME \
		{ \
			typedef B TResult; \
			B	*pb; \
			const A	*pa; \
			const int bStride, \
				  aStride; \
			__host__ __device__ inline MetaUnary##NAME(B* _pb, const A* _pa, const int _bStride=1, const int _aStride=1) \
			 : 	pb(_pb), \
				pa(_pa), \
				bStride(_bStride), \
				aStride(_aStride) \
			{ } \
			__host__ __device__ inline void apply(const int& k) \
			{ \
				pb[k*bStride] = operate(pa[k*aStride]); \
			} \
			__host__ __device__ inline B applySoft(const int& k) \
			{ \
				return operate(pa[k*aStride]); \
			} \
			__host__ __device__ static inline B operate(const A& a) \
			{ \
				return static_cast<B>(__VA_ARGS__); \
			} \
		}; \
		template<typename B, typename A> \
		struct MetaUnary##NAME##Right \
		{ \
			typedef B TResult; \
			B	*pb; \
			const A	pa; \
			__host__ __device__ inline MetaUnary##NAME##Right(B* _pb, const A& _pa) \
			 : 	pb(_pb), \
				pa(_pa) \
			{ } \
			__host__ __device__ inline void apply(const int& k) \
			{ \
				pb[k] = operate(pa); \
			} \
			__host__ __device__ inline B applySoft(const int&) \
			{ \
				return operate(pa); \
			} \
			__host__ __device__ static inline B operate(const A& a) \
			{ \
				return static_cast<B>(__VA_ARGS__); \
			} \
		}; \
		template<int d, typename B, typename A> \
		__host__ __device__ inline void metaUnary##NAME(B* pb, const A* pa) \
		{ \
			MetaUnary##NAME<B, A> op(pb, pa); \
			ForLoop<d>::run(op); \
		} \
		template<int d, typename B, typename A> \
		__host__ __device__ inline void metaUnary##NAME(B* pb, const A& pa) \
		{ \
			MetaUnary##NAME##Right<B, A> op(pb, pa); \
			ForLoop<d>::run(op); \
		} \
		template<int d, typename A> \
		__host__ __device__ inline typename MetaUnary##NAME<A, A>::TResult metaUnary##NAME##Sum(const A* pa, const int& aStride=1) \
		{ \
			MetaUnary##NAME<A, A> op(NULL, pa, 1, aStride); \
			return ForLoop<d>::sum(op); \
		}

	#define META_BINARY_OPERATOR(NAME, ...) \
		template<typename C, typename A, typename B> \
		struct MetaBinary##NAME \
		{ \
			typedef C TResult; \
			C	*pc; \
			const A	*pa; \
			const B	*pb; \
			const int cStride, \
				  aStride, \
				  bStride; \
			__host__ __device__ inline MetaBinary##NAME(C* _pc, const A* _pa, const B* _pb, const int& _cStride=1, const int& _aStride=1, const int& _bStride=1) \
			 : 	pc(_pc), \
				pa(_pa), \
				pb(_pb), \
				cStride(_cStride), \
				aStride(_aStride), \
				bStride(_bStride) \
			{ } \
			__host__ __device__ inline void apply(const int& k) \
			{ \
				pc[k*cStride] = operate(pa[k*aStride], pb[k*bStride]); \
			} \
			__host__ __device__ inline C applySoft(const int& k) \
			{ \
				return operate(pa[k*aStride], pb[k*bStride]); \
			} \
			__host__ __device__ static inline C operate(const A& a, const B& b) \
			{ \
				return static_cast<C>(__VA_ARGS__); \
			} \
		}; \
		template<typename C, typename A, typename B> \
		struct MetaBinary##NAME##Left \
		{ \
			typedef C TResult; \
			C	*pc; \
			const A	pa; \
			const B	*pb; \
			__host__ __device__ inline MetaBinary##NAME##Left(C* _pc, const A& _pa, const B* _pb) \
			 : 	pc(_pc), \
				pa(_pa), \
				pb(_pb) \
			{ } \
			__host__ __device__ inline void apply(const int& k) \
			{ \
				pc[k] = operate(pa, pb[k]); \
			} \
			__host__ __device__ inline C applySoft(const int& k) \
			{ \
				return operate(pa, pb[k]); \
			} \
			__host__ __device__ static inline C operate(const A& a, const B& b) \
			{ \
				return static_cast<C>(__VA_ARGS__); \
			} \
		}; \
		template<typename C, typename A, typename B> \
		struct MetaBinary##NAME##Right \
		{ \
			typedef C TResult; \
			C	*pc; \
			const A	*pa; \
			const B	pb; \
			__host__ __device__ inline MetaBinary##NAME##Right(C* _pc, const A* _pa, const B& _pb) \
			 : 	pc(_pc), \
				pa(_pa), \
				pb(_pb) \
			{ } \
			__host__ __device__ inline void apply(const int& k) \
			{ \
				pc[k] = operate(pa[k], pb); \
			} \
			__host__ __device__ inline C applySoft(const int& k) \
			{ \
				return operate(pa[k], pb); \
			} \
			__host__ __device__ static inline C operate(const A& a, const B& b) \
			{ \
				return static_cast<C>(__VA_ARGS__); \
			} \
		}; \
		template<int d, typename C, typename A, typename B> \
		__host__ __device__ inline void metaBinary##NAME(C* pc, const A* pa, const B* pb) \
		{ \
			MetaBinary##NAME<C, A, B> op(pc, pa, pb); \
			ForLoop<d>::run(op); \
		} \
		template<int d, typename C, typename A, typename B> \
		__host__ __device__ inline void metaBinary##NAME(C* pc, const A& pa, const B* pb) \
		{ \
			MetaBinary##NAME##Left<C, A, B> op(pc, pa, pb); \
			ForLoop<d>::run(op); \
		} \
		template<int d, typename C, typename A, typename B> \
		__host__ __device__ inline void metaBinary##NAME(C* pc, const A* pa, const B& pb) \
		{ \
			MetaBinary##NAME##Right<C, A, B> op(pc, pa, pb); \
			ForLoop<d>::run(op); \
		} \
		template<int d, typename C, typename A, typename B> \
		__host__ __device__ inline typename MetaBinary##NAME<C, A, B>::TResult metaBinary##NAME##Sum(const A* pa, const B* pb, const int& aStride=1, const int& bStride=1) \
		{ \
			MetaBinary##NAME<C, A, B> op(NULL, pa, pb, 1, aStride, bStride); \
			return ForLoop<d>::sum(op); \
		}

	META_UNARY_OPERATOR(Equal, a)
	META_UNARY_OPERATOR(Plus, a)
	META_UNARY_OPERATOR(Minus, -a)
	META_UNARY_OPERATOR(Square, a*a)
	META_BINARY_OPERATOR(Plus, a+b)
	META_BINARY_OPERATOR(Minus, a-b)
	META_BINARY_OPERATOR(Product, a*b)
	META_BINARY_OPERATOR(Quotient, a/b)
	// Other operators can be added thereafter.
}

#endif

