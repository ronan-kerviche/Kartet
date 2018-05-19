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
	\file    Vec.hpp
	\brief   Vector maths tools.
	\author  R. Kerviche
	\date    May 13th 2017
**/

#ifndef __KARTET_VECTOR_MATHS_TOOLS__
#define __KARTET_VECTOR_MATHS_TOOLS__

	#include "Core/LibTools.hpp"
	#include "Core/Meta.hpp"
	#include "Core/Mat.hpp"

namespace Kartet
{
	/**
	\brief Small vector object (of static size).
	\tparam d Number of rows.
	\tparam T Type of the matrix.

	Small vector object with static size :
	\code
	Kartet::Vec<4,double> v1; // Unitialized value.
	Kartet::Mat<2float> v2(0.0); // Cleared to 0.
	Vec3f v3 = vec3(1,0,0); // 3x<float>.
	Vec4z v4 = vec4(0,0,1,1.0+2.0*I()); // 4xComplex<double>.
	\endcode
	**/
	template<int d, typename T>
	struct Vec : public Mat<d,1,T>
	{
		__host__ __device__ inline const T& operator()(const int& i) const;
		__host__ __device__ inline T& operator()(const int& i);
		__host__ __device__ inline Vec(void);
		__host__ __device__ inline Vec(const T& val); // Ambiguous?
		__host__ __device__ inline Vec(const Mat<d,1,T>& o);
		template<typename U>
		__host__ __device__ inline Vec(const Mat<d,1,U>& o);
		template<typename U>
		__host__ __device__ inline Vec(const U* ptr);
		using Mat<d,1,T>::operator();
		using Mat<d,1,T>::operator=;
		using Mat<d,1,T>::clear;
		using Mat<d,1,T>::set;
		__host__ __device__ inline const T& x(void) const;
		__host__ __device__ inline T& x(void);
		__host__ __device__ inline const T& y(void) const;
		__host__ __device__ inline T& y(void);
		__host__ __device__ inline const T& z(void) const;
		__host__ __device__ inline T& z(void);
		__host__ __device__ inline const T& w(void) const;
		__host__ __device__ inline T& w(void);

	};

// Type aliases :
	/**
	\typedef Vec2i
	\brief Alias to Vec<2,int> type, see Kartet::Vec for more information.
	\related Kartet::Vec
	\typedef Vec2ui
	\brief Alias to Vec<2,unsigned int> type, see Kartet::Vec for more information.
	\related Kartet::Vec
	\typedef Vec2f
	\brief Alias to Vec<2,float> type, see Kartet::Vec for more information.
	\related Kartet::Vec
	\typedef Vec3f
	\brief Alias to Vec<3,float>, see Kartet::Vec for more information.
	\related Kartet::Vec
	\typedef Vec4f
	\brief Alias to Vec<4,float>, see Kartet::Vec for more information.
	\relatedalso Kartet::Vec
	\typedef Vec2c
	\brief Alias to Vec<2,Complex<float> > type, see Kartet::Vec for more information.
	\related Kartet::Vec
	\typedef Vec3c
	\brief Alias to Vec<3,Complex<float> >, see Kartet::Vec for more information.
	\related Kartet::Vec
	\typedef Vec4c
	\brief Alias to Vec<4,Complex<float> >, see Kartet::Vec for more information.
	\relatedalso Kartet::Vec
	\typedef Vec2d
	\brief Alias to Vec<2,double> type, see Kartet::Vec for more information.
	\related Kartet::Vec
	\typedef Vec3d
	\brief Alias to Vec<3,double>, see Kartet::Vec for more information.
	\related Kartet::Vec
	\typedef Vec4d
	\brief Alias to Vec<4,double>, see Kartet::Vec for more information.
	\relatedalso Kartet::Vec
	\typedef Vec2z
	\brief Alias to Vec<2,Complex<double> > type, see Kartet::Vec for more information.
	\related Kartet::Vec
	\typedef Vec3z
	\brief Alias to Vec<3,Complex<double> >, see Kartet::Vec for more information.
	\related Kartet::Vec
	\typedef Vec4z
	\brief Alias to Vec<4,Complex<double> >, see Kartet::Vec for more information.
	\relatedalso Kartet::Vec
	**/
	typedef Vec<2,int> Vec2i;
	typedef Vec<2,int> Vec2ui;
	typedef Vec<2,float> Vec2f;
	typedef Vec<3,float> Vec3f;
	typedef Vec<4,float> Vec4f;
	typedef Vec<2,Complex<float> > Vec2c;
	typedef Vec<3,Complex<float> > Vec3c;
	typedef Vec<4,Complex<float> > Vec4c;
	typedef Vec<2,double> Vec2d;
	typedef Vec<3,double> Vec3d;
	typedef Vec<4,double> Vec4d;
	typedef Vec<2,Complex<double> > Vec2z;
	typedef Vec<3,Complex<double> > Vec3z;
	typedef Vec<4,Complex<double> > Vec4z;

// Traits :
	template<int d, typename T>
	struct Traits<Vec<d,T> >
	{
		typedef Traits<T> SubTraits;
		typedef typename SubTraits::BaseType BaseType;
		static const bool 	isConst 	= false,
					isArray		= false,
					isPointer 	= false,
					isReference 	= false,
					isComplex 	= SubTraits::isComplex,
					isFloatingPoint = SubTraits::isFloatingPoint,
					isMatrix	= true,
					isVector	= true;
	};

// Functions :
	/**
	\brief Default constructor.
	The elements are left unitialized.
	**/
	template<int d, typename T>
	__host__ __device__ inline Vec<d,T>::Vec(void)
	{ }

	/**
	\brief Vector constructor.
	\param val Fill value.
	**/
	template<int d, typename T>
	__host__ __device__ inline Vec<d,T>::Vec(const T& val)
	 : Mat<d,1,T>(val)
	{ }

	/**
	\brief Vector copy constructor.
	\param o The vector to be copied.
	**/
	template<int d, typename T>
	__host__ __device__ inline Vec<d,T>::Vec(const Mat<d,1,T>& o)
	 : Mat<d,1,T>(o)
	{ }

	/**
	\brief Vector copy constructor.
	\param o The vector to be copied.
	**/
	template<int d, typename T>
	template<typename U>
	__host__ __device__ inline Vec<d,T>::Vec(const Mat<d,1,U>& o)
	 : Mat<d,1,T>(o)
	{ }

	/**
	\brief Matrix constructor.
	\param ptr Pointer to the array to be copied. It must contain d elements.
	**/
	template<int d, typename T>
	template<typename U>
	__host__ __device__ inline Vec<d,T>::Vec(const U* ptr)
	 : Mat<d,1,T>(ptr)
	{ }

	/**
	\brief Read a vector element.
	\param i Row index.
	\return A constant reference to the value of the vector at the index \f$(i)\f$.
	**/
	template<int d, typename T>
	__host__ __device__ inline const T& Vec<d,T>::operator()(const int& i) const
	{
		return Mat<d,1,T>::m[i];
	}

	/**
	\brief Read/write a vector element.
	\param i Row index.
	\return A reference to the value of the vector at the index \f$(i)\f$.
	**/
	template<int d, typename T>
	__host__ __device__ inline T& Vec<d,T>::operator()(const int& i)
	{
		return Mat<d,1,T>::m[i];
	}

	/**
	\brief Read the first vector element.
	\return A constant reference to the value of the first vector element.
	**/
	template<int d, typename T>
	__host__ __device__ inline const T& Vec<d,T>::x(void) const
	{
		return Mat<d,1,T>::m[0];
	}

	/**
	\brief Read/write the first vector element.
	\return A reference to the value of the first vector element.
	**/
	template<int d, typename T>
	__host__ __device__ inline T& Vec<d,T>::x(void)
	{
		return Mat<d,1,T>::m[0];
	}

	/**
	\brief Read the second vector element.
	\return A constant reference to the value of the second vector element.
	**/
	template<int d, typename T>
	__host__ __device__ inline const T& Vec<d,T>::y(void) const
	{
		return Mat<d,1,T>::m[1];
	}

	/**
	\brief Read/write the second vector element.
	\return A reference to the value of the second vector element.
	**/
	template<int d, typename T>
	__host__ __device__ inline T& Vec<d,T>::y(void)
	{
		return Mat<d,1,T>::m[1];
	}

	/**
	\brief Read the third vector element.
	\return A constant reference to the value of the third vector element.
	**/
	template<int d, typename T>
	__host__ __device__ inline const T& Vec<d,T>::z(void) const
	{
		STATIC_ASSERT_VERBOSE(d>2, INVALID_DIMENSION)
		return Mat<d,1,T>::m[2];
	}

	/**
	\brief Read/write the third vector element.
	\return A reference to the value of the third vector element.
	**/
	template<int d, typename T>
	__host__ __device__ inline T& Vec<d,T>::z(void)
	{
		STATIC_ASSERT_VERBOSE(d>2, INVALID_DIMENSION)
		return Mat<d,1,T>::m[2];
	}

	/**
	\brief Read the fourth vector element.
	\return A constant reference to the value of the fourth vector element.
	**/
	template<int d, typename T>
	__host__ __device__ inline const T& Vec<d,T>::w(void) const
	{
		STATIC_ASSERT_VERBOSE(d>3, INVALID_DIMENSION)
		return Mat<d,1,T>::m[3];
	}

	/**
	\brief Read/write the fourth vector element.
	\return A reference to the value of the fourth vector element.
	**/
	template<int d, typename T>
	__host__ __device__ inline T& Vec<d,T>::w(void)
	{
		STATIC_ASSERT_VERBOSE(d>3, INVALID_DIMENSION)
		return Mat<d,1,T>::m[3];
	}

// Non-members :
	/**
	\brief Make a 2D vector.
	\param x Element at \f$(0)\f$.
	\param y Element at \f$(1)\f$.
	\return A 2D vector with the given elements and the highest rated type.
	\related Vec
	**/
	template<typename T0, typename T1>
	__host__ __device__ inline Vec<2,typename ResultingType<T0,T1>::Type> vec2(const T0& x, const T1& y)
	{
		Vec<2,typename ResultingType<T0,T1>::Type> v;
		v(0) = x;
		v(1) = y;
		return v;
	}

	/**
	\brief Make a 2D vector.
	\param a All values.
	\return A 2D vector filled with the given element.
	\related Vec
	**/
	template<typename T>
	__host__ __device__ inline Vec<2,T> vec2(const T& a)
	{
		return Vec<2,T>(a);
	}

	/**
	\brief Make a 3D vector.
	\param x Element at \f$(0)\f$.
	\param y Element at \f$(1)\f$.
	\param z Element at \f$(2)\f$.
	\return A 3D vector with the given elements and the highest rated type.
	\related Vec
	**/
	template<typename T0, typename T1, typename T2>
	__host__ __device__ inline Vec<3,typename ResultingType3<T0,T1,T2>::Type> vec3(const T0& x, const T1& y, const T2& z)
	{
		Vec<3,typename ResultingType3<T0,T1,T2>::Type> v;
		v(0) = x;
		v(1) = y;
		v(2) = z;
		return v;
	}

	/**
	\brief Make a 3D vector.
	\param a All values.
	\return A 3D vector filled with the given element.
	\related Vec
	**/
	template<typename T>
	__host__ __device__ inline Vec<3,T> vec3(const T& a)
	{
		return Vec<3,T>(a);
	}

	/**
	\brief Make a 4D vector.
	\param x Element at \f$(0)\f$.
	\param y Element at \f$(1)\f$.
	\param z Element at \f$(2)\f$.
	\param w Element at \f$(3)\f$.
	\return A 4D vector with the given elements and the highest rated type.
	\related Vec
	**/
	template<typename T0, typename T1, typename T2, typename T3>
	__host__ __device__ inline Vec<4,typename ResultingType4<T0,T1,T2,T3>::Type> vec4(const T0& x, const T1& y, const T2& z, const T2& w)
	{
		Vec<4,typename ResultingType4<T0,T1,T2,T3>::Type::Type> v;
		v(0) = x;
		v(1) = y;
		v(2) = z;
		v(3) = w;
		return v;
	}

	/**
	\brief Make a 4D vector.
	\param a All values.
	\return A 4D vector filled with the given element.
	\related Vec
	**/
	template<typename T>
	__host__ __device__ inline Vec<4,T> vec4(const T& a)
	{
		return Vec<4,T>(a);
	}

	/**
	\brief Binary matrix-vector product operator.
	\param a Input matrix.
	\param b Input vector.
	\return A new vector set to \f$a \times b\f$.
	\related Vec
	**/
	template<int ra, int ca, typename Ta, int rb, typename Tb>
	__host__ __device__ inline Vec<ra,typename ResultingType<Ta,Tb>::Type> operator*(const Mat<ra,ca,Ta>& a, const Vec<rb,Tb>& b)
	{
		STATIC_ASSERT_VERBOSE(ca==rb, INVALID_DIMENSION)
		Vec<ra,typename ResultingType<Ta,Tb>::Type> res;
		for(int i=0; i<ra; i++)
			res(i) = metaBinaryProductSum<ca, typename ResultingType<Ta,Tb>::Type>(a.m+i, b.m, ra, 1);
		return res;
	}

// Misc. :
	/**
	\brief Print the vector to a standard stream.
	\param os Output standard stream.
	\param v Input vector.
	\return A reference to the output stream.
	\related Vec

	Example :
	\code
		Vec4d v(1.0);
		std::cout << "4D Vector : " << std::endl;
		std::cout << v << std::endl;
	\endcode
	**/
	template<int d, typename T>
	__host__ std::ostream& operator<<(std::ostream& os, const Vec<d,T>& v)
	{
		const T zero = static_cast<T>(0),
			minusZero = -static_cast<T>(0);
		// Change the floatting point format if not specified :
		const std::ios_base::fmtflags originalFlags = os.flags();
		const bool forcedFloattingFormat = (os.flags() & std::ios_base::floatfield)!=0;
		if(!forcedFloattingFormat)
			os.setf(std::ios_base::scientific);

		const int precision = forcedFloattingFormat? os.precision() : 3;
		const int originalPrecision = os.precision(precision);
		const char fillCharacter = ' ';
		const char originalFill = os.fill(fillCharacter);
		const bool flag = !(os.flags() & std::ios_base::showpos);
		os << '(';
		for(int k=0; k<(d-1); k++)
		{
			const bool f = (!Traits<T>::isComplex && softLargerEqual(v.m[k], zero)) && !((IsSame<T, float>::value || IsSame<T, double>::value) && compareBits(v.m[k], minusZero));
			if(flag && f)
				os << fillCharacter;
			os << v.m[k] << ", ";
		}
		const bool f = (!Traits<T>::isComplex && softLargerEqual(v.m[d-1], zero)) && !((IsSame<T, float>::value || IsSame<T, double>::value) && compareBits(v.m[d-1], minusZero));
		if(flag && f)
			os << fillCharacter;
		os << v.m[d-1] << ')';
		// Reset :
		os.precision(originalPrecision);
		os.fill(originalFill);
		os.flags(originalFlags);
		return os;
	}
}

#endif

