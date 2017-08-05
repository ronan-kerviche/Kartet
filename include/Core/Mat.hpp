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
	\file    Mat.hpp
	\brief   Matrix maths tools.
	\author  R. Kerviche
	\date    June 26th 2017
**/

#ifndef __KARTET_MATRIX_MATHS_TOOLS__
#define __KARTET_MATRIX_MATHS_TOOLS__

	#include "Core/LibTools.hpp"
	#include "Core/Meta.hpp"
	#include "Core/Complex.hpp"
	#include "Core/Traits.hpp"

namespace Kartet
{
	/**
	\brief Small matrix object (of static size).
	\tparam r Number of rows.
	\tparam c Number of columns.
	\tparam T Type of the matrix.

	Small matrix object with static size :
	\code
	Kartet::Mat<3,4,double> m1; // Unitialized value.
	Kartet::Mat<2,4,float> m2(0.0); // Cleared to 0.
	Mat3f m3 = Mat3f::identity(); // 3x3 <float> identity.
	Mat4z m4 = Mat4d::identity(); // 4x4 Complex<double> identity, initialized from a double identity.
	\endcode
	**/
	template<int r, int c, typename T>
	struct Mat
	{
		STATIC_ASSERT_VERBOSE((r*c)>1, INVALID_DIMENSION)
		typedef T BaseType;
		static const int	rows = r,
					cols = c,
					dim = r*c;
		T m[r*c];

		__host__ __device__ inline Mat(void);
		__host__ __device__ inline Mat(const T& val);
		__host__ __device__ inline Mat(const Mat<r, c, T>& o);
		template<typename U>
		__host__ __device__ inline Mat(const Mat<r, c, U>& o);
		template<typename U>
		__host__ __device__ inline Mat(const U* ptr);
		__host__ __device__ inline const T& operator()(const int& i, const int& j) const;
		__host__ __device__ inline T& operator()(const int& i, const int& j);
		__host__ __device__ inline Mat<r,1,T> col(const int& j);
		// Coumpound assignment operators :
		#define COUMPOUND_ASSIGNMENT(OP) \
			__host__ __device__ inline const Mat<r,c,T>& operator OP (const Mat<r, c, T>& o); \
			template<typename U> \
			__host__ __device__ inline const Mat<r,c,T>& operator OP (const Mat<r, c, U>& o);
		COUMPOUND_ASSIGNMENT(=)
		COUMPOUND_ASSIGNMENT(+=)
		COUMPOUND_ASSIGNMENT(-=)	
		#undef COUMPOUND_ASSIGNMENT
		template<typename U>
		__host__ __device__ inline const Mat<r,c,T>& operator*=(const U& o);
		template<typename U>
		__host__ __device__ inline const Mat<r,c,T>& operator/=(const U& o);
		__host__ __device__ inline const Mat<r,c,T>& clear(const T& val);

		// Tools :
		__host__ __device__ inline static Mat<r,c,T> identity(void);
	};

	// Type aliases :
	/**
	\typedef Mat2f
	\brief Alias to Mat<2,2,float> type, see Kartet::Mat for more information.
	\related Kartet::Mat
	\typedef Mat3f
	\brief Alias to Mat<3,3,float>, see Kartet::Mat for more information.
	\related Kartet::Mat
	\typedef Mat4f
	\brief Alias to Mat<4,4,float>, see Kartet::Mat for more information.
	\relatedalso Kartet::Mat
	\typedef Mat2c
	\brief Alias to Mat<2,2,Complex<float> > type, see Kartet::Mat for more information.
	\related Kartet::Mat
	\typedef Mat3c
	\brief Alias to Mat<3,3,Complex<float> >, see Kartet::Mat for more information.
	\related Kartet::Mat
	\typedef Mat4c
	\brief Alias to Mat<4,4,Complex<float> >, see Kartet::Mat for more information.
	\relatedalso Kartet::Mat
	\typedef Mat2d
	\brief Alias to Mat<2,2,double> type, see Kartet::Mat for more information.
	\related Kartet::Mat
	\typedef Mat3d
	\brief Alias to Mat<3,3,double>, see Kartet::Mat for more information.
	\related Kartet::Mat
	\typedef Mat4d
	\brief Alias to Mat<4,4,double>, see Kartet::Mat for more information.
	\relatedalso Kartet::Mat
	\typedef Mat2z
	\brief Alias to Mat<2,2,Complex<double> > type, see Kartet::Mat for more information.
	\related Kartet::Mat
	\typedef Mat3z
	\brief Alias to Mat<3,3,Complex<double> >, see Kartet::Mat for more information.
	\related Kartet::Mat
	\typedef Mat4z
	\brief Alias to Mat<4,4,Complex<double> >, see Kartet::Mat for more information.
	\relatedalso Kartet::Mat
	**/
	typedef Mat<2,2,float> Mat2f;
	typedef Mat<3,3,float> Mat3f;
	typedef Mat<4,4,float> Mat4f;
	typedef Mat<2,2,Complex<float> > Mat2c;
	typedef Mat<3,3,Complex<float> > Mat3c;
	typedef Mat<4,4,Complex<float> > Mat4c;
	typedef Mat<2,2,double> Mat2d;
	typedef Mat<3,3,double> Mat3d;
	typedef Mat<4,4,double> Mat4d;
	typedef Mat<2,2,Complex<double> > Mat2z;
	typedef Mat<3,3,Complex<double> > Mat3z;
	typedef Mat<4,4,Complex<double> > Mat4z;

	// Traits :
	template<int r, int c, typename T>
	struct Traits<Mat<r,c,T> >
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
					isVector	= (c==1);
	};

	// Functions :
	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,c,T>::Mat(void)
	{ } // Leave unitialized.

	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,c,T>::Mat(const T& val)
	{
		clear(val);
	}

	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,c,T>::Mat(const Mat<r,c,T>& o)
	{
		metaUnaryEqual<dim>(m, o.m);
	}

	template<int r, int c, typename T>
	template<typename U>
	__host__ __device__ inline Mat<r,c,T>::Mat(const Mat<r,c,U>& o)
	{
		metaUnaryEqual<dim>(m, o.m);
	}

	template<int r, int c, typename T>
	template<typename U>
	__host__ __device__ inline Mat<r,c,T>::Mat(const U* ptr)
	{
		metaUnaryEqual<dim>(m, ptr);
	}

	template<int r, int c, typename T>
	__host__ __device__ inline const T& Mat<r,c,T>::operator()(const int& i, const int& j) const
	{
		return m[j*r+i];
	}

	template<int r, int c, typename T>
	__host__ __device__ inline T& Mat<r,c,T>::operator()(const int& i, const int& j)
	{
		return m[j*r+i];
	}

	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,1,T> Mat<r,c,T>::col(const int& j)
	{
		Mat<r,1,T> v(m+j*r);
		return v;
	}

	template<int r, int c, typename T>
	__host__ __device__ inline const Mat<r,c,T>& Mat<r,c,T>::operator=(const Mat<r,c,T>& o)
	{
		metaUnaryEqual<dim>(this->m, o.m);
		return (*this);
	}

	template<int r, int c, typename T>
	template<typename U>
	__host__ __device__ inline const Mat<r,c,T>& Mat<r,c,T>::operator=(const Mat<r,c,U>& o)
	{
		metaUnaryEqual<dim>(this->m, o.m);
		return (*this);
	}

	#define COUMPOUND_ASSIGNMENT(OP, METAFUN) \
		template<int r, int c, typename T> \
		__host__ __device__ inline const Mat<r,c,T>& Mat<r,c,T>::operator OP (const Mat<r,c,T>& o) \
		{ \
			METAFUN <dim>(this->m, reinterpret_cast<const T*>(this->m), o.m); \
			return (*this); \
		} \
		 \
		template<int r, int c, typename T> \
		template<typename U> \
		__host__ __device__ inline const Mat<r,c,T>& Mat<r,c,T>::operator OP (const Mat<r,c,U>& o) \
		{ \
			METAFUN <dim>(this->m, reinterpret_cast<const T*>(this->m), o.m); \
			return (*this); \
		}
	
	COUMPOUND_ASSIGNMENT(+=, metaBinaryPlus)
	COUMPOUND_ASSIGNMENT(-=, metaBinaryMinus)
	#undef COUMPOUNT_ASSIGNMENT

	template<int r, int c, typename T>
	template<typename U>
	__host__ __device__ inline const Mat<r,c,T>& Mat<r,c,T>::operator*=(const U& o)
	{
		metaBinaryProduct<dim>(this->m, this->m, o);
		return (*this);
	}

	template<int r, int c, typename T>
	template<typename U>
	__host__ __device__ inline const Mat<r,c,T>& Mat<r,c,T>::operator/=(const U& o)
	{
		metaBinaryQuotient<dim>(this->m, this->m, o);
		return (*this);
	}	

	template<int r, int c, typename T>
	__host__ __device__ inline const Mat<r,c,T>& Mat<r,c,T>::clear(const T& val)
	{
		metaUnaryEqual<Mat<r,c,T>::dim>(this->m, val);
		return (*this);
	}

	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,c,T> Mat<r,c,T>::identity(void)
	{
		Mat<r,c,T> res;
		for(int k=0; k<Mat<r,c,T>::dim; k++)
			res.m[k] = (k%(r+1)==0) ? static_cast<T>(1) : static_cast<T>(0);
		return res;
	}

	// Non-members :
	template<int r, int c, typename T>
	__host__ __device__ inline Mat<r,c,T> operator-(const Mat<r,c,T>& a)
	{
		Mat<r,c,T> res;
		metaUnaryMinus<Mat<r,c,T>::dim>(res.m, a.m);
		return res;
	}

	template<int r, int c, typename T, typename U>
	__host__ __device__ inline Mat<r,c,typename ResultingType<T,U>::Type> operator+(const Mat<r,c,T>& a, const Mat<r,c,U>& b)
	{
		Mat<r,c,typename ResultingType<T,U>::Type> res;
		metaBinaryPlus<Mat<r,c,T>::dim>(res.m, a.m, b.m);
		return res;
	}

	template<int r, int c, typename T, typename U>
	__host__ __device__ inline Mat<r,c,typename ResultingType<T,U>::Type> operator-(const Mat<r,c,T>& a, const Mat<r,c,U>& b)
	{
		Mat<r,c,typename ResultingType<T,U>::Type> res;
		metaBinaryMinus<Mat<r,c,T>::dim>(res.m, a.m, b.m);
		return res;
	}

	template<int r, int c, typename T, typename U>
	__host__ __device__ inline Mat<r,c,typename ResultingType<T,U>::Type> operator*(const Mat<r,c,T>& a, const U& b)
	{
		Mat<r,c,typename ResultingType<T,U>::Type> res;
		metaBinaryProduct<Mat<r,c,T>::dim>(res.m, a.m, b);
		return res;
	}

	template<int r, int c, typename T, typename U>
	__host__ __device__ inline Mat<r,c,typename ResultingType<T,U>::Type> operator*(const T& a, const Mat<r,c,U>& b)
	{
		Mat<r,c,typename ResultingType<T,U>::Type> res;
		metaBinaryProduct<Mat<r,c,T>::dim>(res.m, a, b.m);
		return res;
	}

	template<int r, int c, typename T, typename U>
	__host__ __device__ inline Mat<r,c,typename ResultingType<T,U>::Type> modulate(const Mat<r,c,T>& a, const Mat<r,c,U>& b)
	{
		Mat<r,c,typename ResultingType<T,U>::Type> res;
		metaBinaryProduct<Mat<r,c,T>::dim>(res.m, a.m, b.m);
		return res;
	}

	template<int ra, int ca, typename Ta, int rb, int cb, typename Tb>
	__host__ __device__ inline Mat<ra,cb,typename ResultingType<Ta,Tb>::Type> operator*(const Mat<ra,ca,Ta>& a, const Mat<rb,cb,Tb>& b)
	{
		STATIC_ASSERT_VERBOSE(ca==rb, INVALID_DIMENSION)
		Mat<ra,cb,typename ResultingType<Ta,Tb>::Type> res;
		for(int j=0; j<cb; j++)
			for(int i=0; i<ra; i++)
				res(i,j) = metaBinaryProductSum<ca, typename ResultingType<Ta,Tb>::Type>(a.m+i, b.m+j*rb, ra, 1);
		return res;
	}

	template<int r, int c, typename T, typename U>
	__host__ __device__ inline Mat<r,c,typename ResultingType<T,U>::Type> operator/(const Mat<r,c,T>& a, const U& b)
	{
		Mat<r,c,typename ResultingType<T,U>::Type> res;
		metaBinaryQuotient<Mat<r,c,T>::dim>(res.m, a.m, b);
		return res;
	}

	template<int r, typename T>
	__host__ __device__ inline T trace(const Mat<r,r,T>& a)
	{
		return metaUnaryPlusSum<r, T>(a.m, r+1);
	}

	template<int r, int c, typename T>
	__host__ __device__ inline Mat<c,r,T> transpose(const Mat<r,c,T>& a)
	{	
		Mat<c,r,T> res;
		for(int j=0; j<r; j++)
			for(int i=0; i<c; i++)
				res(i,j) = a(j,i);
		return res;
	}

	template<int r, typename T>
	__host__ __device__ inline Mat<r,r,T>& transposeInPlace(Mat<r,r,T>& a)
	{	
		for(int j=0; j<r; j++)
			for(int i=0; i<r; i++)
				swap(a(i,j),a(j,i));
		return a;
	}

	template<int r, int c, typename T>
	__host__ __device__ inline Mat<c,r,T> hermTranspose(const Mat<r,c,T>& a)
	{	
		Mat<c,r,T> res;
		for(int j=0; j<r; j++)
			for(int i=0; i<c; i++)
				res(i,j) = conj(a(j,i));
		return res;
	}

	template<typename T>
	__host__ __device__ inline T determinant2x2(const T* m)
	{
		return m[0]*m[3] - m[1]*m[2];
	}

	template<typename T>
	__host__ __device__ inline T determinant2x2(const Mat<2,2,T>& a)
	{
		return determinant2x2(a.m);
	}

	template<typename T>
	__host__ __device__ inline T determinant3x3(const T* m)
	{
		return m[0]*m[4]*m[8] + m[1]*m[5]*m[6] + m[2]*m[3]*m[7] - m[2]*m[4]*m[6] - m[1]*m[3]*m[8] - m[0]*m[5]*m[7];
	}

	template<typename T>
	__host__ __device__ inline T determinant3x3(const Mat<3,3,T>& a)
	{
		return determinant3x3(a.m);
	}

	template<int r, typename T>
	__host__ __device__ inline T determinant(const T* a)
	{
		if(r==1)
			return *a;
		else if(r==2)
			return determinant2x2(a);
		else if(r==3)
			return determinant3x3(a);
		else
			throw InvalidOperation; // To be implemented.
	}

	template<int r, typename T>
	__host__ __device__ inline T determinant(const Mat<r,r,T>& a)
	{
		return determinant<r>(a.m);
	}

	template<int r, typename T, typename U>
	__host__ __device__ inline typename ResultingType<T,U>::Type dot(const Mat<r,1,T>& a, const Mat<r,1,U>& b)
	{
		return metaBinaryProductSum<r, typename ResultingType<T,U>::Type>(a.m, b.m);
	}

	template<int r, typename T, typename U>
	__host__ __device__ inline typename ResultingType<T,U>::Type hermDot(const Mat<r,1,T>& a, const Mat<r,1,U>& b)
	{
		return metaBinaryHermProductSum<r, typename ResultingType<T,U>::Type>(a.m, b.m);
	}

	template<int r, typename T>
	__host__ __device__ inline T frobenius(const Mat<r,r,T>& a)
	{
		T val = static_cast<T>(0);
		for(int i=0; i<r; i++)
			val += metaBinaryHermProductSum<r, typename ResultingType<T,T>::Type>(a.m+i, a.m+i*r, r, 1);
		return ::sqrt(val);
	}

	template<int r, typename T>
	__host__ __device__ inline typename Traits<T>::BaseType normSquared(const Mat<r,1,T>& v)
	{
		return metaUnaryAbsSquareSum<r, typename Traits<T>::BaseType>(v.m);
	}

	template<int r, typename T>
	__host__ __device__ inline typename Traits<T>::BaseType lengthSquared(const Mat<r,1,T>& v) // alias normSquared
	{
		return normSquared(v);
	}

	template<int r, typename T>
	__host__ __device__ inline typename Traits<T>::BaseType norm(const Mat<r,1,T>& v)
	{
		return ::sqrt(metaUnaryAbsSquareSum<r, typename Traits<T>::BaseType>(v.m));
	}

	template<int r, typename T>
	__host__ __device__ inline typename Traits<T>::BaseType rnorm(const Mat<r,1,T>& v)
	{
		#ifdef __CUDACC__
			return ::rsqrt(metaUnaryAbsSquareSum<r, typename Traits<T>::BaseType>(v.m));
		#else
			return static_cast<typename Traits<T>::BaseType>(1)/::sqrt(metaUnaryAbsSquareSum<r, typename Traits<T>::BaseType>(v.m));
		#endif
	}

	template<int r, typename T>
	__host__ __device__ inline typename Traits<T>::BaseType length(const Mat<r,1,T>& v) // alias norm
	{
		return norm(v);
	}

	template<int r, typename T>
	__host__ __device__ inline typename Traits<T>::BaseType rlength(const Mat<r,1,T>& v) // alias norm
	{
		return rnorm(v);
	}

	template<int r, typename T>
	__host__ __device__ inline Mat<r,1,T> normalize(const Mat<r,1,T>& v)
	{
		#ifdef __CUDACC__
			return v*::rsqrt(metaUnaryAbsSquareSum<r, typename Traits<T>::BaseType>(v.m));
		#else
			return v/norm(v);
		#endif
	}

	template<int r, typename T, typename U>
	__host__ __device__ inline Mat<r,1,typename ResultingType<T,U>::Type> reflect(const Mat<r,1,T>& dir, const Mat<r,1,U>& normal)
	{
		return dir - (static_cast<typename ResultingType<T,U>::Type>(2)*dot(dir, normal))*normal;
	}

// Dimension specific :
	template<typename T0, typename T1, typename T2, typename T3>
	__host__ __device__ inline Mat<2,2,typename ResultingType4<T0,T1,T2,T3>::Type> makeMat2(const T0& m0, const T1& m2, const T2& m1, const T3& m3)
	{
		Mat<2,2,typename ResultingType4<T0,T1,T2,T3>::Type> m;
		m(0,0) = m0;
		m(1,0) = m1;
		m(0,1) = m2;
		m(1,1) = m3;
		return m;
	}

	template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
	__host__ __device__ inline Mat<3,3,typename ResultingType9<T0,T1,T2,T3,T4,T5,T6,T7,T8>::Type> makeMat3(const T0& m0, const T1& m2, const T2& m1, const T3& m3, const T4& m4, const T5& m5, const T6& m6, const T7& m7, const T8& m8)
	{
		Mat<3,3,typename ResultingType9<T0,T1,T2,T3,T4,T5,T6,T7,T8>::Type> m;
		m(0,0) = m0;
		m(1,0) = m1;
		m(2,0) = m2;
		m(0,1) = m3;
		m(1,1) = m4;
		m(2,1) = m5;
		m(0,2) = m6;
		m(1,2) = m7;
		m(2,2) = m8;
		return m;
	}

	template<typename T0, typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15>
	__host__ __device__ inline Mat<4,4,typename ResultingType16<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15>::Type> makeMat4(const T0& m0, const T1& m2, const T2& m1, const T3& m3, const T4& m4, const T5& m5, const T6& m6, const T7& m7, const T8& m8,const T9& m9, const T10& m10, const T11& m11, const T12& m12, const T13& m13, const T14& m14, const T15& m15)
	{
		Mat<4,4,typename ResultingType16<T0,T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15>::Type> m;
		m(0,0) = m0;
		m(1,0) = m1;
		m(2,0) = m2;
		m(3,0) = m3;
		m(0,1) = m4;
		m(1,1) = m5;
		m(2,1) = m6;
		m(3,1) = m7;
		m(0,2) = m8;
		m(1,2) = m9;
		m(2,2) = m10;
		m(3,2) = m11;
		m(0,3) = m12;
		m(1,3) = m13;
		m(2,3) = m14;
		m(3,3) = m15;
		return m;
	}

	template<typename T>
	__host__ __device__ inline Mat<2,2,T> rot2(const T& angle)
	{
		Mat<2,2,T> res;
		const T c = cos(angle),
			s = sin(angle);
		res(0,0) = c;
		res(1,0) = -s;
		res(0,1) = s;
		res(1,1) = c;
		return res;
	}

	template<typename T>
	__host__ __device__ inline Mat<3,3,T> rot3x(const T& angle)
	{
		Mat<3,3,T> res(static_cast<T>(0));
		const T	c = cos(angle),
			s = sin(angle);	
		res(0,0) = static_cast<T>(1);
		res(1,1) = c;
		res(2,1) = -s;
		res(1,2) = s;
		res(2,2) = c;
		return res;
	}

	template<typename T>
	__host__ __device__ inline Mat<3,3,T> rot3y(const T& angle)
	{
		Mat<3,3,T> res(static_cast<T>(0));
		const T	c = cos(angle),
			s = sin(angle);
		res(0,0) = c;
		res(2,0) = -s;	
		res(1,1) = static_cast<T>(1);
		res(0,2) = s;
		res(2,2) = c;
		return res;
	}

	template<typename T>
	__host__ __device__ inline Mat<3,3,T> rot3z(const T& angle)
	{
		Mat<3,3,T> res(static_cast<T>(0));
		const T	c = cos(angle),
			s = sin(angle);
		res(0,0) = c;
		res(1,0) = -s;
		res(0,1) = s;
		res(1,1) = c;	
		res(2,2) = static_cast<T>(1);
		return res;
	}

	template<typename T, typename U>
	__host__ __device__ inline Mat<3,1,typename ResultingType<T,U>::Type> cross(const Mat<3,1,T>& a, const Mat<3,1,U>& b)
	{
		Mat<3,1,typename ResultingType<T,U>::Type> c;
		c.m[0] = a.m[1]*b.m[2]-a.m[2]*b.m[1];
		c.m[1] = a.m[2]*b.m[0]-a.m[0]*b.m[2];
		c.m[2] = a.m[0]*b.m[1]-a.m[1]*b.m[0];
		return c;
	}

// Misc. :
	template<int r, int c, typename T>
	__host__ std::ostream& operator<<(std::ostream& os, const Mat<r,c,T>& v)
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
		for(int i=0; i<r; i++)
		{
			os << " |";
			for(int j=0; j<(c-1); j++)
			{
				const T val = v(i,j);
				const bool f = (!Traits<T>::isComplex && softLargerEqual(val,zero)) && !((IsSame<T, float>::value || IsSame<T, double>::value) && compareBits(val, minusZero));
				if(flag && f)
					os << fillCharacter;
				os << val << ", ";
			}
			const T val = v(i, c-1);
			const bool f = (!Traits<T>::isComplex && softLargerEqual(val,zero)) && !((IsSame<T, float>::value || IsSame<T, double>::value) && compareBits(val, minusZero));
			if(flag && f)
				os << ' ';
			os << val << " | " << std::endl;
		}
		// Reset :
		os.precision(originalPrecision);
		os.fill(originalFill);
		os.flags(originalFlags);
		return os;
	}
}

#endif

