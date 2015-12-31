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
	\file    ArrayFunctions.hpp
	\brief   Functions definitions for the arrays.
	\author  R. Kerviche
	\date    November 1st 2009
**/

#ifndef __KARTET_ARRAY_FUNCTIONS__
#define __KARTET_ARRAY_FUNCTIONS__

// Nullary functions : 
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_IndexI, 		IndexI, 		index_t, 	return i; )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_IndexJ, 		IndexJ, 		index_t, 	return j; )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_IndexK, 		IndexK, 		index_t,	return k; )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_IndexEndI, 	IndexEndI, 		index_t, 	return l.numRows()-i-1; )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_IndexEndJ, 	IndexEndJ, 		index_t, 	return l.numColumns()-i-1; )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_IndexEndK, 	IndexEndK, 		index_t,	return l.numSlices()-i-1; )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_INormExclf, 	INormExclf, 		float, 		return static_cast<float>(i)/static_cast<float>(l.numRows()); )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_JNormExclf, 	JNormExclf, 		float, 		return static_cast<float>(j)/static_cast<float>(l.numColumns()); )
	STANDARD_NULLARY_OPERATOR_DEFINITION(	NuOp_KNormExclf, 	KNormExclf, 		float, 		return static_cast<float>(k)/static_cast<float>(l.numSlices()); )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_INormInclf, 	INormInclf, 		float, 		return static_cast<float>(i)/static_cast<float>(l.numRows()-1); )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_JNormInclf, 	JNormInclf, 		float, 		return static_cast<float>(j)/static_cast<float>(l.numColumns()-1); )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_KNormInclf, 	KNormInclf, 		float, 		return static_cast<float>(k)/static_cast<float>(l.numSlices()-1); )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_INormExcl, 	INormExcl, 		double, 	return static_cast<double>(i)/static_cast<double>(l.numRows()); )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_JNormExcl, 	JNormExcl, 		double, 	return static_cast<double>(j)/static_cast<double>(l.numColumns()); )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_KNormExcl,		KNormExcl,		double, 	return static_cast<double>(k)/static_cast<double>(l.numSlices()); )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_INormIncl,		INormIncl, 		double, 	return static_cast<double>(i)/static_cast<double>(l.numRows()-1); )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_JNormIncl, 	JNormIncl, 		double, 	return static_cast<double>(j)/static_cast<double>(l.numColumns()-1); )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_KNormIncl, 	KNormIncl, 		double, 	return static_cast<double>(k)/static_cast<double>(l.numSlices()-1); )
	STANDARD_NULLARY_OPERATOR_DEFINITION(	NuOp_Index, 		Index, 			index_t, 	return l.getIndex(i, j, k); )
	STANDARD_NULLARY_OPERATOR_DEFINITION(	NuOp_Identityb, 	Identityb, 		bool, 		return ((i==j)?(true):(false)); )
	STANDARD_NULLARY_OPERATOR_DEFINITION(	NuOp_Identityf, 	Identityf, 		float, 		return ((i==j)?(1.0f):(0.0f)); )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_Identity, 		Identity,		double, 	return ((i==j)?(1.0):(0.0)); )
	STANDARD_NULLARY_OPERATOR_DEFINITION(	NuOp_UpperTriangleb, 	UpperTriangleb, 	bool, 		return ((i<=j)?(true):(false)); )
	STANDARD_NULLARY_OPERATOR_DEFINITION(	NuOp_UpperTrianglef, 	UpperTrianglef, 	float, 		return ((i<=j)?(1.0f):(0.0f)); )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_UpperTriangle, 	UpperTriangle,		double, 	return ((i<=j)?(1.0):(0.0)); )
	STANDARD_NULLARY_OPERATOR_DEFINITION(	NuOp_LowerTriangleb, 	LowerTriangleb, 	bool, 		return ((i>=j)?(true):(false)); )
	STANDARD_NULLARY_OPERATOR_DEFINITION(	NuOp_LowerTrianglef, 	LowerTrianglef, 	float, 		return ((i>=j)?(1.0f):(0.0f)); )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_LowerTriangle, 	LowerTriangle,		double, 	return ((i>=j)?(1.0):(0.0)); )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_Radiusf,		Radiusf,		float,		return sqrtf((2.0f*i-l.numRows())*(2.0f*i-l.numRows()) + (2.0f*j-l.numColumns())*(2.0f*j-l.numColumns()))/2.0f; )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_Radius,		Radius,			double,		return sqrt((2.0*i-l.numRows())*(2.0*i-l.numRows()) + (2.0*j-l.numColumns())*(2.0*j-l.numColumns()))/2.0; )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_RadiusNormalizedf,	RadiusNormalizedf,	float,		return sqrtf((2.0f*i-(l.numRows()-1))*(2.0f*i-(l.numRows()-1))/((l.numRows()-1)*(l.numRows()-1)) + (2.0f*j-(l.numColumns()-1))*(2.0f*j-(l.numColumns()-1))/((l.numColumns()-1)*(l.numColumns()-1))); )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_RadiusNormalized,	RadiusNormalized,	double,		return sqrt((2.0*i-(l.numRows()-1))*(2.0*i-(l.numRows()-1))/((l.numRows()-1)*(l.numRows()-1)) + (2.0*j-(l.numColumns()-1))*(2.0*j-(l.numColumns()-1))/((l.numColumns()-1)*(l.numColumns()-1))); )

// Unary functions : 
	// From CUDA, automatic or double precision : 
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_cos, 			cos,			return ::cos(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_cosh, 			cosh,			return ::cosh(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_acos, 			acos,			return ::acos(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_acosh,			acosh,			return ::acosh(a); )
#ifdef __CUDACC__
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_cospi, 			cospi,			return ::cospi(a); )
#endif
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_sin, 			sin,			return ::sin(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_sinh, 			sinh,			return ::sinh(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_asin, 			asin,			return ::asin(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_asinh, 			asinh,			return ::asinh(a); )
#ifdef __CUDACC__
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_sinpi, 			sinpi,			return ::sinpi(a); )
#endif
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_tan, 			tan,			return ::tan(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_tanh, 			tanh,			return ::tanh(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_atan,			atan,			return ::atan(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_atanh, 			atanh,			return ::atanh(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_sqrt, 			sqrt,			return ::sqrt(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_cbrt, 			cbrt,			return ::cbrt(a); )
#ifdef __CUDACC__
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_rsqrt, 			rsqrt,			return ::rsqrt(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_rcbrt, 			rcbrt,			return ::rcbrt(a); )
#endif
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_exp, 			exp,			return ::exp(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_exp2, 			exp2,			return ::exp2(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_exp10, 			exp10,			return ::exp10(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_expm1, 			expm1,			return ::expm1(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erf, 			erf,			return ::erf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erfc, 			erfc,			return ::erfc(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_log, 			log, 			return ::log(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_log2, 			log2,			return ::log2(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_log10, 			log10,			return ::log10(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_log1p, 			log1p,			return ::log1p(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_ilogb, 			ilogb,			return ::ilogb(a); )
#ifdef __CUDACC__
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erfinv, 			erfinv,			return ::erfinv(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erfcx, 			erfcx,			return ::erfcx(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erfcinv, 			erfcinv,		return ::erfcinv(a); )	
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_normcdf,			normcdf,		return ::normcdf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION(	UnOp_normcdfinv,		normcdfinv,		return ::normcdfinv(a); )
#endif
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_lgamma, 			lgamma,			return ::lgamma(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_tgamma, 			tgamma,			return ::tgamma(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_j0, 			j0,			return ::j0(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_j1, 			j1,			return ::j1(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_y0, 			y0,			return ::y0(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_y1, 			y1,			return ::y1(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_isfinite, 			isFinite,		return isfinite(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_isinf, 			isInf,			return isinf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_isnan, 			isNan,			return isnan(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_ceil, 			ceil,			return ::ceil(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_floor,			floor,			return ::floor(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_round, 			round,			return ::round(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_trunc, 			trunc,			return ::trunc(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_rint, 			rint,			return ::rint(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_lrint, 			lrint,			return ::lrint(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_lround, 			lround,			return ::lround(a); )
	// From CUDA, single precision :
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_cosf, 			cosf,			return ::cosf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_coshf, 			coshf,			return ::coshf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_acosf, 			acosf,			return ::acosf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_acoshf,			acoshf,			return ::acoshf(a); )
#ifdef __CUDACC__
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_cospif, 			cospif,			return ::cospif(a); )
#endif
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_sinf, 			sinf,			return ::sinf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_sinhf, 			sinhf,			return ::sinhf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_asinf, 			asinf,			return ::asinf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_asinhf, 			asinhf,			return ::asinhf(a); )
#ifdef __CUDACC__
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_sinpif, 			sinpif,			return ::sinpif(a); )
#endif
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_tanf, 			tanf,			return ::tanf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_tanhf, 			tanhf,			return ::tanhf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_atanf,			atanf,			return ::atanf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_atanhf, 			atanhf,			return ::atanhf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_sqrtf, 			sqrtf,			return ::sqrtf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_cbrtf, 			cbrtf,			return ::cbrtf(a); )
#ifdef __CUDACC__
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_rsqrtf, 			rsqrtf,			return ::rsqrtf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_rcbrtf, 			rcbrtf,			return ::rcbrtf(a); )
#endif
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_expf, 			expf,			return ::expf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_exp2f, 			exp2f,			return ::exp2f(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_exp10f, 			exp10f,			return ::exp10f(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_expm1f, 			expm1f,			return ::expm1f(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erff, 			erff,			return ::erff(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erfcf, 			erfcf,			return ::erfcf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_logf, 			logf, 			return ::logf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_log2f, 			log2f,			return ::log2f(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_log10f, 			log10f,			return ::log10f(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_log1pf, 			log1pf,			return ::log1pf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_ilogbf, 			ilogbf,			return ::ilogbf(a); )
#ifdef __CUDACC__
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erfinvf, 			erfinvf,		return ::erfinvf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erfcxf, 			erfcxf,			return ::erfcxf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erfcinvf, 			erfcinvf,		return ::erfcinvf(a); )	
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_normcdff,			normcdff,		return ::normcdff(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION(	UnOp_normcdfinvf,		normcdfinvf,		return ::normcdfinvf(a); )
#endif
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_lgammaf, 			lgammaf,		return ::lgammaf(a);)
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_tgammaf, 			tgammaf,		return ::tgammaf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_j0f, 			j0f,			return ::j0f(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_j1f, 			j1f,			return ::j1f(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_y0f, 			y0f,			return ::y0f(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_y1f, 			y1f,			return ::y1f(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_ceilf, 			ceilf,			return ::ceilf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_floorf,			floorf,			return ::floorf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_roundf, 			roundf,			return ::roundf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_truncf, 			truncf,			return ::truncf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_rintf, 			rintf,			return ::rintf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_lrintf, 			lrintf,			return ::lrintf(a); )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_lroundf, 			lroundf,		return ::lroundf(a); )
	// Specials : 
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_conj, 			conj,			return conj(a); )
	C2R_UNARY_OPERATOR_DEFINITION(		UnOp_real, 			real,			return real(a); )
	C2R_UNARY_OPERATOR_DEFINITION(		UnOp_imag,			imag,			return imag(a); )
	C2R_UNARY_OPERATOR_DEFINITION(		UnOp_abs,			abs,			return abs(a); )
	C2R_UNARY_OPERATOR_DEFINITION(		UnOp_absSq,			absSq,			return absSq(a); )
	C2R_UNARY_OPERATOR_DEFINITION(		UnOp_angle,			angle,			return angle(a); )
	R2C_UNARY_OPERATOR_DEFINITION(		UnOp_angleToComplex,		angleToComplex,		return angleToComplex(a); )
	R2C_UNARY_OPERATOR_DEFINITION(		UnOp_piAngleToComplex,		piAngleToComplex,	return piAngleToComplex(a); )
	CAST_UNARY_OPERATOR_DEFINITION(		UnOp_cast,			cast,			a )

// Transform functions :
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_fftshift,				fftshift,			l.getPositionFFTShift(i, j, k); )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_ifftshift,				ifftshift,			l.getPositionFFTInverseShift(i, j, k); )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_XFlip,				xFlip,				j = l.numColumns()-(j+1); )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_YFlip,				yFlip,				i = l.numRows()-(i+1); )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_distributeSlice, 			distributeSlice,		k=0; )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_distributeColumn,			distributeColumn,		j=0; )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_distributeElement,			distributeElement,		i=j=k=0; )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_diagonal,				diagonal,			j=i; )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_distributeElementsOnColumns,	distributeElementsOnColumns,	i=j; j=0; )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_distributeElementsOnSlices,	distributeElementsOnSlices,	i=k; k=0; k=0; )
	// Following can be extremly slow on GPU :
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_transpose,				transpose,			const index_t _s = i; i = j; j = _s; )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_forceUpSymmetry,			forceUpSymmetry,		const index_t _i=i, _j=j; i=::max(_i,_j); j=::min(_i,_j); )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_forceDownSymmetry,			forceDownSymmetry,		const index_t _i=i, _j=j; i=::min(_i,_j); j=::max(_i,_j); )

// Layout reinterpretation functions :
	STANDARD_LAYOUT_REINTERPRETATION_OPERATOR_DEFINITION( UnOp_clamp, 		clamp, 				i = lnew.getIClamped(i);
															j = lnew.getJClamped(j);
															k = lnew.getKClamped(k); )
	STANDARD_LAYOUT_REINTERPRETATION_OPERATOR_DEFINITION( UnOp_repeat, 		repeat, 			i = lnew.getIWrapped(i);
															j = lnew.getJWrapped(j);
															k = lnew.getKWrapped(k); )
	STANDARD_LAYOUT_REINTERPRETATION_OPERATOR_DEFINITION( UnOp_expandf,		expandf,			i = l.getINorm<float>(i)*lnew.numRows();
															j = l.getJNorm<float>(j)*lnew.numColumns();
															k = l.getKNorm<float>(k)*lnew.numSlices(); )
	STANDARD_LAYOUT_REINTERPRETATION_OPERATOR_DEFINITION( UnOp_expand,		expand,				i = l.getINorm<double>(i)*lnew.numRows();
															j = l.getJNorm<double>(j)*lnew.numColumns();
															k = l.getKNorm<double>(k)*lnew.numSlices(); )

// Binary functions : 
	/* Note that if the input arguments are not of the type specified by the Cuda library, the call might fail.
	   The error reported is : calling a __host__ function <function name>. */
#ifdef __CUDACC__
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_min,			min,			return ::min(a, b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_max,			max,			return ::max(a, b); )
#else
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_min,			min,			return min(a, b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_max,			max,			return max(a, b); )
#endif
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_atan2,			atan2, 			return ::atan2(a,b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_fmax,			fmax,			return ::fmax(a,b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_fmin,			fmin,			return ::fmin(a,b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_fmod,			fmod,			return ::fmod(a,b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_hypot,			hypot,			return ::hypot(a,b); )
	//STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_rhypot,			rhypot,			return ::rhypot(a,b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_jn,			jn,			return ::jn(a,b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_ldexp,			ldexp,			return ::ldexp(a,b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_pow,			pow,			return ::pow(a,b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_remainder,		remainder,		return ::remainder(a,b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_yn,			yn,			return ::yn(a,b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_atan2f,			atan2f,			return ::atan2f(a,b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_fmaxf,			fmaxf,			return ::fmaxf(a,b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_fminf,			fminf,			return ::fminf(a,b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_fmodf,			fmodf,			return ::fmodf(a,b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_hypotf,			hypotf,			return ::hypotf(a,b); )
	//STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_rhypotf,			rhypotf,		return ::rhypotf(a,b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_jnf,			jnf,			return ::jnf(a,b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_ldexpf,			ldexpf,			return ::ldexpf(a,b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_powf,			powf,			return ::powf(a,b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_remainderf,		remainderf,		return ::remainderf(a,b); )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_ynf,			ynf,			return ::ynf(a,b); )

	R2C_BINARY_OPERATOR_DEFINITION( 	BinOp_ToComplex, 		toComplex,		return toComplex(a, b); )
	R2C_BINARY_OPERATOR_DEFINITION( 	BinOp_ToFloatComplex, 		toFloatComplex,		return toFloatComplex(a, b); )
	R2C_BINARY_OPERATOR_DEFINITION( 	BinOp_ToDoubleComplex, 		toDoubleComplex,	return toDoubleComplex(a, b); )

// Shuffle functions :
	STANDARD_SHUFFLE_FUNCTION_DEFINITION( 	ShuFun_ShuffleIndex, 		shuffleIndex,		l.unpackIndex(v, i, j, k); )
	STANDARD_SHUFFLE_FUNCTION_DEFINITION( 	ShuFun_ShuffleRows, 		shuffleRows,		i = v; )
	STANDARD_SHUFFLE_FUNCTION_DEFINITION( 	ShuFun_ShuffleColumns, 		shuffleColumns,		j = v; )
	STANDARD_SHUFFLE_FUNCTION_DEFINITION( 	ShuFun_ShuffleSlices, 		shuffleSlices,		k = v; )

#endif
