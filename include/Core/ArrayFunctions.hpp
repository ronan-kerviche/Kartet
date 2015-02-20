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

#ifndef __KARTET_ARRAY_FUNCTIONS__
#define __KARTET_ARRAY_FUNCTIONS__

namespace Kartet
{
// Nullary functions : 
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_IndexI, 		IndexI, 		index_t, 	i )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_IndexJ, 		IndexJ, 		index_t, 	j )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_IndexK, 		IndexK, 		index_t,	k )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_INormExclf, 	INormExclf, 		float, 		static_cast<float>(i)/static_cast<float>(l.getNumRows()) )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_JNormExclf, 	JNormExclf, 		float, 		static_cast<float>(j)/static_cast<float>(l.getNumColumns()) )
	STANDARD_NULLARY_OPERATOR_DEFINITION(	NuOp_KNormExclf, 	KNormExclf, 		float, 		static_cast<float>(k)/static_cast<float>(l.getNumSlices()) )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_INormInclf, 	INormInclf, 		float, 		static_cast<float>(i)/static_cast<float>(l.getNumRows()-1) )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_JNormInclf, 	JNormInclf, 		float, 		static_cast<float>(j)/static_cast<float>(l.getNumColumns()-1) )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_KNormInclf, 	KNormInclf, 		float, 		static_cast<float>(k)/static_cast<float>(l.getNumSlices()-1) )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_INormExcl, 	INormExcl, 		double, 	static_cast<double>(i)/static_cast<double>(l.getNumRows()) )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_JNormExcl, 	JNormExcl, 		double, 	static_cast<double>(j)/static_cast<double>(l.getNumColumns()) )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_KNormExcl,		KNormExcl,		double, 	static_cast<double>(k)/static_cast<double>(l.getNumSlices()) )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_INormIncl,		INormIncl, 		double, 	static_cast<double>(i)/static_cast<double>(l.getNumRows()-1) )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_JNormIncl, 	JNormIncl, 		double, 	static_cast<double>(j)/static_cast<double>(l.getNumColumns()-1) )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_KNormIncl, 	KNormIncl, 		double, 	static_cast<double>(k)/static_cast<double>(l.getNumSlices()-1) )
	STANDARD_NULLARY_OPERATOR_DEFINITION(	NuOp_Index, 		Index, 			index_t, 	p )
	STANDARD_NULLARY_OPERATOR_DEFINITION(	NuOp_Identityb, 	Identityb, 		bool, 		((i==j)?(true):(false)) )
	STANDARD_NULLARY_OPERATOR_DEFINITION(	NuOp_Identityf, 	Identityf, 		float, 		((i==j)?(1.0f):(0.0f)) )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_Identity, 		Identity,		double, 	((i==j)?(1.0):(0.0)) )
	STANDARD_NULLARY_OPERATOR_DEFINITION(	NuOp_UpperTriangleb, 	UpperTriangleb, 	bool, 		((i<=j)?(true):(false)) )
	STANDARD_NULLARY_OPERATOR_DEFINITION(	NuOp_UpperTrianglef, 	UpperTrianglef, 	float, 		((i<=j)?(1.0f):(0.0f)) )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_UpperTriangle, 	UpperTriangle,		double, 	((i<=j)?(1.0):(0.0)) )
	STANDARD_NULLARY_OPERATOR_DEFINITION(	NuOp_LowerTriangleb, 	LowerTriangleb, 	bool, 		((i>=j)?(true):(false)) )
	STANDARD_NULLARY_OPERATOR_DEFINITION(	NuOp_LowerTrianglef, 	LowerTrianglef, 	float, 		((i>=j)?(1.0f):(0.0f)) )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_LowerTriangle, 	LowerTriangle,		double, 	((i>=j)?(1.0):(0.0)) )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_Radiusf,		Radiusf,		float,		sqrtf((2.0f*i-l.getNumRows())*(2.0f*i-l.getNumRows()) + (2.0f*j-l.getNumColumns())*(2.0f*j-l.getNumColumns()))/2.0f )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_Radius,		Radius,			double,		sqrt((2.0*i-l.getNumRows())*(2.0*i-l.getNumRows()) + (2.0*j-l.getNumColumns())*(2.0*j-l.getNumColumns()))/2.0 )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_RadiusNormalizedf,	RadiusNormalizedf,	float,		sqrtf((2.0f*i-(l.getNumRows()-1))*(2.0f*i-(l.getNumRows()-1))/((l.getNumRows()-1)*(l.getNumRows()-1)) + (2.0f*j-(l.getNumColumns()-1))*(2.0f*j-(l.getNumColumns()-1))/((l.getNumColumns()-1)*(l.getNumColumns()-1))) )
	STANDARD_NULLARY_OPERATOR_DEFINITION( 	NuOp_RadiusNormalized,	RadiusNormalized,	double,		sqrt((2.0*i-(l.getNumRows()-1))*(2.0*i-(l.getNumRows()-1))/((l.getNumRows()-1)*(l.getNumRows()-1)) + (2.0*j-(l.getNumColumns()-1))*(2.0*j-(l.getNumColumns()-1))/((l.getNumColumns()-1)*(l.getNumColumns()-1))) )

// Unary functions : 
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_conj, 			conj,			conj(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_cos, 			cos,			::cos(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_cosf, 			cosf,			::cosf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_sin, 			sin,			::sin(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_sinf, 			sinf,			::sinf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_sqrt, 			sqrt,			::sqrt(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_sqrtf, 			sqrtf,			::sqrtf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_exp, 			exp,			::exp(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_expf, 			expf,			::expf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_log, 			log, 			::log(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_logf, 			logf,			::logf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_acos, 			acos,			::acos(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_acosh,			acosh,			achosh(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_asin, 			asin,			::asin(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_asinh, 			asinh,			::asinh(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_atan,			atan,			::atan(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_atan2, 			atan2,			::atan2(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_atanh, 			atanh,			::atanh(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_cbrt, 			cbrt,			::cbrt(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_ceil, 			ceil,			::ceil(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_cosh, 			cosh,			::cosh(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_cospi, 			cospi,			::cospi(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erf, 			erf,			::erf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erfc, 			erfc,			::erfc(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erfcinv, 			erfcinv,		::erfcinv(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erfcx, 			erfcx,			::erfcx(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erfinv, 			erfinv,			::erfinv(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_exp10, 			exp10,			::exp10(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_exp2, 			exp2,			::exp2(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_expm1, 			expm1,			::expm1(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_floor,			floor,			::floor(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_ilogb, 			ilogb,			::ilogb(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_isfinite, 			isFinite,		isfinite(a) )	// *
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_isinf, 			isInf,			isinf(a) )	// *
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_isnan, 			isNan,			isnan(a) )	// *
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_j0, 			j0,			::j0(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_j1, 			j1,			::j1(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_lgamma, 			lgamma,			::lgamma(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_log10, 			log10,			::log10(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_log2, 			log2,			::log2(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_log1p, 			log1p,			::log1p(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_lrint, 			lrint,			::lrint(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_lround, 			lround,			::lround(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_rcbrt, 			rcbrt,			::rcbrt(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_rint, 			rint,			::rint(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_round, 			round,			::round(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_rsqrt, 			rsqrt,			::rsqrt(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_sinh, 			sinh,			::sinh(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_sinpi, 			sinpi,			::sinpi(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_tan, 			tan,			::tan(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_tanh, 			tanh,			::tanh(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_tgamma, 			tgamma,			::tgamma(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_trunc, 			trunc,			::trunc(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_y0, 			y0,			::y0(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_y1, 			y1,			::y1(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_acosf, 			acosf,			::acosf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_acoshf, 			acoshf,			::acoshf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_asinf, 			asinf,			::asinf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_asinhf, 			asinhf,			::asinhf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_atanf, 			atanf,			::atanf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_atan2f, 			atan2f,			::atan2f(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_atanhf, 			atanhf,			::atanhf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_cbrtf, 			cbrtf,			::cbrtf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_ceilf, 			ceilf,			::ceilf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_coshf, 			coshf,			::coshf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_cospif, 			cospif,			::cospif(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erff, 			erff,			::erff(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erfcf, 			erfcf,			::erfcf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erfcinvf, 			erfcinvf,		::erfcinvf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erfcxf, 			erfcxf,			::erfcxf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_erfinvf, 			erfinvf,		::erfinvf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_exp10f, 			exp10f,			::exp10f(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_exp2f, 			exp2f,			::exp2f(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_floorf, 			floorf,			::floorf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_ilogbf, 			ilogbf,			::ilogbf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_j0f, 			j0f,			::j0f(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_j1f, 			j1f,			::j1f(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_lgammaf, 			lgammaf,		::lgammaf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_log10f, 			log10f,			::log10f(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_log2f, 			log2f,			::log2f(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_lrintf, 			lrintf,			::lrintf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_lroundf, 			lroundf,		::lroundf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_rcbrtf, 			rcbrtf,			::rcbrtf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_rintf, 			rintf,			::rintf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_roundf, 			roundf,			::roundf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_rsqrtf, 			rsqrtf,			::rsqrtf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_sinhf, 			sinhf,			::sinhf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_sinpif, 			sinpif,			::sinpif(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_tanf, 			tanf,			::tanf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_tanhf, 			tanhf,			::tanhf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_tgammaf, 			tgammaf,		::tgammaf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_truncf, 			truncf,			::truncf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_y0f, 			y0f,			::y0f(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_y1f,			y1f,			::y1f(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_normcdff,			normcdff,		::normcdff(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION( 	UnOp_normcdf,			normcdf,		::normcdf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION(	UnOp_normcdfinvf,		normcdfinvf,		::normcdfinvf(a) )
	STANDARD_UNARY_OPERATOR_DEFINITION(	UnOp_normcdfinv,		normcdfinv,		::normcdfinv(a) )

	C2R_UNARY_OPERATOR_DEFINITION(		UnOp_real, 			real,			real(a) )
	C2R_UNARY_OPERATOR_DEFINITION(		UnOp_imag,			imag,			imag(a) )
	C2R_UNARY_OPERATOR_DEFINITION(		UnOp_abs,			abs,			abs(a) )
	C2R_UNARY_OPERATOR_DEFINITION(		UnOp_absSq,			absSq,			absSq(a) )
	C2R_UNARY_OPERATOR_DEFINITION(		UnOp_angle,			angle,			angle(a) )

	R2C_UNARY_OPERATOR_DEFINITION(		UnOp_angleToComplex,		angleToComplex,		angleToComplex(a) )
	R2C_UNARY_OPERATOR_DEFINITION(		UnOp_piAngleToComplex,		piAngleToComplex,	piAngleToComplex(a) )

	CAST_UNARY_OPERATOR_DEFINITION(		UnOp_cast,			cast,			a )

// Transform functions : 
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_fftshift,				fftshift,			p = l.getIndicesFFTShift(i, j, k); )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_ifftshift,				ifftshift,			p = l.getIndicesFFTInverseShift(i, j, k); )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_XMirror,				xmirror,			j = l.getNumColumns()-(j+1); p = l.getIndex(i, j, k); )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_YMirror,				ymirror,			i = l.getNumRows()-(i+1); p = l.getIndex(i, j, k); )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_DistributeFirstSlice, 		distributeFirstSlice,		p = l.getIndex(i, j, 0); k=0; )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_DistributeFirstVector,		distributeFirstVector,		p = l.getIndex(i, 0, k); j=0; )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_DistributeFirstElement,		distributeFirstElement,		p = l.getIndex(0, 0, 0); i=j=k=0; )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_diagonalVector,			diagonalVector,			p = l.getIndex(i, i, k); j=i; )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_distributeElementsOnColumns,	distributeElementsOnColumns,	p = l.getIndex(j, 0, k); i=j; j=0; )
	STANDARD_TRANSFORM_OPERATOR_DEFINITION( UnOp_distributeElementsOnSlices,	distributeElementsOnSlices,	p = l.getIndex(k, 0, 0); i=k; k=0; k=0; )

// Layout reinterpretation functions :
	STANDARD_LAYOUT_REINTERPRETATION_OPERATOR_DEFINITION( UnOp_clamp, 		clamp, 				i = lnew.getIClamped(i);
															j = lnew.getJClamped(j);
															k = lnew.getKClamped(k);
															p = lnew.getIndex(i, j, k); )
	STANDARD_LAYOUT_REINTERPRETATION_OPERATOR_DEFINITION( UnOp_repeat, 		repeat, 			i = lnew.getIWrapped(i);
															j = lnew.getJWrapped(j);
															k = lnew.getKWrapped(k);
															p = lnew.getIndex(i, j, k); )
	STANDARD_LAYOUT_REINTERPRETATION_OPERATOR_DEFINITION( UnOp_expandf,		expandf,			i = l.getINorm<float>(i)*lnew.getNumRows();
															j = l.getJNorm<float>(j)*lnew.getNumColumns();
															k = l.getKNorm<float>(k)*lnew.getNumSlices();
															p = lnew.getIndex(i, j, k); )
	STANDARD_LAYOUT_REINTERPRETATION_OPERATOR_DEFINITION( UnOp_expand,		expand,				i = l.getINorm<double>(i)*lnew.getNumRows();
															j = l.getJNorm<double>(j)*lnew.getNumColumns();
															k = l.getKNorm<double>(k)*lnew.getNumSlices();
															p = lnew.getIndex(i, j, k); )

// Binary functions : 
	/* Note that if the input arguments are not of the type specified by the Cuda library, the call might fail.
	   The error reported is : calling a __host__ function <function name>. */
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_min,			min,			::min(a, b) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_max,			max,			::max(a, b) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_atan2,			atan2, 			::atan2(static_cast<double>(a),static_cast<double>(b)) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_fmax,			fmax,			::fmax(static_cast<double>(a),static_cast<double>(b)) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_fmin,			fmin,			::fmin(static_cast<double>(a),static_cast<double>(b)) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_fmod,			fmod,			::fmod(static_cast<double>(a),static_cast<double>(b)) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_hypot,			hypot,			::hypot(static_cast<double>(a),static_cast<double>(b)) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_jn,			jn,			::jn(static_cast<double>(a),static_cast<double>(b)) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_ldexp,			ldexp,			::ldexp(static_cast<double>(a),static_cast<double>(b)) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_pow,			pow,			::pow(static_cast<double>(a),static_cast<double>(b)) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_remainder,		remainder,		::remainder(static_cast<double>(a),static_cast<double>(b)) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_yn,			yn,			::yn(static_cast<double>(a),static_cast<double>(b)) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_atan2f,			atan2f,			::atan2f(static_cast<float>(a),static_cast<float>(b)) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_fmaxf,			fmaxf,			::fmaxf(static_cast<float>(a),static_cast<float>(b)) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_fminf,			fminf,			::fminf(static_cast<float>(a),static_cast<float>(b)) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_fmodf,			fmodf,			::fmodf(static_cast<float>(a),static_cast<float>(b)) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_hypotf,			hypotf,			::hypotf(static_cast<float>(a),static_cast<float>(b)) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_jnf,			jnf,			::jnf(static_cast<float>(a),static_cast<float>(b)) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_ldexpf,			ldexpf,			::ldexpf(static_cast<float>(a),static_cast<float>(b)) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_powf,			powf,			::powf(static_cast<float>(a),static_cast<float>(b)) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_remainderf,		remainderf,		::remainderf(static_cast<float>(a),static_cast<float>(b)) )
	STANDARD_BINARY_OPERATOR_DEFINITION(	BinOp_ynf,			ynf,			::ynf(static_cast<float>(a),static_cast<float>(b)) )

	R2C_BINARY_OPERATOR_DEFINITION( 	BinOp_ToComplex, 		toComplex,		toComplex(a, b) )

} // Namespace Kartet

#endif
