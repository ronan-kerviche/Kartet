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
	\file    GetArguments.hpp
	\brief   Arguments tools.
	\author  R. Kerviche
	\date    August 17th 2017
**/

/**
	Example :
	\code
		VecArgument<3, float> positionArg(makeVec3(0, 0, 0), "Position", "Position of the object.");
		MatArgument<3, 3, float> attitudeArg(Mat3d::identity(), "Attitude", "Attitude of the object.");
		std::map<std::string, AbstractArgument*> argumentsList;
		argumentsList["position"] = &positionArg;
		argumentsList["attitude"] = &attitudeArg;

		if(!getArguments(argc, argv, argumentsList))
			throw Kartet::InvalidArgument;
		printArguments(argumentsList);

		std::cout << "Position : " << positionArg.val << std::endl;
		std::cout << "Attitude : " << attitudeArg.val << std::endl;

		// Run with : ./program --position 1,2,3 --attitude "1,0,0;0,1,0;0,0,1"
	\endcode
**/

#ifndef __GET_ARGUMENTS__
#define __GET_ARGUMENTS__

	#include <map>
	#include <string>
	#include <sstream>
	#include <limits>
	#include "Kartet.hpp"

	/**
	\brief Convert string to output type.
	\tparam T Output type.
	\param src Input string.
	\param dest Output value.
	\param strict Enable strict processing, for numbers.
	\return True if the conversion was successful.
	**/
	template<typename T>
	bool fromString(const std::string& str, T& dest, const bool& strict=true)
	{
		if(strict)
		{
			const char spaces[] = " \n\t";
			bool 	rlDot = true,
				rlSign = true,
				rlExp = false;
			size_t	a = str.find_first_not_of(spaces),
				b = str.find_last_not_of(spaces);
			for(size_t k=a; k<=b; k++)
			{
				const char c = str[k];
				if((c=='.' && !rlDot) || ((c=='+' || c=='-') && !rlSign) || ((c=='e' || c=='E') && !rlExp))
					return false;
				if(c>='0' && c<='9')
				{
					rlSign=false;
					rlExp = true;
				}
				else if(c=='-' || c=='+')
					rlSign = false;
				else if(c=='.')
					rlDot=false;
				else if(c=='e' || c=='E')
				{
					rlSign = true;
					rlDot = false;
					rlExp = false;
				}
				else
					return false;
			}
		}
		std::istringstream iss(str);
		return static_cast<bool>(iss >> dest);
	}

	/**
	\brief Convert a value to a string.
	\tparam T Input type.
	\param value Input value, to be converted.
	\return A string containing the human readable value.
	**/
	template<typename T>
	std::string toString(const T& value)
	{
		std::ostringstream oss;
		oss << value;
		return oss.str();
	}

	/**
	\brief Abstract argument class.
	**/
	struct AbstractArgument
	{
		/// If the argument was set.
		bool set;
		/// Name of the argument.
		const std::string name;
		AbstractArgument(const std::string& _name);
		virtual ~AbstractArgument(void);
		virtual bool read(const std::string& str, const std::string& arg="") = 0;
		virtual std::string help(void) const;
		virtual std::string value(void) const = 0;
	};

	/**
	\brief Scalar abstract argument.
	\tparam T Scalar type.
	**/
	template<typename T>
	struct ScalarArgument : public AbstractArgument
	{
		/// Value.
		T	val,
		/// Minimum authorized value.
			minVal,
		/// Maximum authorized value.
			maxVal;
		/// Base help.
		const std::string baseHelp;

		/**
		\brief ScalarArgument constructor.
		\param _val The default value.
		\param _name Name of the argument.
		\param _baseHelp Base help of the argument.
		\param _minVal Minimum authorized value.
		\param _maxVal Maximum authorized value.
		**/
		ScalarArgument(const T& _val, const std::string& _name, const std::string& _baseHelp, const T& _minVal=std::numeric_limits<T>::min(), const T& _maxVal=std::numeric_limits<T>::max())
		 :	AbstractArgument(_name),
			val(_val),
			minVal(_minVal),
			maxVal(_maxVal),
			baseHelp(_baseHelp)
		{ }

		/**
		\brief Destructor.
		**/
		virtual ~ScalarArgument(void)
		{ }

		/**
		\brief Read function.
		\param str Input string.
		\param arg Identifier of the argument
		\return True if the string was read successfully.
		**/
		bool read(const std::string& str, const std::string& arg="")
		{
			if(!fromString(str, val))
			{
				std::cout << "Cannot read value for integer argument \"" << name << "\" (" << arg << ") : \"" << str << "\"." << std::endl;
				return false;
			}
			else if(val<minVal || val>maxVal)
			{
				std::cerr << "Integer argument \"" << name << "\" (" << arg << ") failed the range validation : " << val << " is out of the range [" << minVal << "; " << maxVal << "]." << std::endl;
				return false;
			}
			else
				return true;
		}

		/**
		\brief Returns the formatted help.
		\return A string containing the formatted help.
		**/
		std::string help(void) const
		{
	                return baseHelp + " Default value : " + toString(val) + "; Range : [" + toString(minVal) + "; " + toString(maxVal) + "].";
	        }

		/**
		\brief Returns the current value.
		\return A string containing the current formatted value.
		**/
		std::string value(void) const
		{
			return toString(val);
		}
	};

	template<int r, int c, typename T>
	struct MatArgument : public AbstractArgument
	{
		/// Value.
		Kartet::Mat<r, c, T> val;
		/// Base help.
		const std::string baseHelp;

		/**
		\brief MatArgument constructor.
		\param _val The default value.
		\param _name Name of the argument.
		\param _baseHelp Base help of the argument.
		**/
		MatArgument(const Kartet::Mat<r, c, T>& _val, const std::string& _name, const std::string& _baseHelp)
		 :	AbstractArgument(_name),
			val(_val),
			baseHelp(_baseHelp)
		{ }

		/**
		\brief Destructor.
		**/
		virtual ~MatArgument(void)
		{ }

		/**
		\brief Read function.
		\param str Input string.
		\param arg Identifier of the argument
		\return True if the string was read successfully.
		**/
		bool read(const std::string& str, const std::string& arg="")
		{
			const char separators[] = ",;";
			size_t pos = 0;
			T vals[r*c];
			for(int k=0; k<r; k++)
			{
				for(int l=0; l<r; l++)
				{
					size_t 	nextPos = str.find_first_of(separators, pos);
					if(nextPos==std::string::npos)
					{
						if(l==c-1 && k==r-1)
							nextPos = str.size();
						else
						{
							std::cerr << "Missing component (" << k+1 << ',' << l+1 << " in " << r << 'x' << c << " matrix argument \"" << name << "\" (" << arg << ") : \"" << str << "\"." << std::endl;
							return false;
						}
					}
					if(!fromString(str.substr(pos, nextPos-pos), vals[l*r+k]))
					{
						std::cerr << "Cannot read value for " << r << 'x' << c << " matrix argument \"" << name << "\" (" << arg << ") : \"" << str.substr(pos, nextPos-pos) << "\"." << std::endl;
						return false;
					}
					pos = nextPos+1;
				}
			}
			val.set(vals);
			return true;
		}

		/**
		\brief Returns the formatted help.
		\return A string containing the formatted help.
		**/
		std::string help(void) const
		{
        	        return baseHelp + " In row-major order. Default value : " + value() + ".";
        	}

		/**
		\brief Returns the current value.
		\return A string containing the current formatted value.
		**/
		std::string value(void) const
		{
			std::string str = "";
			for(int k=0; k<r; k++)
			{
				for(int l=0; l<c; l++)
				{
					str += toString(val(k,l));
					if(l<c-1)
						str += ',';
					else if(k<r-1)
						str += ';';
				}
			}
			return str;
		}
	};

	template<int r, typename T>
	struct VecArgument : public AbstractArgument
	{
		/// Value.
		Kartet::Vec<r, T> val;
		/// Base help.
		const std::string baseHelp;

		/**
		\brief VecArgument constructor.
		\param _val The default value.
		\param _name Name of the argument.
		\param _baseHelp Base help of the argument.
		**/
		VecArgument(const Kartet::Vec<r, T>& _val, const std::string& _name, const std::string& _baseHelp)
		 :	AbstractArgument(_name),
                	val(_val),
                	baseHelp(_baseHelp)
		{ }

		/**
		\brief Destructor.
		**/
		virtual ~VecArgument(void)
		{ }

		/**
		\brief Read function.
		\param str Input string.
		\param arg Identifier of the argument
		\return True if the string was read successfully.
		**/
		bool read(const std::string& str, const std::string& arg="")
		{
			const char separators[] = ",;";
			size_t pos = 0;
			T vals[r];
			for(int k=0; k<r; k++)
			{
				size_t nextPos = str.find_first_of(separators, pos);
				if(nextPos==std::string::npos)
				{
					if(k==r-1)
						nextPos = str.size();
					else
					{
						std::cerr << "Missing component " << k+1 << " in " << r << "D vector argument \"" << name << "\" (" << arg << ") : \"" << str << "\"." << std::endl;
						return false;
					}
				}
				if(!fromString(str.substr(pos, nextPos-pos), vals[k]))
				{
					std::cerr << "Cannot read value for " << r << "D vector argument \"" << name << "\" (" << arg << ") : \"" << str.substr(pos, nextPos-pos) << "\"." << std::endl;
					return false;
				}
				pos = nextPos+1;
			}
			if(str.find_first_not_of(" \t\n", pos)!=std::string::npos)
			{
				std::cerr << "Trailing characters for " << r << "D vector argument \"" << name << "\" (" << arg << ") : \"" << str << "\"." << std::endl;
				return false;
			}
			val.set(vals);
			return true;
		}

		/**
		\brief Returns the formatted help.
		\return A string containing the formatted help.
		**/
		std::string help(void) const
		{
        	        return baseHelp + " Default value : " + value() + ".";
        	}

		/**
		\brief Returns the current value.
		\return A string containing the current formatted value.
		**/
		std::string value(void) const
		{
			std::string str = "";
			for(int k=0; k<r; k++)
			{
				str += toString(val(k));
				if(k<r-1)
					str += ',';
			}
			return str;
		}
	};

	struct StringArgument : public AbstractArgument
	{
		/// Value.
		std::string val;
		/// Base help.
		const std::string baseHelp;

		StringArgument(const std::string& _val, const std::string& _name, const std::string& _baseHelp);
		virtual ~StringArgument(void);
		bool read(const std::string& str, const std::string& arg="");
		std::string help(void) const;
		std::string value(void) const;
	};

	void printHelp(const std::map<std::string, AbstractArgument*>& arguments, const std::string& programName);
	bool getArguments(const std::vector<std::string>& args, const std::map<std::string, AbstractArgument*>& arguments, const std::string& programName="");
	bool getArguments(const int& argc, char const* const* argv, const std::map<std::string, AbstractArgument*>& arguments);
	bool getArguments(const std::string& str, const std::map<std::string, AbstractArgument*>& arguments, const std::string& programName="");
	void printArguments(const std::map<std::string, AbstractArgument*>& arguments);

#endif

