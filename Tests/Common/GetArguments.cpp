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
	\file    GetArguments.cpp
	\brief   Arguments tools.
	\author  R. Kerviche
	\date    August 17th 2017
**/

#include <iostream>
#include "GetArguments.hpp"

// AbstractArgument :
	/**
	\brief Constructor.
	\param _name Name of the argument.
	**/
	AbstractArgument::AbstractArgument(const std::string& _name)
	 :	set(false),
		name(_name)
	{ }

	/**
	\brief Destructor.
	**/
	AbstractArgument::~AbstractArgument(void)
	{ }

	/**
	\brief Test if the argument takes a value.
	\return True if the argument takes a value.
	**/
	bool AbstractArgument::takeValue(void) const
	{
		return true;
	}

	bool AbstractArgument::read(const std::string&, const std::string&)
	{
		return true;
	}

	/**
	\brief Returns the formatted help.
	\return A string containing the formatted help.
	**/
	std::string AbstractArgument::help(void) const
	{
		return "";
	}

// ToggleArgument :
	ToggleArgument::ToggleArgument(const std::string& _name, const std::string& _baseHelp)
	 :	AbstractArgument(_name),
		baseHelp(_baseHelp)
	{ }

	ToggleArgument::~ToggleArgument(void)
	{ }

	bool ToggleArgument::takeValue(void) const
	{
		return false;
	}

	/**
	\brief Returns the formatted help.
	\return A string containing the formatted help.
	**/
	std::string ToggleArgument::help(void) const
	{
                return baseHelp;
        }

	/**
	\brief Returns the current value.
	\return A string containing the current formatted value.
	**/
	std::string ToggleArgument::value(void) const
	{
		if(set)
			return "set";
		else
			return "unset";
	}

// StringArgument :
	/**
	\brief StringArgument constructor.
	\param _val The default value.
	\param _name Name of the argument.
	\param _baseHelp Base help of the argument.
	**/
	StringArgument::StringArgument(const std::string& _val, const std::string& _name, const std::string& _baseHelp)
	 :	AbstractArgument(_name),
		val(_val),
		baseHelp(_baseHelp)
	{ }

	/**
	\brief Destructor.
	**/
	StringArgument::~StringArgument(void)
	{ }

	/**
	\brief Read function.
	\param str Input string.
	\param arg Identifier of the argument
	\return True if the string was read successfully.
	**/
	bool StringArgument::read(const std::string& str, const std::string&)
	{
		val = str;
		return !val.empty();
	}

	/**
	\brief Returns the formatted help.
	\return A string containing the formatted help.
	**/
	std::string StringArgument::help(void) const
	{
		return baseHelp + " Default value : \"" + val + "\".";
	}

	/**
	\brief Returns the current value.
	\return A string containing the current formatted value.
	**/
	std::string StringArgument::value(void) const
	{
		return val;
	}

// Tools :
	size_t longestArgumentSize(const std::map<std::string, AbstractArgument*>& arguments)
	{
		size_t s = 0;
		for(std::map<std::string, AbstractArgument*>::const_iterator it=arguments.begin(); it!=arguments.end(); it++)
			s = std::max(s, it->first.size());
		return s;
	}

	/**
	\brief Display the arguments help.
	\param arguments The list of arguments.
	\param programName The name of the program.
	**/
	void printHelp(const std::map<std::string, AbstractArgument*>& arguments, const std::string& programName)
	{
		const std::string argPrefix = "--";
		const size_t longest = longestArgumentSize(arguments);
		std::cout << "Help : " << std::endl;
		std::cout << programName << " [Arguments...]" << std::endl;
		std::cout << "Arguments : " << std::endl;
		for(std::map<std::string, AbstractArgument*>::const_iterator it=arguments.begin(); it!=arguments.end(); it++)
			std::cout << "  " << argPrefix << it->first << std::string(longest-it->first.size(), ' ') << " : " << it->second->help() << std::endl;
	}

	/**
	\brief Parse arguments.
	\param args The input arguments.
	\param arguments The list of arguments to be matched against.
	\param programName The name of the program.
	\return True if the parsing operation was successful.

	'--help' is reserved to automatically display the help and return false.
	**/
	bool getArguments(const std::vector<std::string>& args, const std::map<std::string, AbstractArgument*>& arguments, const std::string& programName)
	{
		const std::string argPrefix = "--";
		for(std::vector<std::string>::const_iterator it=args.begin(); it!=args.end(); it++)
		{
			if(it->size()<=argPrefix.size() || it->substr(0, argPrefix.size())!=argPrefix)
			{
				std::cerr << "Unknown argument : " << *it << std::endl;
				std::cerr << "To print the help : " << programName << ' ' << argPrefix << "help" << std::endl;
				return false;
			}
			if(it->substr(argPrefix.size())=="help")
			{
				printHelp(arguments, programName);
				return false;
			}
			std::map<std::string, AbstractArgument*>::const_iterator ita=arguments.find(it->substr(argPrefix.size()));
			if(ita==arguments.end())
			{
				std::cerr << "Invalid argument : " << *it << std::endl;
				std::cerr << "To print the help : " << programName << ' ' << argPrefix << "help" << std::endl;
				return false;
			}


			if(ita->second->takeValue())
			{
				it++;
				if(it==args.end())
				{
					std::cerr << "Missing value for argument : " << argPrefix << ita->first << ", \"" << ita->second->name << "\"." << std::endl;
					std::cerr << "To print the help : " << programName << ' ' << argPrefix << "help" << std::endl;
				}
				else if(ita->second->set)
				{
					std::cerr << "Argument \"" << ita->second->name << "\" (with new value : " << *it << ") was already set." << std::endl;
					return false;
				}
				else if(!ita->second->read(*it, argPrefix + ita->first))
					return false;
				else
					ita->second->set = true;
			}
			else
			{
				if(ita->second->set)
				{
					std::cerr << "Argument \"" << ita->second->name << "\" was already toggled." << std::endl;
					return false;
				}
				else
					ita->second->set = true;
			}
		}
		return true;
	}

	/**
	\brief Parse arguments.
	\param argc The number of arguments.
	\param argv The arguments.
	\param arguments The list of arguments to be matched against.
	\return True if the parsing operation was successful.

	'--help' is reserved to automatically display the help and return false.
	**/
	bool getArguments(const int& argc, char const* const* argv, const std::map<std::string, AbstractArgument*>& arguments)
	{
		const std::string programName(argv[0]);
		std::vector<std::string> args;
		for(int k=1; k<argc; k++)
			args.push_back(std::string(argv[k]));
		return getArguments(args, arguments, programName);
	}

	/**
	\brief Parse arguments.
	\param str A string containing a list of space separated arguments.
	\param arguments The list of arguments to be matched against.
	\param programName The name of the program.
	\return True if the parsing operation was successful.

	'--help' is reserved to automatically display the help and return false.
	**/
	bool getArguments(const std::string& str, const std::map<std::string, AbstractArgument*>& arguments, const std::string& programName)
	{
		const char spaces[] = " \t\n";
		std::vector<std::string> args;
		size_t pos = 0;
		while(pos<=str.size() && pos!=std::string::npos)
		{
			const size_t	b = str.find_first_not_of(spaces, pos),
					e = (b!=std::string::npos) ? str.find_first_of(spaces, b) : std::string::npos;
			if(b!=std::string::npos)
			{
				if(e!=std::string::npos)
					args.push_back(str.substr(b, e-b));
				else
					args.push_back(str.substr(b));
			}
			pos = e;
		}
		return getArguments(args, arguments, programName);
	}

	/**
	\brief Print the values of the arguments.
	\param arguments The list of arguments.
	**/
	void printArguments(const std::map<std::string, AbstractArgument*>& arguments)
	{
		int setCount = 0;
		for(std::map<std::string, AbstractArgument*>::const_iterator it=arguments.begin(); it!=arguments.end(); it++)
			setCount += (it->second->set ? 1 : 0);
		std::cout << arguments.size() << " Argument(s) (" << setCount << " set) : " << std::endl;
		for(std::map<std::string, AbstractArgument*>::const_iterator it=arguments.begin(); it!=arguments.end(); it++)
		{
			if(it->second->set)
				std::cout << "  [SET] ";
			else
				std::cout << "  [DEF] ";
			std::cout << it->second->name << " : " << it->second->value() << std::endl;
		}
	}

