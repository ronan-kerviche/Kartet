function [T, R, C, S] = readLayoutFromFile(arg)
% function A = readLayoutFromFile(arg)
%
% ARGUMENTS : 
% arg		: Either the filename (as a string) or a file ID (otherwise).
% RETURNS :
% [T, R, C, S]	: The layout structure, T is the type index, R is the number
%		  of rows, C is the number of columns, S is the number of
%		  slices.

	if(isstr(arg))
		fileId = fopen(arg,'rb');
	else
		fileId = arg;
	end

	assert(fileId>=0, 'Bad File ID.');

	% Read the header :
	header = fread(fileId, 8, 'char*1');
	
	% This header should match the header in Array.hpp :
	if(strcmp(char(header.'), 'KARTET01')==0)
		fclose(fileId);
		error('Bad header : %s', header);
	end

	% Read the sizes :
	T = fread(fileId, 1, '*int32');
	R = fread(fileId, 1, '*int64');
	C = fread(fileId, 1, '*int64');
	S = fread(fileId, 1, '*int64');

	% Close, if needed :
	if(isstr(arg))
		fclose(fileId);
	end
end
