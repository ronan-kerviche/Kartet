function convertArrayFile(inputFilename, outputFilename)
	
	% Read the header : 
	fileId = fopen(inputFilename,'rb');
	assert(fileId>=0, 'Bad File ID.');
	header = fread(fileId, 8, 'char*1');

	% Rewind : 
	frewind(fileId);

	if(strcmp(char(header.'), 'KARTET01'))
		fprintf('Found KARTET01 header ...\n');
		A = KARTET01_readArrayFromFile(fileId);
	elseif(strcmp(char(header.'), 'KARTET02'))
		fprintf('Found KARTET02 header ...\n');
		A = readArrayFromFile(fileId);		% Current version.
	else
		error('Unknown header found in %s : %s', inputFilename, char(header.'));
	end

	% Write in current version : 
	fprintf('Writing converted array to %s ...\n', outputFilename);
	writeArrayToFile(A, outputFilename);
end

function A = KARTET01_readArrayFromFile(arg)
% function A = readArrayFromFile(arg)
%
% ARGUMENTS : 
% arg	: Either the filename (as a string) or a file ID (otherwise).
% RETURNS :
% A	: The array loaded, in its original type.

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
	
	% This list should match the list in TypeTools.hpp (note that void is at index 0 and is omitted)
	types = {	struct('typename', 'uint8', 'isComplex', false), ...		% bool
			struct('typename', 'uint8', 'isComplex', false), ...		% unsigned char
			struct('typename', 'int8', 'isComplex', false), ...		% char
			struct('typename', 'int8', 'isComplex', false), ...		% signed char
			struct('typename', 'uint16', 'isComplex', false), ...		% unsigned short
			struct('typename', 'int16', 'isComplex', false), ...		% short
			struct('typename', 'int16', 'isComplex', false), ...		% signed short
			struct('typename', 'uint32', 'isComplex', false), ...		% unsigned int
			struct('typename', 'int32', 'isComplex', false), ...		% int
			struct('typename', 'int32', 'isComplex', false), ...		% signed int
			struct('typename', 'uint32', 'isComplex', false), ...		% unsigned long
			struct('typename', 'int32', 'isComplex', false), ...		% long
			struct('typename', 'uint32', 'isComplex', false), ...		% signed long
			struct('typename', 'uint64', 'isComplex', false), ...		% unsigned long long
			struct('typename', 'int64', 'isComplex', false), ...		% long long
			struct('typename', 'int64', 'isComplex', false), ...		% signed long long
			struct('typename', 'single', 'isComplex', false), ...		% float
			struct('typename', 'double', 'isComplex', false), ...		% double
			struct('typename', 'NOTSUPPORTED', 'isComplex', false), ...	% (long double)
			struct('typename', 'single', 'isComplex', true), ...		% complex<float>
			struct('typename', 'double', 'isComplex', true), ...		% complex<double>
			};

	assert(T>=1 && T<=numel(types), sprintf('Unknown type code : %d.', T));

	% Get the data :
	nElements = R*C*S;
	if(types{T}.isComplex)
		nElements = nElements * 2;
	end
	
	A = fread(fileId, nElements, sprintf('*%s', types{T}.typename));
	if(types{T}.isComplex)
		A = A(1:2:end) + 1i* A(2:2:end);
	end
	A = reshape(A, [R, C, S]);
	
	% Close, if needed :
	if(isstr(arg))
		fclose(fileId);
	end
end

