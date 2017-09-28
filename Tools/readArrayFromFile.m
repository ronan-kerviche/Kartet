function A = readArrayFromFile(arg)
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
	header = fread(fileId, 8, 'char*1', 0, 'l');
	
	% This header should match the header in Array.hpp :
	if(strcmp(char(header.'), 'KARTET02')==0)
		fclose(fileId);
		error('Bad header : %s', header);
	end

	% Read the sizes :
	T = fread(fileId, 1, '*int32', 0, 'l');
	X = fread(fileId, 1, '*uint8', 0, 'l')>0;
	R = fread(fileId, 1, '*int64', 0, 'l');
	C = fread(fileId, 1, '*int64', 0, 'l');
	S = fread(fileId, 1, '*int64', 0, 'l');
	
	% This list should match the list in TypeTools.hpp (note that void is at index 0 and is omitted)
	types = {	struct('typename', 'uint8'), ...		% bool
			struct('typename', 'int8'), ...			% char
			struct('typename', 'uint8'), ...		% unsigned char
			struct('typename', 'int16'), ...		% short
			struct('typename', 'uint16'), ...		% unsigned short
			struct('typename', 'int32'), ...		% int
			struct('typename', 'uint32'), ...		% unsigned int
			struct('typename', 'int64'), ...		% long long
			struct('typename', 'uint64'), ...		% unsigned long long
			struct('typename', 'single'), ...		% float
			struct('typename', 'double'), ...		% double
			struct('typename', 'NOTSUPPORTED'), ...		% (long double)
			};

	assert(T>=1 && T<=numel(types), sprintf('Unknown type code : %d.', T));

	% Get the data :
	nElements = R*C*S;
	if(X)
		nElements = nElements * 2;
	end
	
	A = fread(fileId, nElements, sprintf('*%s', types{T}.typename), 0, 'l');
	if(X)
		A = A(1:2:end) + 1i* A(2:2:end);
	end
	A = reshape(A, [R, C, S]);
	
	% Close, if needed :
	if(isstr(arg))
		fclose(fileId);
	end
end

