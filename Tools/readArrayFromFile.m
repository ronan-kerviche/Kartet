function A = readArrayFromFile(arg)

	if(isstr(arg))
		fileId = fopen(arg,'rb');
	else
		fileId = arg;
	end

	assert(fileId>=0, 'Bad File ID.');

	% Read the header :
	header = fread(fileId, 8, 'char*1');
	
	if(strcmp(char(header.'), 'KARTET01')==0)
		fclose(fileId);
		error('Bad header : %s', header);
	end

	% Read the sizes :
	T = fread(fileId, 1, '*int32');
	R = fread(fileId, 1, '*int64');
	C = fread(fileId, 1, '*int64');
	S = fread(fileId, 1, '*int64');
	
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
			struct('typename', 'float', 'isComplex', false), ...		% float
			struct('typename', 'double', 'isComplex', false), ...		% double
			struct('typename', 'NOTSUPPORTED', 'isComplex', false), ...
			struct('typename', 'float', 'isComplex', true), ...		% float
			struct('typename', 'double', 'isComplex', true), ...		% double
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
end
