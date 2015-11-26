function writeArrayToFile(A, arg)
% function writeArrayToFile(A, arg)
%
% ARGUMENTS : 
% A	: The original array.
% arg	: Either the filename (as a string) or a file ID (otherwise).

	assert(~isempty(A),'Array is empty.');
	assert(ndims(A)<=3,'Array has more than three dimensions.');

	% Get the type data first :
	cName = class(A);
	isComplex = ~isreal(A);

	if(isComplex)
		% Force :
		A = double(A);
	end

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
	for typeId=1:numel(types)
		if(strcmpi(types{typeId}.typename,cName) && types{typeId}.isComplex==isComplex)
			break;
		end
	end
	
	% Open the file, if needed :
	if(isstr(arg))
		fileId = fopen(arg,'wb');
	else
		fileId = arg;
	end

	assert(fileId>=0, 'Bad File ID.');

	% This header should match the header in Array.hpp :
	fwrite(fileId, 'KARTET01', 'char*1');

	% Write the remaining header parts :
	fwrite(fileId, typeId, '*int32');
	fwrite(fileId, size(A,1), '*int64');
	fwrite(fileId, size(A,2), '*int64');
	fwrite(fileId, size(A,3), '*int64');
	fwrite(fileId, A(:), sprintf('*%s', types{typeId}.typename));

	% Close, if needed :
	if(isstr(arg))
		fclose(fileId);
	end
end

