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
	for typeId=1:numel(types)
		if(strcmpi(types{typeId}.typename,cName))
			break;
		end
	end
	assert(typeId<numel(types) || strcmpi(types{numel(types)}.typename,cName), sprintf('Array type ''%s''is not supported.', cName));
	
	% Open the file, if needed :
	if(isstr(arg))
		fileId = fopen(arg,'wb');
	else
		fileId = arg;
	end

	assert(fileId>=0, 'Bad File ID.');

	% This header should match the header in Array.hpp :
	fwrite(fileId, 'KARTET02', 'char*1');

	% Write the remaining header parts :
	fwrite(fileId, typeId, '*int32');
	fwrite(fileId, uint8(isComplex), '*uint8');
	fwrite(fileId, size(A,1), '*int64');
	fwrite(fileId, size(A,2), '*int64');
	fwrite(fileId, size(A,3), '*int64');
	fwrite(fileId, A(:), sprintf('*%s', types{typeId}.typename));

	% Close, if needed :
	if(isstr(arg))
		fclose(fileId);
	end
end

