#!/bin/bash
for filename in $(find . -type f -iname '*'); do 
	h=$(head -c 8 "$filename");
	if [[ "$h" == "KARTET"* ]]; then
		if [ "$h" != "KARTET02" ]; then
			echo "[$h] $filename";
		else
			echo "[________] $filename";
		fi
	fi
done

