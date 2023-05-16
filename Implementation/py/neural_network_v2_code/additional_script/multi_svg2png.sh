#!/bin/bash

cd ../results/itof_out_new_batch

for d in */ ; do
    cd "$d"/plots
    mkdir svg
    for file in *; do
    	if [ -f "$file" ]; then 
        	mv $file ./svg
    	fi 
    done
    cd svg
    svg2png
    cd ../../..
done

