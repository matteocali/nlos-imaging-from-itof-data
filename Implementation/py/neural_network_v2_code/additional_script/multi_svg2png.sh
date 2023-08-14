#!/bin/bash

cd '../results/test using the straight through estimator'

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

