#!/bin/bash

cd results/itof_ssim_grad_batch_results

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

