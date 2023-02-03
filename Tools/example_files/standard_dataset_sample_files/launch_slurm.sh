#!/bin/bash

cd slurm_data/scripts

for i in *.sh
do
	sbatch $i
done
