#!/bin/bash
#SBATCH --partition=cpu2,cpu4,cpu8,student,shared
#SBATCH --time 14-0:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=50
#SBATCH --job-name=dataset_batch0b_index
#SBATCH --output=../logs/batch0b_index/dataset_batch0b_index.out
#SBATCH --error=../logs/batch0b_index/dataset_batch0b_index.err
#SBATCH --mem=20G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=YOUR_EMAIL_ADDRESS
#SBATCH --requeue


cd "<path_to_folder>/xml_files/batch0b_index"
for i in *.xml
do
	echo $i
	file_name=$(basename $i .xml)
	batch_name="batch0b_index"
	
	out_folder="../../renders/$batch_name/$file_name"
	mkdir -p "$out_folder"
	echo "$out_folder"
	
	LD_LIBRARY_PATH="<path_to_folder>/mitsuba2-transient-nlos/build/dist/" "<path_to_folder>/mitsuba2-transient-nlos/build/dist/mitsuba" $i -o "${out_folder}."
done