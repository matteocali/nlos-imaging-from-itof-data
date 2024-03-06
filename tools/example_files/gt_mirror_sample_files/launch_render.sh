#!/bin/bash
#SBATCH --partition=cpu2,cpu4,cpu8,student,shared
#SBATCH --time 7-0:00:00 
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --job-name=depth_gt
#SBATCH --output=logs/depth_gt.out
#SBATCH --error=logs/depth_gt.err
#SBATCH --mem=20G
#SBATCH --mail-type=FAIL,END
#SBATCH --mail-user=YOUR_EMAIL_ADDRESS
#SBATCH --requeue

source <path_to_folder>/mitsuba2-transient-nlos/setpath.sh

echo "##### BATCH 01 #####"
cd xml_files/batch01

for i in *.xml
do
	echo $i
	file_name=$(basename "$i" .xml)
	
	out_folder="../../renders/batch01/$file_name"
	mkdir -p $out_folder
	
	LD_LIBRARY_PATH="<path_to_folder>/mitsuba2-transient-nlos/build/dist/" "<path_to_folder>/mitsuba2-transient-nlos/build/dist/mitsuba" $i -o "${out_folder}."
done
echo "####################"
echo ""

echo "##### BATCH 02 #####"
cd ../batch02

for i in *.xml
do
	echo $i
	file_name=$(basename "$i" .xml)
	
	out_folder="../../renders/batch02/$file_name"
	mkdir -p $out_folder
	
	LD_LIBRARY_PATH="<path_to_folder>/mitsuba2-transient-nlos/build/dist/" "<path_to_folder>/mitsuba2-transient-nlos/build/dist/mitsuba" $i -o "${out_folder}."
done
echo "####################"
echo ""

echo "##### BATCH 03 #####"
cd ../batch03

for i in *.xml
do
	echo $i
	file_name=$(basename "$i" .xml)
	
	out_folder="../../renders/batch03/$file_name"
	mkdir -p $out_folder
	
	LD_LIBRARY_PATH="<path_to_folder>/mitsuba2-transient-nlos/build/dist/" "<path_to_folder>/mitsuba2-transient-nlos/build/dist/mitsuba" $i -o "${out_folder}."
done
echo "####################"
echo ""

echo "##### BATCH 04 #####"
cd ../batch04

for i in *.xml
do
	echo $i
	file_name=$(basename "$i" .xml)
	
	out_folder="../../renders/batch04/$file_name"
	mkdir -p $out_folder
	
	LD_LIBRARY_PATH="<path_to_folder>/mitsuba2-transient-nlos/build/dist/" "<path_to_folder>/mitsuba2-transient-nlos/build/dist/mitsuba" $i -o "${out_folder}."
done
echo "####################"
echo ""

echo "##### BATCH 05 #####"
cd ../batch05

for i in *.xml
do
	echo $i
	file_name=$(basename "$i" .xml)
	
	out_folder="../../renders/batch05/$file_name"
	mkdir -p $out_folder
	
	LD_LIBRARY_PATH="<path_to_folder>/mitsuba2-transient-nlos/build/dist/" "<path_to_folder>/mitsuba2-transient-nlos/build/dist/mitsuba" $i -o "${out_folder}."
done
echo "####################"
echo ""

echo "##### BATCH 06 #####"
cd ../batch06

for i in *.xml
do
	echo $i
	file_name=$(basename "$i" .xml)
	
	out_folder="../../renders/batch06/$file_name"
	mkdir -p $out_folder
	
	LD_LIBRARY_PATH="<path_to_folder>/mitsuba2-transient-nlos/build/dist/" "<path_to_folder>/mitsuba2-transient-nlos/build/dist/mitsuba" $i -o "${out_folder}."
done
echo "####################"
echo ""

echo "##### BATCH 07 #####"
cd ../batch07

for i in *.xml
do
	echo $i
	file_name=$(basename "$i" .xml)
	
	out_folder="../../renders/batch07/$file_name"
	mkdir -p $out_folder
	
	LD_LIBRARY_PATH="<path_to_folder>/mitsuba2-transient-nlos/build/dist/" "<path_to_folder>/mitsuba2-transient-nlos/build/dist/mitsuba" $i -o "${out_folder}."
done
echo "####################"
echo ""

echo "##### BATCH 08 #####"
cd ../batch08

for i in *.xml
do
	echo $i
	file_name=$(basename "$i" .xml)
	
	out_folder="../../renders/batch08/$file_name"
	mkdir -p $out_folder
	
	LD_LIBRARY_PATH="<path_to_folder>/mitsuba2-transient-nlos/build/dist/" "<path_to_folder>/mitsuba2-transient-nlos/build/dist/mitsuba" $i -o "${out_folder}."
done
echo "####################"
echo ""