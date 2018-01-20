#!/bin/bash
# Original author: Koustav Ghosal, V-SENSE, ghosalm@scss.tcd.ie
# Modified by Corentin Chéron 


if [ ! -f "./tiny-imagenet-200.zip" ]; then
  echo "Downloading tiny imagenet..."
  wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
fi


mkdir -p data
mkdir -p data/train
mkdir -p data/val

echo "extracting..."
unzip -qq tiny-imagenet-200.zip -d temp/

echo "creating training data..."
for folder in `ls temp/tiny-imagenet-200/train`
	do
		mkdir data/train/$folder/
		mkdir data/val/$folder/
		mv temp/tiny-imagenet-200/train/$folder/images/*.JPEG data/train/$folder/
	done

awk '{printf("%s %s\n",$1,$2)}' temp/tiny-imagenet-200/val/val_annotations.txt > ./validation_data.id

echo "creating validation data..."
while read -r line
do
    arr=($line)
	cp temp/tiny-imagenet-200/val/images/`echo ${arr[0]}` data/val/${arr[1]}/${arr[0]}
done < validation_data.id
echo "Complete"

rm -rf temp
