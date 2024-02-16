#!/bin/bash

for file in frame_*.png
do
  # Extract the number part from the filename.
  number=$(echo $file | sed 's/frame_0*//;s/.png//')
  
  # Subtract 1 to start from 0.
  new_number=$(printf "%06d" $((number - 1)))
  
  # Rename the file.
  mv "$file" "frame_${new_number}.png"
done

