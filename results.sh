#!/bin/bash

for depth in 1 2
do 
    python3 hierarchical_classification.py --loss "cross_entropy" --kfold True --depth $depth
done


elements=("1. 0." "0.75 0.25" "0.5 0.5" "0.25 0.75" "0. 1.")

    # Iterate over the elements array
for element in "${elements[@]}"
do
    # Split the string representation into individual values
    IFS=' ' read -r -a tuple <<< "$element"

    # Access each value in the tuple
    value1="${tuple[0]}"
    value2="${tuple[1]}"

    # Print the values
    echo "Value 1: $value1"
    echo "Value 2: $value2"

    python3 hierarchical_classification.py --loss "hierarchical_bce_loss" --kfold True --weights $value1 $value2 --depth 2
done