#!/bin/bash

# for depth in 1 2
# do 
#     python3 hierarchical_classification.py --loss "cross_entropy" --kfold True --depth $depth
# done


elements=("1. 0." "0.9 0.1" "0.8 0.2" "0.7 0.3" "0.6 0.4" "0.4 0.6" "0.3 0.7" "0.2 0.8" "0.1 0.9", "0. 1.")
elements=("0.1 0.9")

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