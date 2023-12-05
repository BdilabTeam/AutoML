#!/bin/bash

namespace="dljobs"

cd "jobs"
# Get the list of pods in the specified namespace
pod_list=$(kubectl get pods -n "$namespace" --no-headers -o custom-columns=":metadata.name")

# Extract unique prefixes from pod names
pod_prefixes=($(echo "$pod_list" | sed -nE 's/^([^-]+-[0-9]+-[0-9]+-[0-9]+-[0-9]+)-.*/\1/p' | sort -u))

# Iterate over the unique pod prefixes
for prefix in "${pod_prefixes[@]}"; do
    # Formulate the YAML filename for deletion
    yaml_file="${prefix}.yaml"

    # Check if the YAML file exists
    if [ -f "$yaml_file" ]; then
        echo "Deleting pods related to $yaml_file"
        kubectl delete -f "$yaml_file" -n "$namespace"
    else
        echo "YAML file $yaml_file not found"
    fi
done
de