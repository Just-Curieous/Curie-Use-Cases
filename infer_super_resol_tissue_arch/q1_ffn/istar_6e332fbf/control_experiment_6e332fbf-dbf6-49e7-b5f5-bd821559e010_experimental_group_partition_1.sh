#!/bin/bash

# Experimental Group Partition 1 script for neural network imputation task
# UUID: 6e332fbf-dbf6-49e7-b5f5-bd821559e010
# Group: experimental_group_partition_1

# Set paths
WORKSPACE="/workspace/istar_6e332fbf-dbf6-49e7-b5f5-bd821559e010"
VENV_PATH="/workspace/istar_6e332fbf-dbf6-49e7-b5f5-bd821559e010/venv"
PYTHON_PATH="$VENV_PATH/bin/python"
RESULTS_FILE="$WORKSPACE/results_6e332fbf-dbf6-49e7-b5f5-bd821559e010_experimental_group_partition_1.txt"
DATA_DIR="$WORKSPACE/data/demo/"

# Ensure output directory exists
mkdir -p "$WORKSPACE/states"
mkdir -p "$WORKSPACE/cnts-super"

# Start logging
{
    echo "=== Experimental Group: Neural Network Architecture Comparison ==="
    echo "UUID: 6e332fbf-dbf6-49e7-b5f5-bd821559e010"
    echo "Group: experimental_group_partition_1"
    echo "Start time: $(date)"
    echo ""

    # Check GPU availability
    echo "=== GPU Information ==="
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        echo "GPU is available for computation"
        DEVICE="cuda"
    else
        echo "No GPU found, using CPU"
        DEVICE="cpu"
    fi
    echo ""

    # Check Python environment
    echo "=== Python Environment ==="
    $PYTHON_PATH --version
    echo ""

    # Define the five architectures to test
    declare -A architectures=(
        ["deeper_network"]="--hidden-layers=256,128,64 --activation=relu --learning-rate=0.001 --optimizer=adam"
        ["wider_network"]="--hidden-layers=512,512 --activation=relu --learning-rate=0.001 --optimizer=adam"
        ["with_dropout"]="--hidden-layers=256,128 --activation=relu --learning-rate=0.001 --optimizer=adam --dropout=0.2"
        ["with_batch_norm"]="--hidden-layers=256,128 --activation=relu --learning-rate=0.001 --optimizer=adam --batch-norm"
        ["optimizer_variation"]="--hidden-layers=256,128 --activation=relu --learning-rate=0.001 --optimizer=adamw --weight-decay=0.01"
    )

    # Create a temporary directory for storing individual results
    TEMP_DIR="$WORKSPACE/temp_results"
    mkdir -p "$TEMP_DIR"

    # Initialize results summary
    echo "=== Architecture Comparison Results ===" > "$TEMP_DIR/summary.txt"
    echo "Architecture,Final RMSE,Training Time (s)" >> "$TEMP_DIR/summary.txt"

    # Run each architecture
    for arch_name in "${!architectures[@]}"; do
        echo ""
        echo "=== Testing Architecture: $arch_name ==="
        echo "Configuration: ${architectures[$arch_name]}"
        
        # Clean up previous states if they exist
        rm -rf "$WORKSPACE/states"
        mkdir -p "$WORKSPACE/states"
        
        # Clean up previous results file
        rm -f "$WORKSPACE/rmse_results.txt"
        
        # Start timing
        start_time=$(date +%s)
        
        # Run the experiment with the specific architecture
        echo "Command: $PYTHON_PATH $WORKSPACE/flexible_nn.py $DATA_DIR --epochs=400 --device=$DEVICE ${architectures[$arch_name]}"
        cd $WORKSPACE
        $PYTHON_PATH $WORKSPACE/flexible_nn.py $DATA_DIR --epochs=400 --device=$DEVICE ${architectures[$arch_name]}
        
        # End timing
        end_time=$(date +%s)
        training_time=$((end_time - start_time))
        
        # Check if experiment completed successfully
        if [ $? -eq 0 ]; then
            echo "=== $arch_name Completed Successfully ==="
            
            # Copy the results file to a unique name
            if [ -f "$WORKSPACE/rmse_results.txt" ]; then
                cp "$WORKSPACE/rmse_results.txt" "$TEMP_DIR/${arch_name}_results.txt"
                
                # Extract the final RMSE (assuming it's the last entry)
                final_rmse=$(tail -n 1 "$WORKSPACE/rmse_results.txt" | cut -d',' -f2)
                
                # Add to summary
                echo "$arch_name,$final_rmse,$training_time" >> "$TEMP_DIR/summary.txt"
                
                echo "Final RMSE for $arch_name: $final_rmse"
                echo "Training time: $training_time seconds"
            else
                echo "No results file found for $arch_name"
                echo "$arch_name,N/A,$training_time" >> "$TEMP_DIR/summary.txt"
            fi
        else
            echo "=== $arch_name Failed ==="
            echo "$arch_name,FAILED,$training_time" >> "$TEMP_DIR/summary.txt"
        fi
        
        echo ""
    done

    # Compile all results into the final results file
    echo "=== Compiling Final Results ==="
    
    # Add individual architecture results
    for arch_name in "${!architectures[@]}"; do
        echo "" >> "$TEMP_DIR/summary.txt"
        echo "=== Detailed Results for $arch_name ===" >> "$TEMP_DIR/summary.txt"
        
        if [ -f "$TEMP_DIR/${arch_name}_results.txt" ]; then
            # Add configuration
            grep "^# " "$TEMP_DIR/${arch_name}_results.txt" >> "$TEMP_DIR/summary.txt"
            
            # Add training metrics header
            grep "^epoch" "$TEMP_DIR/${arch_name}_results.txt" >> "$TEMP_DIR/summary.txt"
            
            # Add every 50th epoch for brevity
            grep -v "^#" "$TEMP_DIR/${arch_name}_results.txt" | grep -v "^epoch" | awk 'NR % 50 == 0 || NR == 1' >> "$TEMP_DIR/summary.txt"
        else
            echo "No results available" >> "$TEMP_DIR/summary.txt"
        fi
    done
    
    # Find the best architecture based on RMSE
    echo "" >> "$TEMP_DIR/summary.txt"
    echo "=== Best Architecture ===" >> "$TEMP_DIR/summary.txt"
    best_arch=$(grep -v "Architecture" "$TEMP_DIR/summary.txt" | grep -v "===" | grep -v "^$" | sort -t',' -k2,2n | head -n 1)
    if [ -n "$best_arch" ]; then
        echo "$best_arch" >> "$TEMP_DIR/summary.txt"
        best_name=$(echo "$best_arch" | cut -d',' -f1)
        best_rmse=$(echo "$best_arch" | cut -d',' -f2)
        echo "Best architecture: $best_name with RMSE $best_rmse" >> "$TEMP_DIR/summary.txt"
    else
        echo "Could not determine best architecture" >> "$TEMP_DIR/summary.txt"
    fi
    
    # Copy the summary to the results file
    cat "$TEMP_DIR/summary.txt" > "$RESULTS_FILE"
    
    echo ""
    echo "Results saved to: $RESULTS_FILE"
    echo ""
    echo "End time: $(date)"
    echo "=== Experiment Complete ==="

} 2>&1 | tee "$RESULTS_FILE"

# Clean up temporary directory
rm -rf "$TEMP_DIR"