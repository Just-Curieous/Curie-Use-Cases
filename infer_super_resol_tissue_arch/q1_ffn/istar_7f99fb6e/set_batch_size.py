#!/usr/bin/env python3
import sys
import os

# This script modifies the impute.py file to set a fixed batch size of 27

def set_batch_size(file_path, batch_size=27):
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Replace the dynamic batch size calculation with a fixed value
    old_line = "    batch_size = min(128, n_train//16)"
    new_line = f"    batch_size = {batch_size}  # Fixed batch size for control experiment"
    
    if old_line in content:
        content = content.replace(old_line, new_line)
        with open(file_path, 'w') as f:
            f.write(content)
        print(f"Successfully set batch size to {batch_size} in {file_path}")
        return True
    else:
        print(f"Could not find the batch size line in {file_path}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python set_batch_size.py <file_path> [batch_size]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 27
    
    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist")
        sys.exit(1)
    
    success = set_batch_size(file_path, batch_size)
    sys.exit(0 if success else 1)