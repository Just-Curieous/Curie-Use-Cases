import os
import sys
import argparse
import traceback
from train import main as train_main

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Diabetic Retinopathy Detection')
    
    parser.add_argument('--train_csv', type=str, default='/workspace/mle_dataset/train.csv',
                        help='Path to the training CSV file')
    parser.add_argument('--test_csv', type=str, default='/workspace/mle_dataset/test.csv',
                        help='Path to the test CSV file')
    parser.add_argument('--train_img_dir', type=str, default='/workspace/mle_dataset/train_images',
                        help='Path to the training images directory')
    parser.add_argument('--test_img_dir', type=str, default='/workspace/mle_dataset/test_images',
                        help='Path to the test images directory')
    parser.add_argument('--output_dir', type=str, default='/workspace/mle_d4bfab9d-b8a0-41ef-9c6c-a28b75e69bc7/output',
                        help='Directory to save outputs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of epochs to train for')
    
    return parser.parse_args()

def main():
    """Main function"""
    try:
        # Parse arguments
        args = parse_args()
        
        # Print arguments
        print("Arguments:")
        for arg in vars(args):
            print(f"  {arg}: {getattr(args, arg)}")
        
        # Check if files and directories exist
        if not os.path.exists(args.train_csv):
            raise FileNotFoundError(f"Training CSV file not found: {args.train_csv}")
        
        if not os.path.exists(args.test_csv):
            raise FileNotFoundError(f"Test CSV file not found: {args.test_csv}")
        
        if not os.path.exists(args.train_img_dir):
            raise FileNotFoundError(f"Training images directory not found: {args.train_img_dir}")
        
        if not os.path.exists(args.test_img_dir):
            raise FileNotFoundError(f"Test images directory not found: {args.test_img_dir}")
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Run training and evaluation
        train_main(
            args.train_csv,
            args.test_csv,
            args.train_img_dir,
            args.test_img_dir,
            args.output_dir,
            args.batch_size,
            args.num_epochs
        )
        
        return 0
    
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())