import os
import sys
import time
import logging
from datetime import datetime

from src.config import RESULTS_DIR, LOGS_DIR, EXPERIMENT_ID, CONTROL_GROUP, RESULTS_FILE
from src.train import run_cross_validation
from src.utils import save_summary, set_seed

def setup_logging():
    """Set up logging configuration."""
    log_file = os.path.join(LOGS_DIR, f'experiment_{EXPERIMENT_ID}_{CONTROL_GROUP}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def main():
    """Main function to run the experiment workflow."""
    # Set up logging
    logger = setup_logging()
    
    # Log experiment start
    logger.info(f"Starting experiment {EXPERIMENT_ID} - {CONTROL_GROUP}")
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Set seed for reproducibility
    set_seed()
    
    try:
        # Run cross-validation
        logger.info("Running cross-validation...")
        start_time = time.time()
        summary = run_cross_validation()
        end_time = time.time()
        
        # Log experiment completion
        logger.info(f"Cross-validation completed in {(end_time - start_time) / 60:.2f} minutes")
        logger.info(f"Average Validation Kappa: {summary['avg_val_kappa']:.4f}")
        logger.info(f"Average Generalization Gap: {summary['avg_gen_gap']:.4f}")
        
        # Save summary results
        results_path = os.path.join(RESULTS_DIR, RESULTS_FILE)
        save_summary(summary, results_path)
        logger.info(f"Results saved to {results_path}")
        
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Error during experiment: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()