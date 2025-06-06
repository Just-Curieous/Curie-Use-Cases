
Dataset directory: None.                                 Briefly explore the directory `ls * | head -n 50` to understand the structure.
# Inferring Super-Resolution Tissue Architecture by Integrating Spatial Transcriptomics and Histology

To run the FFN  model training script
```python
python impute.py data/demo/ --epochs=400 --device='cuda' 
```
- You can modify `impute.py` to implement your experiment plan: `cp impute.py impute_plan_1.py`
- You only need to run `impute.py` relavent code to train and evaluate the model. 
- You should train model for 400 epochs.
- The result will be stored to `rmse_results.txt`
- The dataset is prepared already, do not create sythetic data.
- DO NOT run other files like `train.py` or write from scratch.
- Your environment has been setup.
- If you want to retrain model, you need to delete the existing model: `rm -r ${prefix}states`.