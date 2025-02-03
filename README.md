This code version runs perfectly with changed Necker.py file.
Now primarily two files are changed in this branch.
Adapter and Necker.py. The training started from 0.8825 AUROC in the first 3 epochs.
But later on never exceeded it.



This is what is included in Necker.py
This what 

### Plan for AUROC Improvements


```python

### Key Improvements:
1. Temperature-scaled layer weights
2. Feature refinement blocks with residual connections
3. Channel-wise attention mechanism
4. Skip connections between layers
5. Enhanced upsampling with conv+BN+ReLU
