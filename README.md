This code version runs perfectly with changed Necker.py file.
Now primarily two files are changed in this branch.
Adapter and Necker.py. The training .9438, AUROC 



This is what is included in Necker.py
This what 

### Plan for Error-Free Implementation

1. **Analyze Error**
- Channel mismatch in MultiScaleFusion
- Input: 768 channels
- Expected: 1024 channels

2. **Fix Steps**
- Use input channel size for convolutions
- Maintain channel dimensions through operations
- Validate dimensions before concatenation

```python


python train.py --config_path config/brainmri.yaml
```

Key fixes:
- Corrected channel dimensions in MultiScaleFusion
- Used input channel size throughout
- Maintained channel consistency in all operations
- Added proper dimension checks