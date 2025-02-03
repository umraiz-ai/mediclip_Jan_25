This code version runs perfectly with changed Necker.py file.
Now primarily two files are changed in this branch.
Adapter and Necker.py. The training 0.9291, AUROC 
But later on never exceeded it.



This is what is included in Necker.py
This what 

### Plan for AUROC Improvements


```python

### Key Improvements:
### Step-by-Step Plan to Improve AUROC
### Plan for AUROC Improvements in 

Necker.py



### Plan for AUROC Improvements in 

Necker.py



1. **Analysis**
- Previous attempts focused on complex attention
- Need simpler but effective approach
- Focus on feature quality over quantity

2. **Strategy**
- Enhance token processing
- Add cross-scale interaction
- Improve feature calibration
- Keep computational overhead low

3. **Implementation Plan**
- Adaptive token mixing
- Scale-wise feature enhancement
- Learnable fusion weights
- Lightweight refinement blocks


4. **Key Changes**
- Simplified architecture
- Added scale-specific enhancement
- Learnable calibration parameters
- Improved feature mixing

5. **Test**
```bash
python train.py --config_path config/brainmri.yaml
```

This version focuses on quality of features rather than quantity of operations, which should help improve AUROC.