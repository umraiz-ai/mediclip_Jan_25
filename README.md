This code version runs perfectly with changed Necker.py file.
Now primarily two files are changed in this branch.
Adapter and Necker.py. The training .0.9271, AUROC 

this repo has updated adapter. updated Necker and updated CoOp.py file.



Especially CoOp.py


## Step-by-Step Plan to Improve AUROC

1. **Add Attention Mechanism**
2. **Enhance Condition Integration**
3. **Add Temperature Scaling**
4. **Implement Prompt Ensembling**
5. **Add Contrastive Loss Component**

```python



Key Improvements:
1. Added MultiHeadAttention for better context modeling
2. Enhanced condition embedding with deeper network
3. Implemented learnable temperature scaling
4. Added prompt position ensembling
5. Improved normalization and attention flow

To use these improvements, update the training loop to include all three positions (end, middle, front) and adjust the learning rate for the new parameters.

