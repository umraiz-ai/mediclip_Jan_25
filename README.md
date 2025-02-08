This code version runs perfectly with changed Necker.py file.
Now primarily two files are changed in this branch.
Adapter and Necker.py. The training .0.9267, AUROC 

this repo has updated adapter. updated Necker and updated CoOp.py file.



Especially CoOp.py

## Plan for Visual Prompt Tuning (VPT)

1. Create VPTLayer
2. Create VisualPromptTuner
3. Integrate with existing PromptMaker
4. Maintain compatibility with current pipeline

```python



Key Features:
1. VPTLayer: Handles prompt injection at each transformer layer
2. VisualPromptTuner: Manages visual prompt tuning across all layers
3. Updated PromptMaker: Integrates VPT with existing text prompts
4. Maintains compatibility with existing pipeline
5. Added option to enable/disable VPT



