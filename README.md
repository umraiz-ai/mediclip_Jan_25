This code version runs perfectly with changed Necker.py file.
Now primarily two files are changed in this branch.
Adapter and Necker.py. The training .9485, AUROC 

this repo has updated adapter. updated Necker and updated CoOp.py file.
This is the implementation of the CoCoOp. 


Especially CoOp.py


CoOp.py

 code:

## Core Components and Working Principle

The code implements a conditional prompt learning system with three main classes:

1. **TextEncoder**
- Takes text and converts it into meaningful representations
- Uses CLIP's transformer to process text
- Applies positional embeddings to understand word order
- Projects text into a shared space with images

2. **PromptLearner**
- Creates learnable prompt templates for both normal and abnormal medical conditions
- Handles three prompt positions: start (front), middle, and end
- Now includes condition-aware prompts through a new condition embedding layer
- Combines fixed text with learnable components
- Processes prompts differently based on their position (front/middle/end)

3. **PromptMaker**
- Acts as the main interface combining TextEncoder and PromptLearner
- Takes image features and optional conditions as input
- Generates text features that align with image features
- Normalizes the outputs for better comparison

## Flow of Operation

1. Input text prompts are tokenized and split into prefix/suffix parts
2. Learnable components (ctx vectors) are initialized for each class and position
3. When processing:
   - Conditions modify the learnable components if provided
   - Prompts are assembled based on their position (front/middle/end)
   - Text encoder converts prompts into feature vectors
   - Features are normalized for comparison with image features

This code essentially creates a learnable bridge between medical images and text descriptions, allowing the system to adapt its understanding based on both the content and additional conditions.
