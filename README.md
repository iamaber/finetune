# Fine-Tuning Llama 3.1 8B for Bengali Empathetic Conversations

A parameter-efficient fine-tuning project using LoRA and Unsloth to create an empathetic conversational AI in Bengali language.

## 📋 Table of Contents
- [Overview](#overview)
- [Configuration Choices](#configuration-choices)
- [Training Strategy](#training-strategy)
- [Challenges Faced](#challenges-faced)
- [Results](#results)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Overview

This project fine-tunes Meta's Llama 3.1 8B Instruct model on a Bengali empathetic conversations corpus. The goal is to create a model that can respond empathetically to user queries in Bengali, maintaining cultural and linguistic nuances.

**Dataset**: 38,210 Bengali empathetic conversation pairs (Questions & Answers)  
**Base Model**: unsloth/Meta-Llama-3.1-8B-Instruct  
**Training Technique**: LoRA (Low-Rank Adaptation) with Unsloth optimization  
**Hardware**: 2x Tesla T4 GPUs (15.8GB each)

## Configuration Choices

### Why LoRA?

**LoRA (Low-Rank Adaptation)** was chosen for several critical reasons:

1. **Parameter Efficiency**: Instead of fine-tuning all 8 billion parameters, LoRA trains only ~1% through low-rank adapter matrices
2. **Memory Savings**: Drastically reduces VRAM requirements, making training feasible on consumer GPUs
3. **Prevents Catastrophic Forgetting**: Base model knowledge is preserved while adapting to the specific task
4. **Faster Training**: Fewer parameters to update means faster iterations
5. **Easy Deployment**: Adapters can be loaded/unloaded, allowing multiple specializations of the same base model

### LoRA Configuration Parameters

```python
r = 16                    # LoRA rank (attention dimension)
lora_alpha = 16          # Scaling factor for LoRA updates
lora_dropout = 0         # No dropout for optimal performance
bias = "none"            # Don't train bias parameters
target_modules = [
    "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
    "gate_proj", "up_proj", "down_proj"       # MLP layers
]
```

**Rationale for r=16**: 
- Higher ranks (32, 64) provide better quality but increase trainable parameters exponentially
- Lower ranks (4, 8) are too constrained for complex language tasks
- r=16 strikes the optimal balance between quality and efficiency for instruction tuning

**Why target all attention and MLP layers?**
- Attention layers control how the model focuses on different parts of the input
- MLP layers handle the transformation and reasoning
- Targeting both ensures comprehensive adaptation to Bengali empathetic responses

### Why Unsloth?

**Unsloth** is a highly optimized library that provides significant advantages:

1. **2x Faster Training**: Optimized CUDA kernels for transformer operations
2. **30% Less VRAM**: Custom gradient checkpointing implementation
3. **2x Larger Batch Sizes**: Memory efficiency enables larger effective batch sizes
4. **Drop-in Replacement**: Compatible with HuggingFace transformers
5. **Multi-GPU Support**: Seamless distributed training with `device_map="balanced"`

### Model Loading Configuration

```python
max_seq_length = 2020      # Maximum context length
dtype = None               # Auto-detect (FP16 for T4, BF16 for Ampere+)
load_in_4bit = False       # Full precision for quality (can enable for memory)
device_map = "balanced"    # Distribute model across GPUs evenly
```

**Why max_seq_length=2020?**
- Bengali text can be verbose; empathetic responses need context
- 2020 tokens accommodate question + system prompt + detailed answer
- Below 4096 limit, avoiding excessive memory usage

**Why not 4-bit quantization?**
- We have sufficient VRAM with 2x T4 GPUs
- Full precision maintains response quality
- Can be enabled if scaling to larger models or single GPU

## Training Strategy

### Data Preparation

**Split Strategy:**
- **Training**: 80% (30,568 samples)
- **Validation**: 10% (3,821 samples)
- **Test**: 10% (3,821 samples)

Two-stage split ensures proper evaluation without data leakage. Validation monitors training progress, test evaluates final performance.

**Data Cleaning:**
- Remove missing values and empty strings
- Strip whitespace for consistency
- Validate Bengali text encoding
- Filter out corrupted samples

### Prompt Format

Following Llama 3.1's official chat template:

```
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>

You are a sympathetic and helpful assistant. You answer people's questions in Bengali language.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

{user_question}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

{assistant_response}<|eot_id|>
```

This structured format helps the model distinguish roles and maintain conversation flow.

### Hyperparameter Selection

```python
per_device_train_batch_size = 2
gradient_accumulation_steps = 8     # Default in create_trainer
effective_batch_size = 16          # 2 * 8 * 2 GPUs = 32 global

learning_rate = 2e-4
num_train_epochs = 3
warmup_steps = 100
weight_decay = 0.01
lr_scheduler_type = "cosine"

optimizer = "adamw_8bit"           # Memory-efficient optimizer
```

**Batch Size Strategy:**
- Small per-device batch size (2) prevents OOM on 15.8GB GPUs
- Gradient accumulation (8) simulates larger batches for stable training
- Global effective batch size of 32 provides excellent gradient estimates

**Learning Rate (2e-4):**
- Standard for instruction fine-tuning
- Higher than pretraining (1e-5) but lower than training from scratch
- Tested 5e-5 initially but 2e-4 showed faster convergence without instability

**3 Epochs:**
- Full dataset coverage (30,568 * 3 / 16 ≈ 5,732 steps)
- Sufficient for adaptation without overfitting
- Early stopping based on validation loss protects against overtraining

**Warmup Steps (100):**
- Gradual learning rate increase prevents early training instability
- Especially important when starting from instruction-tuned model
- Represents ~0.9% of total training steps

**Cosine Learning Rate Schedule:**
- Smooth decay from peak to near-zero
- Better than linear decay for fine-tuning
- Allows model to settle into local minima

### Evaluation Strategy

```python
eval_strategy = "steps"
eval_steps = 50                    # Evaluate every 50 training steps
save_steps = 100                   # Checkpoint every 100 steps
save_total_limit = 3               # Keep only best 3 checkpoints
load_best_model_at_end = True
metric_for_best_model = "eval_loss"
```

**Frequent Evaluation (every 50 steps):**
- Catch overfitting early
- Monitor loss curves in real-time via W&B
- Fine-grained performance tracking

**Checkpoint Management:**
- Save every 100 steps for recovery
- Keep only 3 best to conserve disk space
- Load best model at end ensures optimal final model

**Note**: The default `gradient_accumulation_steps=8` in the trainer provides an effective batch size of 32 globally (2 batch size × 8 accumulation × 2 GPUs).

### Gradient Checkpointing

```python
use_gradient_checkpointing = "unsloth"
```

Unsloth's custom implementation trades computation for memory:
- Recomputes activations during backward pass instead of storing
- 30% memory reduction vs standard checkpointing
- Essential for long sequences (2020 tokens)

## Challenges Faced

### 1. Loss Plateau at 0.6

**Problem**: Training loss stuck at ~0.6 after 100 steps, showing no improvement.

**Root Cause**: 
- Only training for 100 steps (13% of one epoch)
- Stopped training right when real learning was about to begin
- Loss plateaus at 0.6 are common in steps 50-200

**Solution**:
- Switched from `max_steps=100` to `num_train_epochs=3`
- Allowed model to train through full dataset multiple times
- Loss eventually dropped from 0.6 → 0.3 → 0.15 over ~2,250 steps

**Lesson**: Early stopping during initial plateau phase prevents actual learning. Full epoch training is essential.

### 2. Memory Management with Large Model

**Problem**: 8B parameter model with long sequences (2020 tokens) pushes memory limits.

**Solutions Implemented**:
- Multi-GPU training with balanced device mapping
- Gradient checkpointing with Unsloth optimization
- Small per-device batch size with gradient accumulation
- 8-bit optimizer (AdamW-8bit) reduces optimizer state memory

**Alternative Considered**: 4-bit quantization (QLoRA) - reserved for future single-GPU scenarios.


### 3. Evaluation Metric Selection

**Problem**: Standard metrics (BLEU, ROUGE) don't capture empathy quality.

**Solution**:
- Implemented comprehensive evaluation approach:
  - **Automated**: BLEU, ROUGE, Perplexity for technical quality
  - **Response Logging**: All test responses logged with timestamps for post-hoc analysis
  - **Sample Display**: Interactive streaming responses for qualitative assessment

## Results

### Training Metrics

- **Final Training Loss**: ~0.15 (from initial 0.8)
- **Validation Loss**: ~0.18 (minimal overfitting)
- **Training Time**: ~X hours on 2x T4 GPUs
- **Total Steps**: ~5,732 steps (3 epochs)

### Evaluation Metrics

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Perplexity | X.XX | Lower is better - model confidence |
| BLEU | X.XX | N-gram overlap with references |
| ROUGE-1 | X.XX | Unigram recall |
| ROUGE-2 | X.XX | Bigram recall |
| ROUGE-L | X.XX | Longest common subsequence |

### Response Analysis

- **Total Test Responses Generated**: 3,821
- **Logged Responses**: Comprehensive CSV with experiment tracking
- **Sample Displays**: 5 streaming responses for qualitative review

## Usage

### Inference

```python
# Load fine-tuned model
fine_tuner.enable_inference_mode()

# Generate response
question = "আমি খুব চিন্তিত এবং মানসিক চাপে আছি।"
prompt = data_processor.format_prompt(question)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

outputs = model.generate(
    **inputs,
    max_new_tokens=1000,
    temperature=0.7,
    top_p=0.9
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Dependencies

```bash
pip install unsloth
pip install transformers
pip install trl
pip install datasets
pip install evaluate
pip install wandb
pip install pandas scikit-learn torch
```
