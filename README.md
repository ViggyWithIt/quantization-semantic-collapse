# Semantic Collapse under Structural Constraint: The Mechanics of Knowledge Erasure in 4-Bit LLMs

**Date-** December 2025  
**Author:** Julian Vignes, 
**Device:** Apple M2 Silicon (MLX Framework)

## Abstract
Parameter-efficient quantization is standard for Edge AI but its mechanistic impact on reasoning is poorly understood. This research demonstrates a phenomenon I term **"Semantic Collapse under Structural Constraint."** 

Using representation engineering on Llama-3-8B, I show that 4-bit quantization creates a competitive interference between **Instruction Following** (structure) and **Knowledge Retrieval** (factuality). While 4-bit models maintain high cosine similarity to structural goal vectors (e.g. "Speak in JSON"), they suffer a catastrophic drop in factual accuracy compared to 8-bit baselines.

## The Discovery
When forced to output a long-horizon history of the Roman Empire in strictly valid JSON:

1.  8-Bit Model (control) maintains valid JSON *and* correctly identifies historical dates (e.g. Augustus dies in 14 AD)
2.  4-Bit Model (test) maintains valid JSON but **hallucinates dates** (e.g. Augustus dies in 1 AD)

This suggests that low-bit quantization doesn't degrade capabilities uniformly. Instead, it creates a magnitude-priority effect where sparse, high-magnitude features (aka syntax) are preserved, while diffuse, low-magnitude features (knowledge) are erased by quantization noise.

## Results

### Visualizing the drift
The plot below tracks the **Cosine Similarity** between the model's residual stream and the JSON-Constraint vector over 600 tokens.

<img width="1000" height="600" alt="steering_final_0 0" src="https://github.com/user-attachments/assets/e83f8549-9a12-44b1-8dcb-f88db3a9c628" />


*Figure 1-   the 4-bit model maintains a stable structural alignment (flat cosine similarity), proving it "knows" it has to speak JSON. However, the content generated during this stable period is factually incorrect.*

### Comparison Table
| Feature | 4-Bit Quantized | 8-Bit Control |
| :--- | :--- | :--- |
| **Constraint** | "strict JSON only" | "strict JSON only" |
| **Structural Integrity** | ✅ Success (valid JSON) | ✅ Success (valid JSON) |
| **Factual Accuracy** | ❌ **Failure** (hallucinated dates) | ✅ **Success** (correct dates) |
| **Mechanism** | Semantic Collapse | Intact Retrieval |

## Reproduction

### Environment
Experiments were conducted on a MacBook Pro (M2) using the Apple MLX framework.

### Installation
```bash
pip install mlx mlx-lm numpy matplotlib
```

### Running the Probe
To replicate the 4-bit failure mode....
```bash
python src/probe_experiment.py
```
*(Note- make sure `STEERING_COEFF` is set to 0.0 in the script to observe the baseline collapse.)*

## Methodology
1.  **Vector Extraction-** I extracted a "Goal Vector" by subtracting the hidden states of conflicting prompts ("Speak JSON" - "Speak Text") at Layer 16
2.  **Intervention -** I injected a custom probe into the inference loop to track the cosine similarity of the residual stream against this vector in real-time
3.  **Stress test-** the model was forced to generate 600+ tokens of dense factual content under rigid formatting constraints to induce the collapse
