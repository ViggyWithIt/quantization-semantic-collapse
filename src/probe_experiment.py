import mlx.core as mx
from mlx_lm import load, generate
import numpy as np
import matplotlib.pyplot as plt
import gc

# experiment config
#We use the 4-bit model to demonstrate the "semantic collapse" phenomenon
# to run the control group (which works correctly), change this to the 8-bit version.
MODEL_ID = "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"

# Layer 16 is the middle of the network where high-level reasoning typically occurs
LAYER_IDX = 16 
MAX_TOKENS = 600

# STEERING COEFFICIENT
# 0.0 = Observational Mode (See the failure happen naturally)
# 1.5 = Interventional Mode (attempt to fix the drift via vector injection)
STEERING_COEFF = 0.0 

print(f"Loading Model: {MODEL_ID}...")
model, tokenizer = load(MODEL_ID)


# STEP 0: MODEL HEALTH CHECK
# before modifying the network, we make sure the base model is loaded correctly
print("\n--- SANITY CHECK (standard generation) ---")
test_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nSay 'Hello World' and stop.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
_ = generate(model, tokenizer, prompt=test_prompt, max_tokens=10, verbose=True)
print("------------------------------------------\n")


# STEP 1: INSTRUMENTATION (the probe)
# We define a wrapper class that sits inside the model
# It captures the "hidden states" (the model's thoughts) as they pass through

captured_residuals = []

class InstrumentatedLayer:
    def __init__(self, original_layer):
        self.base_layer = original_layer
        self.steering_vector = None

    # We use *args and **kwargs to ensure we don't break position embeddings (RoPE)
    def __call__(self, *args, **kwargs):
        # 1. run the original layer logic
        output = self.base_layer(*args, **kwargs)
        
        #2. Handle return format (output can be a tuple or a single tensor)
        if isinstance(output, tuple):
            h = output[0]  # Hidden state is usually the first element
            rest = output[1:]
        else:
            h = output
            rest = ()
        
        # 3. CAPTURE STATE- record the model's activation at the last token
        try:
            last_token_state = h[:, -1, :]
            mx.eval(last_token_state) #Force evaluation to save to memory
            captured_residuals.append(np.array(last_token_state))
        except Exception:
            pass # Ignore shape mismatches during warm-up

        # 4. OPTIONAL STEERING: inject vector if coefficient > 0
        if self.steering_vector is not None and STEERING_COEFF != 0.0:
            h = h + (self.steering_vector * STEERING_COEFF)

        #5. return data to the next layer
        if isinstance(output, tuple):
            return (h,) + rest
        return h

    # Pass any other attribute access to the original layer
    def __getattr__(self, name):
        return getattr(self.base_layer, name)

print(f"Injecting analysis probe at Layer {LAYER_IDX}...")

# Locate/replace the target layer with our instrumented version
if hasattr(model, 'model'):
    model.model.layers[LAYER_IDX] = InstrumentatedLayer(model.model.layers[LAYER_IDX])
else:
    model.layers[LAYER_IDX] = InstrumentatedLayer(model.layers[LAYER_IDX])

# Keep a reference to control the probe later
probe_instance = model.model.layers[LAYER_IDX]


# STEP 2: DEFINE THE GOAL DIRECTION
# We use contrastive extraction to find the direction in latent space 
# that represents "Speaking in JSON"

print("Step 1: extracting latent Goal Vector...")

def extract_state(text):
    """runs a forward pass and returns the hidden state at the probe layer"""
    captured_residuals.clear()
    ids = tokenizer.encode(text)
    model(mx.array([ids])) # Run model (probe will capture state)
    if not captured_residuals: return None
    return captured_residuals[-1]

# Prompt A- enforces the constraint
prompt_positive = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou MUST answer strictly in JSON format.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nMars.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
# Prompt B - forbids the constraint
prompt_negative = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou MUST answer in plain text.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nMars.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

vec_pos = extract_state(prompt_positive)
vec_neg = extract_state(prompt_negative)

if vec_pos is None or vec_neg is None:
    raise ValueError("Error: probe didn't capture any data.")

# The Goal Vector is the difference direction
goal_vector = vec_pos - vec_neg
goal_vector = goal_vector / (np.linalg.norm(goal_vector) + 1e-9) # Normalize

#load the vector into the probe (for optional steering)
probe_instance.steering_vector = mx.array(goal_vector)
print("Goal Vector extracted successfully.")


# STEP 3: STRESS TEST (LONG GENERATION)
# We force the model to generate a long sequence to test stability over time..

print(f"Step 2: running generation (Steering Coeff={STEERING_COEFF})...")
print("-" *40)

long_prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou MUST answer strictly in JSON format.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\nWrite a history of the Roman Empire (year 1 to 500). Each entry must be a JSON object: {'year': 1, 'event': '...'}.<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

captured_residuals.clear()

# we use the native generate function to ensure correct caching and positional embeddings
generate(model, tokenizer, prompt=long_prompt, max_tokens=MAX_TOKENS, verbose=True)

print("\n"+"-"*40)


#STEP 4: VISUALIZE RESULTS
#We plot the alignment (Cosine Similarity) over time

print("Step 3: Plotting alignment trends...")
# filter to analyze only the generated tokens (exclude the prompt)
residuals_to_analyze = captured_residuals[-(MAX_TOKENS):]

similarities = []
goal_flat = goal_vector.flatten().astype(np.float32)

for h in residuals_to_analyze:
    h_flat = h.flatten().astype(np.float32)
    # calculate Cosine Similarity
    sim = np.dot(h_flat, goal_flat) / (np.linalg.norm(h_flat) * 1.0)
    similarities.append(sim)

# plotting
plt.figure(figsize=(10, 6))
plt.plot(similarities, color='blue', alpha=0.3, label='Cosine Similarity (raw)')

# add a moving avg trend line for clarity
window = 50
if len(similarities) > window:
    smoothed = np.convolve(similarities, np.ones(window)/window, mode='valid')
    plt.plot(range(len(smoothed)), smoothed, color='black', linewidth=2, label='Trend (Moving Avg)')

plt.title(f"Structural Goal Stability (4-Bit Model)")
plt.xlabel("Generated Tokens")
plt.ylabel("Alignment with 'JSON Constraint' Vector")
plt.axhline(0, color='gray', linestyle='--')
plt.legend()
plt.tight_layout()
plt.savefig(f"drift_plot_{STEERING_COEFF}.png")
print(f"Analysis complete Plot saved to drift_plot_{STEERING_COEFF}.png")