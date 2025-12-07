import outlines
import json
import os
import vllm

# 1. SETUP: Load the JSON Brain
print("Loading HSK1 Vocabulary...")
with open("hsk1.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    vocab_list = data["vocab"]

# 2. SETUP: Load Model (Optimized for 4090)
print("Loading Model...")
llm = vllm.LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_memory_utilization=0.8,
    max_model_len=8192,
    dtype="bfloat16" # Native 4090 precision
)
model = outlines.models.VLLMOffline(llm)

# We build a Regex that says: (word1|word2|word3) repeated one or more times
# We essentially create a "Finite State Machine" for the LLM.
vocab_regex = f"({'|'.join(vocab_list)})+"

# 3. THE GENERATOR: Compile the constraint
# This tells the 4090: "If a token isn't in this regex, set its probability to 0"
generator = outlines.Generator(model, outlines.regex(vocab_regex))

user_prompt = "Write a short story about a person who loves their cat."

prompt = f"""<|im_start|>system
You are a helpful Chinese teacher. You must write a simple story using ONLY the allowed characters.
Do not output English. Do not list words. Write coherent sentences.<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""

print(f"\nPrompting for: {user_prompt}\n")

# 6. RUN IT
# We generate 3 variations.
for i in range(3):
    print(f"--- Generating Story {i+1} ---")
    response = generator(prompt)
    print(response)
    print("\n")