import outlines
import json
import vllm
from vllm import SamplingParams # <--- Import this to control length

# 1. SETUP: Load Vocabulary
with open("hsk1.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    vocab_list = data["vocab"]

# 2. SETUP: Load Model
llm = vllm.LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    gpu_memory_utilization=0.8,
    max_model_len=8192,
    dtype="bfloat16"
)
model = outlines.models.VLLMOffline(llm)

# 3. THE REGEX
# We add implicit whitespace support just in case, though Chinese barely uses it.
vocab_regex = f"({'|'.join(vocab_list)})+"

# 4. THE GENERATOR
generator = outlines.Generator(model, outlines.regex(vocab_regex))

# 5. THE SAMPLER (This fixes the cut-off!)
# We tell vLLM explicitly: "Generate exactly 100 tokens."
sampling_params = SamplingParams(
    max_tokens=100, 
    temperature=0.7, # Creativity
    stop=None        # We handle stopping manually
)

# 6. HELPER: The "Sentence Trimmer"
def clean_output(text):
    # Find the last Chinese period "。" and cut everything after it.
    last_period = text.rfind("。")
    if last_period != -1:
        return text[:last_period+1]
    return text

# 7. RUN IT
user_prompt = "Write a short story about a person who loves their cat."
prompt = f"""<|im_start|>system
You are a helpful Chinese teacher. You must write a simple story using ONLY the allowed characters.
Do not output English. Do not list words. Write coherent sentences.<|im_end|>
<|im_start|>user
{user_prompt}<|im_end|>
<|im_start|>assistant
"""

print(f"\nPrompting for: {user_prompt}\n")

for i in range(3):
    print(f"--- Generating Story {i+1} ---")
    
    # PASS THE SAMPLING PARAMS HERE
    response = generator(prompt, sampling_params=sampling_params)
    
    # Clean the result
    final_story = clean_output(str(response))
    
    print(final_story)
    print("\n")