import pandas as pd
import numpy as np

from huggingface_hub import login

hf_token = 'your hugging face token' ###apply in hugging face
login(hf_token)

import torch
from llm2vec import LLM2Vec

l2v = LLM2Vec.from_pretrained(
    "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp",
    peft_model_name_or_path="McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-unsup-simcse",
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)

text_all=pd.read_csv('/home/rliuaj/balance_sheet/text_all.csv')

text_list=[i for i in text_all['text']]


import pandas as pd
import torch

# Create an empty DataFrame
df_reps = pd.DataFrame()

# Assuming l2v.encode() is the encoding function
# Example of how this might work
for i in range(0, len(text_list), 500):
    try:
        # Get the current chunk of 500 texts
        current_texts = text_list[i:i+500]

        # Encode and normalize
        d_reps_1 = l2v.encode(current_texts)  # This should give you embeddings for the current batch
        d_reps_norm_1 = torch.nn.functional.normalize(d_reps_1, p=2, dim=1)  # Normalize the embeddings
        
        # Convert the tensor to a DataFrame (assuming each row is an embedding vector)
        # If d_reps_norm_1 is a 2D tensor, each row can be a vector representing a text
        df_chunk = pd.DataFrame(d_reps_norm_1.detach().numpy())  # Convert to numpy array for pandas

        # Append to the main DataFrame
        df_reps = pd.concat([df_reps, df_chunk], ignore_index=True)

        # Save the DataFrame after processing each chunk
        df_reps.to_csv("text_embeddings.csv", index=False)

        print(f"Processed and saved {i + 500 if i + 500 <= len(text_list) else len(text_list)} texts.")
    
    except Exception as e:
        print(f"Error processing batch {i//500 + 1}: {e}")
        continue

