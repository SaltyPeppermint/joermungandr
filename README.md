## Data format

[CLS]	Sent 1	[SEP]	Sent 2	[SEP]	Sent 3	[SEP]

## Attention:

Full bidirectional attention

## The "Next Sentence" Objective

For triples, the "Next Sentence Prediction" (NSP) becomes a "Valid Conversation Continuation" check.
Positive Sample (Label 1):

Input: [CLS] S_t [SEP] S_{t+1} [SEP] S_{t+2} [SEP]

Negative Sample (Label 0):

Pick a random sentence S 
Input: [CLS] S_t [SEP] S_{t+1} [SEP] S_{rand} [SEP]

## Changes from default bert besides 3 sentence Objective

- RoPE: Not needing to deal with absolute embeddings.
- Pre-Norm + RMSNorm: Standard now I guess
- No Biases in Linear layers: Saves parameters 
- GQA: Reduced KV cache overhead, once I implement it. Actually I dont know if this makes sense for an encoder only model
