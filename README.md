# Joermungandr

## Special Tokens

The tokenizer is trained with SentencePiece. Special token IDs are fixed:

| ID  | Token   | Usage                                   |
|-----|---------|---------------------------------------- |
| 0   | [PAD]   | Padding                                 |
| 1   | [UNK]   | Unknown token                           |
| 2   | [CLS]   | Encoder start token / Decoder BOS       |
| 3   | [SEP]   | Encoder segment separator / Decoder EOS |
| 4   | [MASK]  | MLM masking                             |
| 5+  | content | Regular vocabulary                      |

Train the tokenizer with `scripts/train_tokenizer.sh`.

## Data Format

```
[CLS] Sent_1 [SEP] Sent_2 [SEP] Sent_3 [SEP]
```

## Attention

Full bidirectional attention.

## The "Next Sentence" Objective

For triples, the "Next Sentence Prediction" (NSP) becomes a "Valid Conversation Continuation" check.

**Positive sample (label 1):**
```
[CLS] S_t [SEP] S_{t+1} [SEP] S_{t+2} [SEP]
```

**Negative sample (label 0):**
```
[CLS] S_t [SEP] S_{t+1} [SEP] S_rand [SEP]
```

## Changes from BERT

- **RoPE**: No absolute position embeddings needed
- **Pre-Norm + RMSNorm**: More stable training
- **No biases**: Saves parameters
- **GQA**: Reduced KV cache overhead (TBD if useful for encoder-only)
- **SwiGLU**: Better and cheaper than GELU FFN
