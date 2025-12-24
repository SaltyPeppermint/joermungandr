# Joermungandr

## Special Tokens

The tokenizer is trained with SentencePiece. Special token IDs are fixed:

| ID  | Token   | Usage                                   |
|-----|---------|---------------------------------------- |
| 0   | [PAD]   | Padding                                 |
| 1   | [UNK]   | Unknown token                           |
| 2   | [BOS]   | Used as Encoder [CLS] / Decoder [BOS]   |
| 3   | [EOS]   | Used as Encoder [SEP] / Decoder [EOS]   |
| 4   | [MASK]  | MLM masking                             |
| 5+  | content | Regular vocabulary                      |

Train the tokenizer with `scripts/train_tokenizer.sh`.

## Encoder

### Data Format

```
[CLS] Sent_1 [SEP] Sent_2 [SEP] Sent_3 [SEP]
```

### Attention

Full bidirectional attention.

### Next Sentence Objective

For triples, the "Next Sentence Prediction" (NSP) becomes a "Valid Conversation Continuation" check.

**Positive sample (label 1):**
```
[CLS] S_t [SEP] S_{t+1} [SEP] S_{t+2} [SEP]
```

**Negative sample (label 0):**
```
[CLS] S_t [SEP] S_rand [SEP] S_{t+2} [SEP]
```

## Seq2Seq

### Data Format

```
[BOS] Sent_1 [SEP] Sent_3 [SEP]
[BOS] Sent_2
```

### Attention

Full bidirectional attention for Encoder, Causal Masking for Decoder

## Changes from BERT

- **RoPE**: No absolute position embeddings needed
- **Pre-Norm + RMSNorm**: More stable training
- **No biases**: Saves parameters
- **GQA**: Reduced KV cache overhead
- **SwiGLU**: Better and cheaper than GELU FFN
