#!/bin/bash

# A script to train a SentencePiece tokenizer using the spm_train command-line tool.
# It sets up special tokens for BERT-style models, ensuring a consistent vocabulary structure.

set -e # Exit immediately if a command exits with a non-zero status.
set -o pipefail # Fail a pipe if any sub-command fails.

# --- Default Arguments ---
INPUT_FILE=""
MODEL_PATH="tokenizer"
MODEL_PREFIX="spm_bpe"
VOCAB_SIZE=32000
MODEL_TYPE="bpe"

# --- Help Message ---
function usage() {
  cat <<EOF
Usage: $0 -i <input_file> [-p <model_path>] [-m <model_prefix>] [-v <vocab_size>] [-t <model_type>]

Trains a SentencePiece tokenizer for BERT-style models.

Options:
  -i <input_file>     Path to the raw text file for training (required).
  -p <model_path>     Directory to save the model (default: "tokenizer").
  -m <model_prefix>   Prefix for model and vocab files (default: "spm_bpe").
  -v <vocab_size>     Vocabulary size (default: 32000).
  -t <model_type>     Model type: bpe, unigram, char, word (default: "bpe").
  -h                  Display this help message.
EOF
  exit 1
}

# --- Parse Command-line Arguments ---
while getopts "i:p:m:v:t:h" opt; do
  case ${opt} in
    i) INPUT_FILE="$OPTARG" ;;
    p) MODEL_PATH="$OPTARG" ;;
    m) MODEL_PREFIX="$OPTARG" ;;
    v) VOCAB_SIZE="$OPTARG" ;;
    t) MODEL_TYPE="$OPTARG" ;;
    h) usage ;;
    \?) usage ;;
  esac
done

# --- Validate Required Arguments ---
if [ -z "${INPUT_FILE}" ]; then
  echo "Error: Input file is required." >&2
  usage
fi

if [ ! -f "${INPUT_FILE}" ]; then
  echo "Error: Input file not found at: ${INPUT_FILE}" >&2
  exit 1
fi

# --- Main Logic ---
echo "Starting tokenizer training..."

# Create model directory if it doesn't exist
mkdir -p "${MODEL_PATH}"
FULL_PREFIX="${MODEL_PATH}/${MODEL_PREFIX}"

# Special tokens for BERT/seq2seq models
# We ensure the following ID assignments:
#   0: [PAD]  - Padding token
#   1: [UNK]  - Unknown token (handled by SentencePiece)
#   2: [CLS]  - Encoder start token / Decoder BOS
#   3: [SEP]  - Encoder segment separator / Decoder EOS
#   4: [MASK] - MLM masking token
#
# We disable SentencePiece's built-in BOS/EOS (which would be <s>/</s>)
# and instead define [CLS], [SEP], [MASK] explicitly as user symbols.
PAD_ID=0
UNK_ID=1
BOS_ID=-1  # Disable built-in BOS
EOS_ID=-1  # Disable built-in EOS
USER_SYMBOLS="[CLS],[SEP],[MASK]"

# Check if spm_train is available
if ! command -v spm_train &> /dev/null; then
    echo "Error: 'spm_train' command not found." >&2
    echo "Please install SentencePiece or ensure it is in your PATH." >&2
    exit 1
fi

# Construct and run the spm_train command
spm_train \
  --input="${INPUT_FILE}" \
  --model_prefix="${FULL_PREFIX}" \
  --vocab_size="${VOCAB_SIZE}" \
  --model_type="${MODEL_TYPE}" \
  --pad_id="${PAD_ID}" \
  --unk_id="${UNK_ID}" \
  --bos_id="${BOS_ID}" \
  --eos_id="${EOS_ID}" \
  --user_defined_symbols="${USER_SYMBOLS}" \
  --hard_vocab_limit=false

echo "Tokenizer training complete."
echo "Model saved to ${FULL_PREFIX}.model"
echo "Vocabulary saved to ${FULL_PREFIX}.vocab"

# You can verify the created vocabulary and special tokens by inspecting
# the generated '${FULL_PREFIX}.vocab' file.
