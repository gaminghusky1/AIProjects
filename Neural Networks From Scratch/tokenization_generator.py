import sentencepiece as spm

SPECIAL_TOKENS = [
    ">newconversation<",
    ">persona<",
    ">personb<",
    ">endoftext<",
]

def train_sentencepiece(
    corpus_path="TransformerData/conversations.txt",
    model_prefix="TransformerData/spm_dialogue",
    vocab_size=3000,
    model_type="bpe",
    character_coverage=1.0,
):
    """
    Trains a SentencePiece tokenizer that treats your dialogue markers as
    single, indivisible tokens.

    Writes:
      - {model_prefix}.model
      - {model_prefix}.vocab
    """
    spm.SentencePieceTrainer.Train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,

        # Make your markers atomic tokens
        user_defined_symbols=SPECIAL_TOKENS,

        # Reserve standard special ids (optional but common)
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,

        # Helpful normalization defaults
        normalization_rule_name="nmt_nfkc_cf",
        remove_extra_whitespaces=True,
        input_sentence_size=2000000,   # ok to keep even if corpus is small
        shuffle_input_sentence=True,
    )

train_sentencepiece()