import sentencepiece as spm

corpus = './_data/korean.txt,./_data/english.txt'
vocab_size = 8000
model_prefix = "sp-bpt-anderson"

spm.SentencePieceTrainer.train(
    input=corpus,
    model_prefix=model_prefix,
    vocab_size=vocab_size,
    model_type="bpe",
    max_sentence_length=500,
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    pad_piece="<pad>",
    unk_piece="<unk>",
    bos_piece="<s>",
    eos_piece="</s>",
    user_defined_symbols="<sep>,<cls>,<mask>",
    byte_fallback=False,
    num_threads=16
)
