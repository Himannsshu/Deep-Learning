import torch
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import WordLevelTrainer
from torch.utils.data import DataLoader, Dataset, random_split
from pathlib import Path
from typing import Any
import warnings

def build_tokenizer(config, ds, lang):
  tokenizer_path = Path(config['tokenizer_file'].format(lang))

  if not tokenizer_path.exists():
    tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = WordLevelTrainer(special_tokens=['[UNK]', '[PAD]', '[SOS]', '[EOS]'], min_frequency=2)
    tokenizer.train_from_iterator(get_all_sentence(ds, lang), trainer=trainer)
    tokenizer.save(str(tokenizer_path))
  else:
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
  return tokenizer

def get_all_sentence(ds, lang):
  for pair in ds:
    yield pair['translation'][lang]

def casual_mask(size):
  mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
  return mask == 0

class BilingualDataset(Dataset):

  def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) -> None:
    super().__init__()

    self.seq_len = seq_len
    self.ds = ds
    self.tokenizer_src = tokenizer_src
    self.tokenizer_tgt = tokenizer_tgt
    self.src_lang = src_lang
    self.tgt_lang = tgt_lang
    self.sos_token = torch.tensor([tokenizer_tgt.token_to_id('[SOS]')], dtype=torch.int64)
    self.eos_token = torch.tensor([tokenizer_tgt.token_to_id('[EOS]')], dtype=torch.int64)
    self.pad_token = torch.tensor([tokenizer_tgt.token_to_id('[PAD]')], dtype=torch.int64)

  def __len__(self):
    return len(self.ds)

  def __getitem__(self, index: Any) -> Any:
    src_target_pair = self.ds[index]
    src_text = src_target_pair['translation'][self.src_lang]
    tgt_text = src_target_pair['translation'][self.tgt_lang]
    enc_input_tokens = self.tokenizer_src.encode(src_text).ids
    dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
    enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
    dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
    if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
      raise ValueError('Sentence is too long')
    encoder_input = torch.cat([
        self.sos_token,
        torch.tensor(enc_input_tokens, dtype=torch.int64),
        self.eos_token,
        torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64),])

    decoder_input = torch.cat([
        self.sos_token,
        torch.tensor(dec_input_tokens, dtype=torch.int64),
        torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),])

    label = torch.cat([
        torch.tensor(dec_input_tokens, dtype=torch.int64),
        self.eos_token,
        torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64),])

    assert encoder_input.size(0) == self.seq_len
    assert decoder_input.size(0) == self.seq_len
    assert label.size(0) == self.seq_len

    return {
        'encoder_input': encoder_input,
        'decoder_input': decoder_input,
        'encoder_mask': (encoder_input != self.pad_token)
        .unsqueeze(0)
        .unsqueeze(0)
        .int(),
        'decoder_mask': (decoder_input != self.pad_token)
        .unsqueeze(0)
        .unsqueeze(0)
        .int()
        & casual_mask(decoder_input.size(0)),
        'label': label,
        'src_text': src_text,
        'tgt_text': tgt_text,
    }

def get_ds(config):
  ds_raw = load_dataset('Helsinki-NLP/opus_books',f"{config['lang_src']}-{config['lang_tgt']}",split='train',)

  tokenizer_src = build_tokenizer(config, ds_raw, config['lang_src'])
  tokenizer_tgt = build_tokenizer(config, ds_raw, config['lang_tgt'])

  train_ds_size = int(0.9 * len(ds_raw))
  val_ds_size = len(ds_raw) - train_ds_size
  train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

  train_ds = BilingualDataset(train_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'],)
  val_ds = BilingualDataset(val_ds_raw,tokenizer_src,tokenizer_tgt,config['lang_src'],config['lang_tgt'],config['seq_len'],)

  max_len_src = 0
  max_len_tgt = 0
  for pair in ds_raw:
    src_ids = tokenizer_src.encode(pair['translation'][config['lang_src']]).ids
    tgt_ids = tokenizer_tgt.encode(pair['translation'][config['lang_tgt']]).ids
    max_len_src = max(max_len_src, len(src_ids))
    max_len_tgt = max(max_len_tgt, len(tgt_ids))

  print(f'Max length of source sentence: {max_len_src}')
  print(f'Max length of target sentence: {max_len_tgt}')

  train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
  val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True)
  return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_config():
  return {
      'batch_size': 8,
      'num_epoches': 20,
      'lr': 10**-4,
      'seq_len': 352,
      'd_model': 512,
      'lang_src': 'en',
      'lang_tgt': 'it',
      'model_folder': 'weights',
      'model_basename': 'tmodel',
      'preload': None,
      'tokenizer_file': 'tokenizer_{0}.json',
      'experiment_name': 'runs/tmodel',
  }

def get_weights_file_path(config, epoch: str):
  model_folder = config['model_folder']
  model_basename = config['model_basename']
  model_filename = f'{model_basename}{epoch}.pt'
  return str(Path('.') / model_folder / model_filename)