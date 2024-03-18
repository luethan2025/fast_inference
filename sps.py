#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
from transformers import T5Tokenizer, T5ForConditionalGeneration

def get_tokenizer(name):
  if name == "small":
    return T5Tokenizer.from_pretrained(
                        "google-t5/t5-small",
                         model_max_length=512,
                         truncation=True,
                         return_tensors="pt")
  elif name == "base":
    return T5Tokenizer.from_pretrained(
                        "google-t5/t5-base",
                        model_max_length=512,
                        truncation=True,
                        return_tensors="pt")
  else:
    raise NotImplementedError

def get_model(name):
  if name == "small":
    return T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")
  elif name == "base":
    return T5ForConditionalGeneration.from_pretrained("google-t5/t5-base")
  else:
    raise NotImplementedError
  
def main(target, draft, prompt, target_sequence_length):
  tokenizer, model = get_tokenizer(target), get_model(target)
  print(target, draft, prompt, target_sequence_length)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(prog="sps.py",
                      description="speculative sampling")
  parser.add_argument(
           '--target',
           required=True,
           help="auto-regressive target model")
  parser.add_argument(
           '--draft',
           required=True,
           help="auto-regressive draft model")
  parser.add_argument(
           '--prompt',
           required=True,
           help="initial prompt sequence")
  parser.add_argument(
           '--target_sequence_length',
           required=True,
           type=int,
           help="target sequence length")
  
  args = parser.parse_args()

  target = args.target
  draft = args.draft
  prompt = args.prompt
  target_sequence_length = args.target_sequence_length

  main(target, draft, prompt, target_sequence_length)
