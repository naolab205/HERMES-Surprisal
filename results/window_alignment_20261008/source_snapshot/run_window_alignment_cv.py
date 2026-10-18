#!/usr/bin/env python
"""Matched window-length diagnostic; preserves archived cohorts and labels."""
import argparse
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_charbag_cv as cv

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument('--eeg-post-ms',type=int,choices=[800,1200],required=True)
parser.add_argument('--eye-post-ms',type=int,choices=[800,1200],required=True)
parser.add_argument('--eye-cache',type=Path,required=True)
window,remaining=parser.parse_known_args()
original_load=cv.load_char_bags
full_shards=None

def load_cropped(*args,**kwargs):
    global full_shards
    bags=original_load(*args,**kwargs)
    full_shards=cv.ShardedEpochs(bags.epochs)
    n=round(.1*128)+round(window.eeg_post_ms/1000*128)
    assert all(x.shape[-1]==167 for x in bags.epochs)
    bags.epochs=[x[...,:n] for x in bags.epochs]
    print(f'[window] EEG -100..+{window.eeg_post_ms}ms: {n} samples',flush=True)
    return bags

original_spatial=cv.build_spatial

def matched_spatial(bags,frame,shards,**kwargs):
    # Fit identical training-only operators/scales on the full archived EEG
    # in every arm. This isolates the model's temporal observation width.
    return original_spatial(bags,frame,full_shards,**kwargs)

cv.load_char_bags=load_cropped
cv.build_spatial=matched_spatial
from foundation import data
original_eye=data.load_eye_char

def matched_eye(bags,**kwargs):
    eye,coverage=original_eye(bags,cache_dir=window.eye_cache,**kwargs)
    assert eye.shape[-1]==round((100+window.eye_post_ms)*50/1000)
    return eye,coverage

data.load_eye_char=matched_eye
sys.argv=[sys.argv[0]]+remaining
cv.main()
