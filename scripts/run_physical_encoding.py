#!/usr/bin/env python3
"""Replicate the locked forward diagnostic after physical-source correction."""
from pathlib import Path
import diagnose_surprisal_response as diagnostic
ROOT=Path(__file__).resolve().parents[1]
diagnostic.OUT=ROOT/'results/single_model_breakthrough_20261006/physical_encoding'
diagnostic.OUT.mkdir(parents=True,exist_ok=True)
(diagnostic.OUT/'sentence_manifest.csv').write_bytes((ROOT/'results/response_mechanism_20261002/sentence_manifest.csv').read_bytes())
if __name__=='__main__':diagnostic.main()
