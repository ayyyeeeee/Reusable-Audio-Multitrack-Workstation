# Demucs six-source checkpoint

- Model: `htdemucs_6s`, signature `5c90dfd2`.
- Official source: https://github.com/facebookresearch/demucs
- Checkpoint: https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/5c90dfd2-34c22ccb.th
- Published mapping: https://github.com/facebookresearch/demucs/blob/main/demucs/remote/files.txt
- SHA-256: `34c22ccb381c6f9fdbf324f04e1e2fe21aaaf293f5ded163a162697ff9a02ddd`
- Download: 52.4 MiB. MIT license, preserved in LICENSE.
- Sources: drums, bass, other, vocals, guitar, piano. All names are candidate labels.
- Official README explicitly calls the six-source model experimental and warns that
  the piano source exhibits substantial bleeding and artifacts.
- Baby sounds, metronomes, and general noise are not independently trained output
  classes. They may appear in vocals, drums, other, and other predicted sources.
- The integration preserves original length/rate/channel count and shared gain.
  It does not claim that model output alone is a lossless partition of the mixture.
- Model is loaded with `weights_only=True`, only HTDemucs and Fraction explicitly
  allowlisted, and after verification of the complete checkpoint hash.

SAM Audio was also reviewed: https://github.com/facebookresearch/sam-audio .
Its text/time-span prompt API is relevant to unusual sound classes, but its official
checkpoints require approved Hugging Face access. It is not installed or presented
as a working feature of this CPU system.
