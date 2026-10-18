Registered before training: independent shared-row-reference candidate.
Evidence: pre-window [-100,0]ms overlaps the prior character's response at
SOA~413ms; distinct local references destroy equality of samples representing
the same continuous EEG interval. This hypothesized impact is not a proven
cause of the decoding failure. Source bandpass already .5–30Hz.
Subtract mask-weighted mean of all real slots' 13-sample pre-windows once per
row, re-zero padded slots, retain default all-slot BatchNorm and all model
parameters. Exactly the same row-average wave as per-character baselining;
only inter-character offsets change. No timing scalar/eye/text/labels supplied.
Control = standard local baseline run norm_control, same raw input, CSD fixed
training-only scaling convention, seed2026, modern/no-deconv, folds0/3,
batch64, max25epochs/patience6, trial-rho selection. No combination with valid
normalization in this screen. Both folds item/conditional gains >0 and trial
no decline are required to expand. After two failed normalization/centering
interventions, pivot rather than tune alpha, extra epochs, or new seeds.
Closest prior basis (not novelty): Alday 2019 doi10.1111/psyp.13451 and
Ehinger/Dimigen2019 https://pmc.ncbi.nlm.nih.gov/articles/PMC6815663/.
This is a reference-frame intervention, not a full deconvolution solution.
