Registered before execution. Reuse the prior fixed metadata-only training
sentence subset (caps1200/600; seed2026). All available readers, all physical
slots, same CSD, four fixed latency windows. Compare zero-sum signed latency
contrasts vs repeated window-mean features and per-observation time-scramble
contrast control. Cross-fit all five training text blocks per development
fold0/3. Fit train book/count target residualization, StandardScaler, Ridge
alpha1000 only on fitting blocks. No validation or test signals/labels used.
Gate: mean conditional rho >.05 and above both controls in both settings.
This gates a linear signed-latency route only; no general impossibility claim.
