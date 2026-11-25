# Server-side verification of the published E16 release

Verified on 2026-11-25 17:14:10 UTC+08:00, directly in the server workspace holding the experiment archives.

- The checkout initially matched public commit `42d503f67d56004fe78b44a6b51e1831eb747e9b`.
- All 42 release-manifest entries matched their SHA256 values.
- All 36 frozen task records, including their complete GT sets, matched the original E16 archive.
- The released summary matched the original server summary exactly.
- All 144 public task/model/budget score records matched every retained field in their original result files.
- All 288 long trajectories matched original token IDs, raw strings, seeds, prompt lengths, and recorded timings.
- Decompressed training bytes matched the original ordered tokenized dataset SHA256.
- The upstream Boolean canonicalizer matched the pinned historical dependency hash.
- CPU statistical analysis and strict canonical re-scoring were rerun successfully on the server.

Server analysis runtime: Python 3.12.3, NumPy 1.26.4, SymPy 1.13.3.

No new GPU training or sampling was run. Full tokenizer prefix re-decoding was not repeated. Private archive locations and authentication material are excluded from this report.
