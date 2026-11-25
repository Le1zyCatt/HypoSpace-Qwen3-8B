# Release verification

Completed on25 November2026:

- Recomputed all144 archived task/model/budget records, four cell means,20,000-resample paired intervals,100,000-draw sign-flip tests,and two-budget Holm-adjustedRR p-values. Both budgets match the historical summary.
- Re-parsed all archived final strings against the pinned upstream Boolean canonicalizer and strictGT membership; distinct valid/parseable counts match. The upstream canonicalizer file hash matches the historical dependency snapshot.
- Verified that decompressed training-data bytes match the original archived SHA256. The576 sequences and masked labels were preserved,not regenerated.
- Checked Python entry-point syntax,local document links,English prose,and publication files for credentials/server identifiers.
- Rendered and visually reviewed all12 editable slides; the schedule totals600 seconds,four speakers at150 seconds each. Notes contain paper/evidence references.

Analysis verification used both the bundled NumPy environment and local Python3.12/NumPy1.26.4 for matching statistics;canonical parsing was checked with local SymPy1.13.2. Historical training versions,including NumPy2.3.5 and SymPy1.14.0,are pinned separately.

Not performed during packaging:GPU retraining/new model sampling,installation of the complete historical CUDA stack,or re-decoding all archived prefixes with the original tokenizer. The optional prefix re-decoding command is provided;historical audits already checked pairing. Portable GPU entry points preserve the historical protocol but are not presented as newly completed end-to-end GPU runs. MS-TTRL raw teammate archives were not independently re-audited here.
