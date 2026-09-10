# Data

`python -m src.data.build_dataset` downloads the official Enterprise ATT&CK STIX bundle from `mitre-attack/attack-stix-data`. The raw bundle is reproducible and intentionally ignored. Its SHA-256 is recorded with the dataset statistics.

`processed/attack_tactic_dataset.csv` contains cleaned procedure text, technique and tactic labels, source entity metadata, citations and the original relationship ID. `processed/splits.csv` maps relationship IDs to the deterministic source-grouped partitions.

The processed dataset is derived from MITRE ATT&CK. Review MITRE's terms of use before redistributing it outside this experiment.
