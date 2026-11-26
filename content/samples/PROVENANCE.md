# Replay photograph

- File: `labrador.jpg` (960 × 640 JPEG, 144,377 bytes).
- Subject: Labrador retriever (an ImageNet class); photograph captured 20 July 2019.
- Author: Dktue.
- Source: https://commons.wikimedia.org/wiki/File:Portrait_of_a_labrador_retriever.jpg
- Download: https://upload.wikimedia.org/wikipedia/commons/thumb/3/35/Portrait_of_a_labrador_retriever.jpg/960px-Portrait_of_a_labrador_retriever.jpg
- Licence: **CC0-1.0 / public domain dedication**, explicitly tagged CC-Zero on the source page.
- Licence URL: https://creativecommons.org/publicdomain/zero/1.0/
- SHA-256: `8e04f295a5a2e907dfe9924b5e2dbee594c0942e8ecca7556fd8d64aa1818777`
- Modification: Wikimedia's 960 px thumbnail; downloaded bytes retained unchanged.

The snapshot console tool alone decodes the JPEG with StbImageSharp 2.30.16
(MIT / public domain dual licence, managed C#, no native decoder). It resizes the
short side to 256 by nearest neighbour, centre crops to 224, and subtracts RGB
constants (123,117,104). This remains placeholder preprocessing, not ANX-12.
The browser and desktop show the crop reconstructed from stored input floats.
Weights are untrained: top-5 indices and probabilities are meaningless predictions.
The former synthetic sample is removed; no test needs it.
