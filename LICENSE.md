# MINR-Mamba License

MINR-Mamba is released for academic research and other non-commercial use.

This repository contains original MINR-Mamba code together with code and
implementation ideas derived from several upstream image restoration projects.
Because part of the codebase is derived from Restormer, which is distributed
under an Academic Public License for non-commercial use, this repository should
be used and redistributed under the same non-commercial restriction unless
separate permission is obtained from the relevant copyright holders.

## Permissions

- Non-commercial academic and research use
- Private use
- Modification
- Redistribution of source code and modified source code

## Limitations

- Commercial use is not permitted without separate permission from the relevant
  copyright holders.
- The software is provided without warranty.
- The authors and upstream contributors are not liable for damages arising from
  use of the software.

## Conditions

- Keep this license file with redistributed copies or modified versions.
- Keep the third-party notices and attribution information in
  `THIRD_PARTY_NOTICES.md`.
- Cite MINR-Mamba and the relevant upstream works when using this repository in
  research.
- If you redistribute modified code derived from this repository, distribute it
  under terms that preserve the non-commercial restriction required by the
  Restormer-derived components.

## Code Provenance

MINR-Mamba includes or adapts code, structure, and implementation ideas from the
following projects and papers:

- **Restormer: Efficient Transformer for High-Resolution Image Restoration**
  - Usage in this repository: BasicSR-style training and validation framework,
    image restoration utilities, model wrapper code, metrics, and deraining
    workflow structure.
  - License basis: Academic Public License for non-commercial use.
  - The original Transformer-only Restormer architecture is not included in this
    cleaned MINR-Mamba package, but derived framework code remains.

- **MambaIR: A Simple Baseline for Image Restoration with State-Space Model**
  - Usage in this repository: state-space restoration design and Mamba-style
    restoration implementation references.
  - Upstream license: Apache License 2.0.

- **MambaIRv2: Attentive State Space Restoration**
  - Usage in this repository: attentive state-space restoration design,
    semantic routing, and related Mamba-style implementation references.
  - Upstream license: Apache License 2.0, as indicated by the upstream MambaIR
    repository.

- **Implicit Neural Representation for Cooperative Low-light Image Enhancement**
  - Usage in this repository: INR-based image restoration ideas and related
    implementation references.
  - No top-level license file was found in the local upstream copy checked for
    this release.

- **Bidirectional Multi-Scale Implicit Neural Representations for Image
  Deraining**
  - Usage in this repository: deraining and pixel-coordinate INR ideas and
    related implementation references.
  - No top-level license file was found in the local upstream copy checked for
    this release.

For more detail, see `THIRD_PARTY_NOTICES.md`.

## Restormer Academic Public License Notice

The following non-commercial academic license notice is retained for
Restormer-derived code:

### Permissions

- Non-commercial use
- Modification
- Distribution
- Private use

### Limitations

- Commercial use
- Liability
- Warranty

### Conditions

- License and copyright notice
- Same license

Restormer is free for use in noncommercial settings: at academic institutions
for teaching and research use, and at non-profit research organizations. You
can use Restormer in your research, academic work, non-commercial work, projects
and personal work. We only ask you to credit us appropriately.

You have the right to use the software, to distribute copies, to receive source
code, to change the software and distribute your modifications or the modified
software. If you distribute verbatim or modified copies of this software, they
must be distributed under this license.

This license guarantees that you're safe when using Restormer in your work, for
teaching or research. This license guarantees that Restormer will remain
available free of charge for nonprofit use. You can modify Restormer to your
purposes, and you can also share your modifications.

If you would like to use Restormer in commercial settings, contact the Restormer
authors to discuss options.
