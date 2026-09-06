# Third-Party Notices

MINR-Mamba combines original project code with code and ideas derived from
existing image restoration projects. Keep this file with the repository when
redistributing the code.

## Restormer: Efficient Transformer for High-Resolution Image Restoration

- Upstream project/paper: Restormer: Efficient Transformer for
  High-Resolution Image Restoration
- Venue: CVPR 2022
- License: Academic Public License, non-commercial use
- Local license file: `LICENSE.md`
- Usage in this repository: BasicSR-style training, validation, image
  restoration utilities, metrics, model wrapper code, and deraining workflow
  structure.

The original Transformer-only architecture has been removed from this cleaned
MINR-Mamba package, but derived framework code remains. Keep the upstream
license notice and follow its non-commercial restriction.

## MambaIR: A Simple Baseline for Image Restoration with State-Space Model

- Upstream project/paper: MambaIR: A Simple Baseline for Image Restoration with
  State-Space Model
- Venue: ECCV 2024
- License: Apache License 2.0
- Usage in this repository: state-space restoration design and Mamba-style
  restoration building blocks.

If code copied from MambaIR is redistributed, retain the Apache-2.0 license
notice required by the upstream project.

## MambaIRv2: Attentive State Space Restoration

- Upstream project/paper: MambaIRv2: Attentive State Space Restoration
- Reported venue/status in upstream README: CVPR 2025 / arXiv 2024
- License: Apache License 2.0, inherited from the MambaIR repository
- Usage in this repository: attentive state-space restoration design, semantic
  routing, and related Mamba-style implementation references.

If code copied from MambaIRv2 is redistributed, retain the Apache-2.0 license
notice required by the upstream project.

## Implicit Neural Representation for Cooperative Low-light Image Enhancement

- Upstream project: NeRCo
- Upstream paper: Implicit Neural Representation for Cooperative Low-light
  Image Enhancement
- Venue: ICCV 2023
- License file found locally: none
- Usage in this repository: INR-based image restoration ideas and related
  implementation references.

If exact NeRCo source code is redistributed, verify permission from the upstream
authors because no top-level license file was found in the local copy.

## Bidirectional Multi-Scale Implicit Neural Representations for Image Deraining

- Upstream project: NeRD-Rain
- Upstream paper: Bidirectional Multi-Scale Implicit Neural Representations for
  Image Deraining
- Venue: CVPR 2024
- License file found locally: none
- Usage in this repository: deraining and pixel-coordinate INR ideas and related
  implementation references.

If exact NeRD-Rain source code is redistributed, verify permission from the
upstream authors because no top-level license file was found in the local copy.

## Notes

This notice is an engineering summary of known code provenance. It is not legal
advice. Before public release, confirm that every copied source file is allowed
to be redistributed under the intended repository terms.
