# Machine Learning — NTUA

Lecture decks and an exercise sheet for the machine learning course at the NTUA School of Civil Engineering.

## Get the PDFs

All slides are published as PDFs on the repository's releases page, filtered to this course:

**[ntua-courses releases — ml-course](https://github.com/constatza/ntua-courses/releases?q=ml-course)**

Pick the newest `ml-course/v…` release and download whichever PDFs you need; no clone or LaTeX is required. Files are named by kind and topic, for example `presentation_cnns.pdf` and `exercise_toy-diffusion-flow.pdf`; a translation adds its language, as in `exercise_cantilever-nn-surrogate_greek.pdf`. Exercise solutions are not published.

## Suggested reading order

1. Supervised learning — problem formulation and linear baselines
2. Neural nets — dense networks, backpropagation, approximation
3. Optimization — first-order algorithms through Adam
4. Generalization — bias, variance, and how to control them
5. Geometric deep learning — symmetry, invariance, equivariance
6. CNNs — convolutional architectures
7. RNNs — recurrent architectures
8. Transformers — attention-based sequence models
9. GNNs — message passing on graphs
10. PINNs & neural operators — scientific ML
11. Diffusion models
12. Flow matching
13. PCA / POD / ROM
14. PDF distances

Exercises: cantilever neural-network surrogate (pairs with decks 1--3); toy diffusion and flow matching (pairs with decks 11 and 12).

## Maintaining the course

Building, releasing and editing this course is covered in the [repository README](../README.md). Topic ownership is tracked in [`TOPICS.toml`](TOPICS.toml): each concept has one owning deck, and other decks cross-reference it instead of re-explaining. Books, slides, datasets and assignments are not in this repository.
