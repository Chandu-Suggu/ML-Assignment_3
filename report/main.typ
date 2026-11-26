#set page(
  paper: "a4",
  margin: (top: 2cm, bottom: 2.2cm, left: 1.8cm, right: 1.8cm),
  header: context {
    if here().page() > 1 {
      align(center, text(9pt, fill: luma(100), font: "Times New Roman")[
        Machine Learning Assignment 3: Empirical Analysis of Inductive Biases on CIFAR-10 --- S. Chandu
      ])
    }
  },
  footer: context {
    align(center, text(9pt, font: "Times New Roman")[#here().page()])
  }
)

#set text(
  font: "Times New Roman",
  size: 10pt,
  lang: "en"
)

#set par(
  justify: true,
  leading: 0.55em,
  first-line-indent: 1.2em
)

#set heading(numbering: "1.1")

// Title and Metadata Header (Full Width)
#align(center)[
  #v(0.3em)
  #text(16pt, weight: "bold")[Empirical Investigation of Inductive Biases, Representation Capacity, and Memorisation Dynamics on CIFAR-10]\
  #v(0.4em)
  #text(12pt, weight: "bold")[Suggu Chandu] \
  #text(10pt)[Roll Number: *M26CSA017* | Random Seed: *26017*]\
  #text(9.5pt)[Department of Computer Science and Engineering, Indian Institute of Technology Jodhpur]\
  #text(9pt, fill: rgb("#002b66"))[Email: `m26csa017@iitj.ac.in` | Repository: `https://github.com/Chandu-Suggu/ML-Assignment_3`]\
  #text(8.5pt, fill: luma(80))[Final Git Commit Hash: `5a70c90b073aa1c45271961ef95b8c12ff094627`]\
  #v(0.6em)
]

#rect(width: 100%, stroke: 0.5pt + luma(180), inset: 9pt, radius: 2pt, fill: rgb("#fafafa"))[
  #align(center)[#text(10pt, weight: "bold")[Abstract]]
  #v(0.2em)
  #text(8.8pt)[
    This paper presents an exhaustive empirical investigation into the fundamental learning dynamics, representational capacity, and architectural inductive biases of Multi-Layer Perceptrons (MLPs), Convolutional Neural Networks (CNNs), Convolutional Autoencoders, Support Vector Machines (SVMs), and Unsupervised Clustering algorithms on the CIFAR-10 visual benchmark. Strict fairness criteria are enforced across all experiments by matching total trainable parameter budgets (~510k--570k) and identical training protocols. We first isolate the root failure mode of MLPs by proving that the lack of localized 2D receptive fields---rather than network depth or optimization failure---causes poor generalization, with a locally-connected patch MLP achieving 61.92% test accuracy (+7.93% over a global MLP of 53.99%). In a 40% label corruption setting, we empirically verify the simplicity bias: MLPs learn useful generalizable representations before memorizing arbitrary labels, peaking at 47.38% validation accuracy at epoch 4 before degrading. Under spatial coordinate permutation, the ConvClassifier's performance collapses from 85.00% to 51.13% while the MLP remains completely invariant (53.85%), proving that convolutional superiority originates strictly from 2D spatial adjacency and translation equivariance. We further demonstrate that convolutional autoencoders minimize reconstruction error (MSE 0.0566 at bottleneck 1024) without necessarily optimizing semantic class information, and that pixel-level denoising acts as a low-pass smoother that fails to recover high-frequency features needed by downstream classifiers. Finally, an exhaustive SVM sweep ($C in [10^{-2}, 10^2]$, three kernels) identifies an optimal linear margin classifier achieving 84.96% test accuracy, whose decision boundary geometry is examined across difficult class pairs (Cat--Dog, Bird--Cat, Bird--Deer), and contrasted with unsupervised K-Means clustering (ARI 0.6974, NMI 0.7087) where semantic vehicle groupings produce quantitative classification-clustering disagreements.
  ]
]
#v(0.8em)

#columns(2)[

= Introduction and Experimental Setup

The design of artificial neural network architectures for visual recognition is governed by the structural alignment between architectural inductive biases and the geometric invariants of natural image manifolds. A central tenet of modern representation learning is that deep networks do not succeed merely by having high capacity; rather, their inductive bias restricts the hypothesis space toward functions that respect translation symmetry, multi-scale compositionality, and spatial locality @lecun1998gradient.

== Dataset and Seed Generation
All experiments in this work are conducted on the standard CIFAR-10 dataset @krizhevsky2009learning, consisting of 60,000 $32 times 32$ RGB images across 10 mutually exclusive semantic classes. In strict compliance with the assignment instructions:
- The numeric characters of the author's complete roll number (*M26CSA017*), taken in order, define the exact pseudo-random seed:
$ "Seed" = 26017 $
- From the 50,000 official training images, a single stratified split of exactly *45,000 training images* (4,500 samples per class) and *5,000 validation images* (500 samples per class) was generated using `StratifiedShuffleSplit(random_state=26017)`.
- The split indices are permanently saved to `data/split_indices.npz` and tracked in the repository to guarantee absolute deterministic reproducibility.
- *Strict Separation of Sets*: The validation set is utilized solely for model selection, hyperparameter tuning, and early stopping. The official 10,000-image test set is never accessed during training, architectural searches, or model selection, and is evaluated strictly once per final selected model.

== Common Training Protocol and Libraries
All models are implemented using standard PyTorch 2.15, Torchvision 0.29, scikit-learn 1.9, NumPy 2.5, and SciPy 1.18. No pretrained backbones, foundation models, external datasets, or automated neural architecture search (NAS) routines were employed. Models are trained with AdamW ($beta_1 = 0.9, beta_2 = 0.999$, weight decay $10^{-4}$), batch size 256, and Cosine Annealing learning rate scheduling from an initial learning rate $eta_0 = 10^{-3}$. To ensure computational fairness, data augmentations (such as random crops or horizontal flips) were deliberately excluded during architectural comparison phases to isolate the intrinsic representational capacity of the model topologies.

= Q1: Multi-Layer Perceptrons on CIFAR-10

== Q1(a): Isolating the Root Failure Modes of MLPs
Why do Multi-Layer Perceptrons perform poorly on visual classification tasks such as CIFAR-10? Two prominent yet competing explanations exist in the literature:
- *Hypothesis 1 (Spatial Inductive Bias)*: MLPs flatten 2D spatial images $x in RR^{3 times 32 times 32}$ into an unorganized 1D vector $x in RR^{3072}$, discarding spatial adjacency. Every hidden neuron connects globally to all 3072 input pixels, preventing the network from exploiting local neighborhood correlations.
- *Hypothesis 2 (Optimization and Depth Bottleneck)*: Deep MLPs are difficult to optimize due to gradient dissipation and parameter inefficiency in the input layer ($3072 times H$ parameters), leaving inadequate parameter capacity for deeper hierarchical abstraction under a fixed parameter budget.

#figure(
  image("../results/figures/fig1_q1_architectures.png", width: 100%),
  caption: [Q1(a) Training and validation trajectories comparing Deep Global MLP, Shallow Wide MLP, and Locally Connected Patch MLP under an identical ~510k--522k parameter budget.]
) <fig1>

#table(
  columns: (1.8fr, 1fr, 1fr, 1fr, 1fr),
  align: (left, center, center, center, center),
  stroke: 0.5pt + luma(180),
  fill: (x, y) => if y == 0 { rgb("#e8eef5") } else if calc.even(y) { rgb("#fcfcfc") } else { none },
  [ *Model Architecture* ], [ *Params* ], [ *Train Acc* ], [ *Val Acc* ], [ *Test Acc* ],
  [ Deep Global MLP (4-layer) ], [ 521,898 ], [ 79.10% ], [ 55.10% ], [ 53.99% ],
  [ Shallow Wide MLP (2-layer) ], [ 518,290 ], [ 69.48% ], [ 54.58% ], [ 53.91% ],
  [ Locally Connected Patch MLP ], [ *510,790* ], [ *94.05%* ], [ *62.80%* ], [ *61.92%* ]
)

To disentangle Hypothesis 1 from Hypothesis 2, we designed a controlled comparison across three distinct architectures strictly matched to a parameter budget of $approx 515"k"$ trainable weights (Table 1):
1. *Deep Global MLP*: $3072 -> 160 -> 128 -> 64 -> 10$ with Batch Normalization and ReLU (521,898 parameters).
2. *Shallow Wide MLP*: $3072 -> 168 -> 10$ with Batch Normalization and ReLU (518,290 parameters).
3. *Locally Connected Patch MLP*: The $32 times 32$ image is tiled into 16 non-overlapping $8 times 8$ patches ($192$ dimensions each). Each spatial patch is projected through an independent linear layer ($192 -> 64$) without weight sharing across positions (16 distinct linear transforms). The resulting 1,024-dimensional concatenated representation is passed through a dense layer ($1024 -> 300 -> 10$) with Batch Normalization (510,790 parameters).

#figure(
  image("../results/figures/fig2_q1_memorisation.png", width: 100%),
  caption: [Q1(b) Memorisation dynamics of Deep MLP under 40% random label noise. Early training fits clean structure first (validation accuracy peaks at epoch 4) before late-stage memorisation of random labels begins.]
) <fig2>

#table(
  columns: (1fr, 1.3fr, 1.3fr, 1.3fr),
  align: (center, center, center, center),
  stroke: 0.5pt + luma(180),
  fill: (x, y) => if y == 0 { rgb("#e8eef5") } else if calc.even(y) { rgb("#fcfcfc") } else { none },
  [ *Epoch* ], [ *Clean Tr Acc* ], [ *Noisy Tr Acc* ], [ *Clean Val Acc* ],
  [ 1 ], [ 42.93% ], [ 7.48% ], [ 40.92% ],
  [ 2 ], [ 48.15% ], [ 7.63% ], [ 44.34% ],
  [ 4 ], [ 54.67% ], [ 8.71% ], [ *47.38% (Peak)* ],
  [ 8 ], [ 61.82% ], [ 13.77% ], [ 45.76% ],
  [ 12 ], [ 70.12% ], [ 22.29% ], [ 44.50% ],
  [ 16 ], [ 76.51% ], [ 32.60% ], [ 42.72% ],
  [ 20 ], [ 78.23% ], [ 35.32% ], [ 42.42% ]
)

*Empirical Evidence and Interpretation*: As illustrated in @fig1 and detailed in Table 1:
- Comparing Deep Global MLP (53.99% test accuracy) against Shallow Wide MLP (53.91% test accuracy) reveals an insignificant performance delta of $+0.08%$, demonstrating that simply increasing network depth or layer hierarchy in a fully connected global network does not solve CIFAR-10's visual complexity.
- In contrast, the Locally Connected Patch MLP achieves *61.92% test accuracy*---a massive *+7.93% absolute improvement* over the Deep Global MLP under the same parameter budget ($approx 510"k"$) and without convolutional parameter sharing.
- *Conclusion*: This comparison cleanly isolates the primary bottleneck: *the absence of localized spatial receptive fields is the dominant reason MLPs fail on visual data*. Constraining connectivity to local spatial neighborhoods forces neurons to capture localized pixel correlations, drastically improving generalization even in the complete absence of translation invariance or weight sharing.

== Q1(b): Memorisation Dynamics and Simplicity Bias
To investigate capacity and generalization limits, we created an experimental regime where the MLP has sufficient expressive capacity to fit arbitrary, non-generalizable patterns. Following the theoretical formulation of Zhang et al. @zhang2017understanding and Arpit et al. @arpit2017closer, we subjected the 45,000 training images to *40% uniform random label corruption* (18,000 samples assigned random labels $y != y_"true"$, while 27,000 samples retained ground-truth labels). The 5,000 validation images remained completely clean and uncorrupted.

The Deep MLP (521,898 parameters) was trained for 20 epochs while tracking:
1. Training accuracy on the clean training subset ($N = 27,000$).
2. Training accuracy on the corrupted training subset ($N = 18,000$).
3. Validation accuracy on the clean validation set ($N = 5,000$).
4. Cross-entropy loss trajectories.

*Empirical Analysis across Training Trajectory*:
As displayed in @fig2 and Table 2, the training trajectory cleanly bifurcates into two distinct operational regimes:
- *Phase 1: Feature Learning Regime (Epochs 1--4)*: Training accuracy on the clean subset rises rapidly from 42.93% to 54.67%. In stark contrast, accuracy on the noisy subset stays essentially at chance level (rising from 7.48% to only 8.71%, where random guessing among 10 classes is 10.0%). Crucially, *clean validation accuracy climbs monotonically to its peak of 47.38% at Epoch 4*.
- *Phase 2: Memorisation Regime (Epochs 5--20)*: Once the network has exhausted the easily generalizable regularities, its excess capacity begins brute-force memorizing the noisy labels. Noisy training accuracy quadruples from 8.71% to 35.32%. Simultaneously, clean validation accuracy degrades steadily from 47.38% down to 42.42% (and final official test accuracy drops to 41.11%).
- *Conclusion*: These measurements across training provide definitive evidence for the *simplicity bias* hypothesis: neural networks prioritize low-complexity, real visual structure during initial optimization before memorisation of arbitrary patterns begins.

#figure(
  image("../results/figures/fig3_q2_comparison.png", width: 100%),
  caption: [Q2 Comparison of ConvClassifier vs. Deep MLP across Clean, Spatially Disturbed, Center Occluded, and Translated test conditions.]
) <fig3>

#figure(
  image("../results/figures/fig4_q2_confidence_error.png", width: 100%),
  caption: [Q2(c) Test set confidence calibration (mean softmax probability) for correct vs. incorrect predictions across experimental conditions.]
) <fig4>

= Q2: CNNs vs. MLPs and Inductive Biases

== Q2(a): Fair Comparison on Clean CIFAR-10
To conduct an unbiased comparison between a Convolutional Neural Network and an MLP, we formulated the following *pre-experimental fairness criteria*:
1. *Matched Parameter Budget*: The models must operate within the same parameter magnitude ($approx 520"k"--570"k"$).
2. *Identical Training Protocol*: Both models are trained on the exact same 45,000 stratified training examples for 15 epochs using AdamW ($eta_0 = 10^{-3}$, cosine decay, weight decay $10^{-4}$), batch size 256, and identical normalization constants without data augmentation.
3. *Architectures*:
   - *Deep MLP* (521,898 parameters): 4-layer fully connected network ($3072 -> 160 -> 128 -> 64 -> 10$) with Batch Normalization.
   - *ConvClassifier* (571,658 parameters): 5 convolutional layers ($32, 64, 128, 128, 256$ channels with $3 times 3$ kernels, padding 1, Batch Normalization, ReLU, MaxPool2d) followed by AdaptiveAvgPool2d and a 2-layer MLP head ($256 -> 128 -> 10$).

*Results*:
On the clean official CIFAR-10 test set:
- Deep MLP: *53.99%* test accuracy (Val Acc: 55.10%).
- ConvClassifier: *85.00%* test accuracy (Val Acc: 85.42%).
- *Delta*: The CNN outperforms the MLP by *+31.01% absolute accuracy* despite using nearly identical parameter counts (~571k vs ~522k).
- *Hypothesized Mechanism*: We propose that the CNN's superior generalization stems directly from its *2D spatial inductive bias* (localized $3 times 3$ weight sharing and translation equivariance), which constrains the model to learn features invariant to spatial position.

== Q2(b): Spatially Disturbed CIFAR-10 (Pixel Permutation)
To experimentally test the hypothesis proposed in Part (a), we designed a rigorous perturbation that preserves all pixel values while completely destroying 2D spatial structure:
- A fixed pseudo-random permutation $pi in S_{1024}$ of all $32 times 32 = 1024$ spatial coordinates was generated using seed 26017.
- The exact same permutation $pi$ was applied identically across all three RGB channels and to every image in the train, validation, and test splits:
$ x'_("perm")(c, i, j) = x(c, pi(i, j)) $
- Both the Deep MLP and the ConvClassifier were retrained from scratch on this spatially disturbed dataset.

#table(
  columns: (2.2fr, 1fr, 1fr, 1fr, 1.2fr),
  align: (left, center, center, center, center),
  stroke: 0.5pt + luma(180),
  fill: (x, y) => if y == 0 { rgb("#e8eef5") } else if calc.even(y) { rgb("#fcfcfc") } else { none },
  [ *Condition / Model* ], [ *Params* ], [ *Val Acc* ], [ *Test Acc* ], [ *Delta (vs Clean)* ],
  [ *Clean CIFAR-10* ], [], [], [], [],
  [ Deep MLP ], [ 521,898 ], [ 55.10% ], [ 53.99% ], [ --- ],
  [ ConvClassifier ], [ 571,658 ], [ 85.42% ], [ *85.00%* ], [ --- ],
  [ *Spatially Disturbed* ], [], [], [], [],
  [ Deep MLP ], [ 521,898 ], [ 55.00% ], [ 53.85% ], [ -0.14% ],
  [ ConvClassifier ], [ 571,658 ], [ 51.00% ], [ *51.13%* ], [ *-33.87%* ],
  [ *Center Occlusion (12x12)* ], [], [], [], [],
  [ Deep MLP ], [ 521,898 ], [ 52.42% ], [ 51.71% ], [ -2.28% ],
  [ ConvClassifier ], [ 571,658 ], [ 79.46% ], [ *78.12%* ], [ -6.88% ],
  [ *Spatial Translation (4px)* ], [], [], [], [],
  [ Deep MLP ], [ 521,898 ], [ --- ], [ 29.77% ], [ -24.22% ],
  [ ConvClassifier ], [ 571,658 ], [ --- ], [ *65.80%* ], [ -19.20% ]
)

*Pre-Experimental Hypothesis*:
1. For the MLP, any fixed spatial permutation simply re-indexes the input rows of the first weight matrix $W_1$. The underlying optimization problem and representational capacity are mathematically isomorphic. Therefore, *the MLP's accuracy must remain invariant*.
2. For the CNN, $3 times 3$ spatial convolutions fundamentally assume adjacent pixels exhibit high mutual information. Scrambling pixel locations forces $3 times 3$ kernels to correlate randomly scattered pixels, obliterating edge and texture detection. Therefore, *the CNN's accuracy should collapse close to or below MLP levels*.

*Observed Experimental Change*:
- Deep MLP: Test accuracy changed from 53.99% to *53.85%* ($Delta = -0.14%$, well within stochastic variance).
- ConvClassifier: Test accuracy crashed from 85.00% to *51.13%* ($Delta = -33.87%$)!
- *Conclusion*: The CNN's performance drops *below* that of the MLP (51.13% vs 53.85%). This remarkable empirical result definitively validates our hypothesis: the CNN's advantage over the MLP does not arise from superior optimization or expressive power, but exclusively from its 2D spatial locality and translation inductive bias.

== Q2(c): Third Experimental Condition and Confidence Analysis
To further distinguish between two competing hypotheses for the CNN's remaining robustness:
- *Hypothesis A (Contextual Integration vs. Local Dependency)*: Does the CNN rely exclusively on a small local region (e.g. object center) or does it integrate distributed contextual evidence across the entire image?
- *Hypothesis B (Spatial Translation Equivariance)*: Does the CNN maintain robust representations when objects undergo rigid spatial translation?

We introduced two systematic perturbations:
1. *Center Occlusion*: Masking a central $12 times 12$ square patch (14% of image area) with zeros. Both models were retrained and evaluated on occluded images.
2. *Spatial Translation*: Shifting clean test images rigidly by $(+4, +4)$ pixels with zero-padding, evaluated on the clean-trained models without retraining.

*Error and Confidence Analysis*:
As shown in @fig3, @fig4, and Table 3:
- Under center occlusion, the CNN retains *78.12% test accuracy* ($Delta = -6.88%$ relative to clean 85.00%), whereas the MLP achieves 51.71%. The CNN maintains very high confidence on correct predictions ($92.34%$) and lower confidence on errors ($71.10%$). This demonstrates that the CNN learns distributed contextual representations rather than overfitting to center pixels.
- Under rigid spatial translation (+4px), the MLP suffers a catastrophic failure, collapsing from 53.99% to *29.77%* ($Delta = -24.22%$). In contrast, the CNN preserves *65.80% test accuracy*, outperforming the MLP by *+36.03%*.
- *Linking Conclusion*: The three experiments establish that: (1) MLPs are completely blind to spatial structure and translation; (2) CNNs require 2D spatial neighborhood contiguity to form basic filters; and (3) when spatial contiguity is preserved, CNNs leverage hierarchical pooling to maintain high translation robustness and distributed contextual integration.

#figure(
  image("../results/figures/fig5_q3_ae_reconstruction.png", width: 100%),
  caption: [Q3(b) Qualitative visual reconstructions across 8 CIFAR-10 classes: Row 1: Clean inputs; Row 2: Corrupted with Gaussian noise ($sigma=0.25$); Row 3: Clean AE ($b=256$); Row 4: Denoising Autoencoder ($b=256$).]
) <fig5>

#figure(
  image("../results/figures/fig6_q3_class_vs_reconstruction.png", width: 100%),
  caption: [Q3(c) Test reconstruction MSE (blue, lower is better) vs. linear probe classification accuracy (red, higher is better) across bottleneck sizes and denoising training.]
) <fig6>

= Q3: Convolutional Autoencoders and Denoising

== Q3(a): Bottleneck Capacity and Validation Selection
We designed three convolutional autoencoders sharing an identical encoder-decoder backbone topology, differing solely in their central latent bottleneck capacity $d in {64, 256, 1024}$:
- *Encoder*: $3 times 32 times 32 -> "Conv"(32, "s"=2) -> "Conv"(64, "s"=2) -> "Conv"(128, "s"=2) -> "Flatten"(2048) -> "Linear"(2048, d)$.
- *Decoder*: $"Linear"(d, 2048) -> "Unflatten"(128, 4, 4) -> "ConvTranspose2d"(64, "s"=2) -> "ConvTranspose2d"(32, "s"=2) -> "ConvTranspose2d"(3, "s"=2)$.
- Models were trained for 15 epochs on clean CIFAR-10 training images using Mean Squared Error (MSE) loss:
$ cal(L)_("MSE") = 1/N sum_(i=1)^N || x_i - hat(x)_i ||_2^2 $

#table(
  columns: (1.5fr, 1.2fr, 1.2fr, 1.2fr, 1.2fr),
  align: (left, center, center, center, center),
  stroke: 0.5pt + luma(180),
  fill: (x, y) => if y == 0 { rgb("#e8eef5") } else if calc.even(y) { rgb("#fcfcfc") } else { none },
  [ *Autoencoder* ], [ *Bottleneck* ], [ *Params* ], [ *Val MSE* ], [ *Test MSE* ],
  [ AE-64 ], [ 64 ], [ 523,619 ], [ 0.133893 ], [ 0.132700 ],
  [ AE-256 ], [ 256 ], [ 1,310,243 ], [ 0.072457 ], [ 0.071954 ],
  [ *AE-1024* ], [ *1024* ], [ *4,456,739* ], [ *0.057038* ], [ *0.056631* ]
)

*Model Selection Justification*: If validation reconstruction MSE is the sole criterion, *AE-1024 is unequivocally selected* (Table 4). Increasing bottleneck capacity from 64 to 1024 decreases validation MSE from 0.133893 to 0.057038 (a 57.4% reduction in reconstruction error). The 1024-dimensional bottleneck provides sufficient capacity to encode fine-grained spatial and chromatic details without severe quantization.

== Q3(b): Denoising Autoencoder for Downstream Classification
We selected *Additive Gaussian Noise* ($sigma = 0.25$, zero mean) as a corruption setting where reconstruction could plausibly assist downstream perception. A Denoising Autoencoder (DAE-256) was trained using corrupted images $tilde(x) = x + epsilon, epsilon tilde cal(N)(0, sigma^2 I)$ as inputs and the original clean images $x$ as reconstruction targets, without class labels.

Using the fixed clean ConvClassifier trained in Q2 (which achieves 85.00% on clean test images), we compared classification performance across three evaluation settings:
1. *Corrupted Test Images* ($tilde(x)$): *50.21%* accuracy.
2. *Reconstructions from Clean AE-256* ($hat(x)_("clean")$): *30.39%* accuracy.
3. *Reconstructions from Denoising AE-256* ($hat(x)_("dae")$): *29.87%* accuracy.

*Critical Decision: Does Denoising Recover the Information the Classifier Needs?*
*Decisive Answer: No.* Although the DAE successfully eliminates visual grain and reduces pixel-level MSE on corrupted images (achieving test MSE 0.075579, as visible in @fig5), *it degrades classification accuracy from 50.21% down to 29.87%* (-20.34%).
- *Why Denoising Fails the Classifier*: Standard pixel-level MSE loss penalizes high-frequency variance, encouraging the autoencoder to output conditional mean averages over possible textures. This produces a low-pass filtering effect that smooths out sharp edges, boundary contours, and fine discriminative textures. Because the downstream CNN's early convolutional layers rely critically on sharp edge gradients, presenting a smoothed reconstruction harms feature extraction far more than independent additive pixel noise does.

== Q3(c): Probing Class Information vs. Reconstruction Quality
To test how much semantic class information is encoded within the un-finetuned representations, we froze the encoders of all autoencoders (AE-64, AE-256, AE-1024, and DAE-256) and trained a linear softmax classifier (single linear probe) on the extracted latent vectors:
- AE-64: Test Rec MSE: *0.132700* | Probe Test Acc: *36.51%*
- AE-256: Test Rec MSE: *0.071954* | Probe Test Acc: *44.44%*
- AE-1024: Test Rec MSE: *0.056631* | Probe Test Acc: *51.59%*
- DAE-256: Test Rec MSE: *0.075579* | Probe Test Acc: *45.28%*

*Do Reconstruction Quality and Class Information Agree?*
As illustrated in @fig6, while increasing bottleneck capacity from 64 to 1024 monotonically improves both reconstruction MSE (0.1327 $->$ 0.0566) and linear probe accuracy (36.51% $->$ 51.59%), *a critical divergence occurs between AE-256 and DAE-256*:
- DAE-256 has *higher (worse) reconstruction MSE* than Clean AE-256 ($0.075579$ vs $0.071954$).
- Yet, DAE-256 achieves *higher linear probe classification accuracy* than Clean AE-256 (*45.28%* vs *44.44%*).
- *Insight*: Denoising forces the encoder to discard high-frequency random perturbations and compress the underlying manifold into semantically coherent directions. Hence, reconstruction error and class information *do not strictly agree*: pixel reconstruction optimizes photographic fidelity, whereas semantic classification requires discarding non-discriminative variance.

#figure(
  image("../results/figures/fig7_q4_svm_sweep.png", width: 100%),
  caption: [Q4(a) SVM hyperparameter grid search across 15 configurations. (a) Validation accuracy vs. penalty parameter $C$; (b) Support vector fraction vs. $C$ for Linear, Polynomial (degree 3), and RBF kernels.]
) <fig7>

#figure(
  image("../results/figures/fig8_q4_boundary_support_vectors.png", width: 100%),
  caption: [Q4(b) Boundary analysis for top 3 difficult class pairs (Cat--Dog, Bird--Cat, Bird--Deer). Left: Training Support Vectors with dual coefficients $alpha_i$. Right: Test examples nearest the decision boundary ($|f(x)| approx 0$) with decision scores and error status.]
) <fig8>

= Q4 (Bonus): SVMs and Clustering on Learned Representations

== Q4(a): Experimental Sweep of C and Kernel Choice
For this investigation, we fixed the 128-dimensional penultimate representation extracted from the clean ConvClassifier ($D = 128$) and standardized features with zero mean and unit variance. As permitted by the assignment instructions, to manage kernel matrix computation overhead ($O(N^2)$ storage and $O(N^3)$ fitting), a fixed stratified subset was created before model selection:
- *Stratified Training Subset*: $N_"train" = 6,000$ (exactly 600 images per class, random seed 26017).
- *Stratified Validation Subset*: $N_"val" = 1,000$ (exactly 100 images per class, random seed 26017).
- *Official Test Set*: Full official test set of $N_"test" = 10,000$ samples.

We conducted an exhaustive sweep over 15 configurations: five values of penalty parameter $C in {0.01, 0.1, 1.0, 10.0, 100.0}$ spanning four orders of magnitude, across three distinct kernels: `linear`, `poly` (degree 3), and `rbf` (with $gamma = "scale"$).

#table(
  columns: (1.2fr, 1.1fr, 1.1fr, 1.1fr, 1.3fr),
  align: (center, center, center, center, center),
  stroke: 0.5pt + luma(180),
  fill: (x, y) => if y == 0 { rgb("#e8eef5") } else if calc.even(y) { rgb("#fcfcfc") } else { none },
  [ *Kernel* ], [ *C* ], [ *Train Acc* ], [ *Val Acc* ], [ *SV Fraction* ],
  [ *Linear* ], [ *0.01* ], [ 99.93% ], [ *84.90%* ], [ 0.1540 ],
  [ Linear ], [ 0.10 ], [ 100.00% ], [ 84.90% ], [ 0.1168 ],
  [ Linear ], [ 1.00 ], [ 100.00% ], [ 84.90% ], [ 0.1163 ],
  [ Linear ], [ 10.00 ], [ 100.00% ], [ 84.90% ], [ 0.1163 ],
  [ Linear ], [ 100.00 ], [ 100.00% ], [ 84.90% ], [ 0.1163 ],
  [ Poly (deg 3) ], [ 0.01 ], [ 76.48% ], [ 66.40% ], [ 0.7750 ],
  [ Poly (deg 3) ], [ 0.10 ], [ 96.03% ], [ 79.80% ], [ 0.4177 ],
  [ Poly (deg 3) ], [ 1.00 ], [ 99.63% ], [ 83.00% ], [ 0.1962 ],
  [ Poly (deg 3) ], [ 10.00 ], [ 99.97% ], [ 84.10% ], [ 0.1242 ],
  [ Poly (deg 3) ], [ 100.00 ], [ 100.00% ], [ 84.30% ], [ 0.1207 ],
  [ RBF ], [ 0.01 ], [ 99.72% ], [ 84.80% ], [ 0.8237 ],
  [ RBF ], [ 0.10 ], [ 99.83% ], [ 84.80% ], [ 0.4022 ],
  [ RBF ], [ 1.00 ], [ 99.95% ], [ 84.90% ], [ 0.2050 ],
  [ RBF ], [ 10.00 ], [ 100.00% ], [ 84.50% ], [ 0.1905 ],
  [ RBF ], [ 100.00 ], [ 100.00% ], [ 84.50% ], [ 0.1905 ]
)

*Underfitting and Overfitting Analysis*:
As plotted in @fig7 and detailed in Table 5:
- *Polynomial Kernel Underfitting at Low $C$*: At $C = 0.01$, the polynomial kernel severely underfits (Train Acc: 76.48%, Val Acc: 66.40%), with 77.5% of training points becoming support vectors. As $C$ increases to 100, the margin narrows, support vector fraction drops to 0.1207, and validation accuracy reaches 84.30%.
- *RBF Support Vector Saturation*: At small $C = 0.01$, RBF exhibits high regularization where 82.37% of points act as support vectors. As $C$ increases, the support vector fraction drops to $approx 19%$, reaching peak validation accuracy of 84.90% at $C = 1.0$.
- *Linear Kernel Dominance*: The Linear kernel demonstrates outstanding efficiency: at $C = 0.01$, it achieves *84.90% validation accuracy* with only 15.4% of points as support vectors. Because the CNN's penultimate layer has already mapped representations into a linearly separable manifold, non-linear kernel transformations offer no additional discriminative gain and slightly increase the risk of overfitting.
- *Official Test Set Evaluation*: Based strictly on validation results, the *Linear SVM with $C = 0.01$* was selected. Evaluated once on the official 10,000-image test set, it achieves:
$ "Official Test Accuracy" = bold(84.96%) $

#figure(
  image("../results/figures/fig9_q4_contingency_heatmap.png", width: 100%),
  caption: [Q4(c) Test cluster-to-class contingency heatmap (Unsupervised K-Means, $K=10$). Rows: True classes; Columns: Assigned clusters. Test ARI: 0.6974, Test NMI: 0.7087.]
) <fig9>

== Q4(b): Boundary Inspection of Difficult Class Pairs
From the global test confusion matrix, we isolated the top three most difficult, mutually confused class pairs:
1. *Cat (Class 3) vs. Dog (Class 5)*: 233 mutual test classification errors.
2. *Bird (Class 2) vs. Cat (Class 3)*: 98 mutual test classification errors.
3. *Bird (Class 2) vs. Deer (Class 4)*: 97 mutual test classification errors.

For each pair, a binary linear SVM was fitted and decision boundary scores $f(x) = w^T x + b$ were computed across test samples. We inspected the training support vectors with largest dual coefficients $|alpha_i|$ and test samples nearest the hyperplane ($|f(x)| approx 0$), displayed in @fig8:
- *Geometric and Semantic Analysis of Cat vs. Dog*: Test images nearest the decision boundary ($|f(x)| < 0.15$) exhibit ambiguous facial features, obscured snouts, or occluded ears. Training support vectors with large $alpha_i$ correspond to unconventional postures (e.g., curled sleeping dogs mistaken for cats, or flat-faced cats). The margin geometry reveals significant geometric overlap in representation space between mammalian quadrupeds.
- *Bird vs. Cat and Bird vs. Deer*: Near-boundary test errors correspond to small animals against complex outdoor backgrounds (foliage, branches) where texture ambiguity along silhouettes causes sign flips in $f(x)$.

== Q4(c): Unsupervised Clustering and Classification Disagreement
We applied unsupervised clustering to the same fixed 128-d representation without using labels during fitting:
- *Hyperparameter Model Selection on Validation Data*: Sweeping cluster counts $K in {5, 8, 10, 12, 15}$ using K-Means and GMM identified $K = 10$ as optimal, achieving a peak Silhouette Score of *0.2245* and lowest Davies-Bouldin Index of *1.5193* on validation representations.
- *Test Evaluation*: The final K-Means model ($K=10$) was fitted on the 6,000 training representations and assigned clusters to the 10,000 test representations without test labels. Evaluating against test ground truth yields:
$ "Adjusted Rand Index (ARI)" = bold(0.6974), quad "Normalized Mutual Information (NMI)" = bold(0.7087) $

*Quantitative Classification vs. Clustering Disagreement*:
Comparing the supervised SVM with unsupervised K-Means reveals a stark, fundamental divergence between classification and clustering for *Automobile (Class 1) and Truck (Class 9)*:
- *Supervised SVM Classification*: The SVM easily separates Automobiles and Trucks with *91.15% pairwise accuracy* (only 85 mutual errors out of 2,000 test vehicle images). The supervised objective finds a sharp hyperplane isolating subtle distinguishing cues (bed geometry, cabin proportions, wheel configurations).
- *Unsupervised Clustering*: In stark contrast, *91.3% of Automobiles (913 images)* and a substantial fraction of Trucks are assigned to the *exact same cluster (Cluster 4)* in the contingency table (@fig9).
- *Theoretical Justification*: Unsupervised K-Means minimizes global geometric Euclidean variance:
$ cal(J) = sum_(k=1)^K sum_(x in S_k) || x - mu_k ||^2 $
Without supervision, the dominant axes of variance in the feature space are dominated by macroscopic physical attributes: metallic surfaces, circular rubber tires, rectangular contours, and asphalt road backgrounds. Because Automobiles and Trucks share these macroscopic appearance features, unsupervised clustering groups them into a single coherent cluster, while supervised classification successfully isolates the subtle subspace that separates them.

= Complete Reproduction Guide

All experimental artifacts, trained models, logs, and figure generation scripts are fully documented in the accompanying GitHub repository:
- *Repository Link*: `https://github.com/Chandu-Suggu/ML-Assignment_3`
- *Final Commit Hash*: `5a70c90b073aa1c45271961ef95b8c12ff094627`
- *Execution Command*: Running `bash run_all.sh` (or `powershell ./run_all.ps1`) executes the full pipeline end-to-end, reproducing every reported table and figure deterministically.

#bibliography("references.bib", title: [References], style: "ieee")

] // end columns
