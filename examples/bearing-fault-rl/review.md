# Paper Review

**Risk score (computed):** 55/100

**Citation check:** 67/67 quotes verified against the paper text; 11 marked not found in the paper

## Summary

The paper compares supervised feedforward networks (a standard ANN and an ANN trained with a reward-based loss) with Deep Q-Network (DQN) classifiers for bearing fault diagnosis on the University of Ottawa UORED vibration dataset, split 80/20 into training and test data. Every model is trained on two input representations: 1000-point raw-signal windows with 50% overlap, and vectors of statistical features such as mean, RMS and kurtosis. On raw windows the standard DQN reaches only 33.06% accuracy after close to 11 hours of training, while a timestep-based DQN with a replay buffer reaches 62.57%, near the ANN's 67.45%. On statistical features the DQN reaches 73.33% against 70.00% for the ANN, and 78.33% with a reward matrix tuned by trial and error. The paper does not report repeated runs, seeds, hardware or code.

## Main claims

1. RL models can match the performance of supervised learning models for bearing fault classification under controlled conditions.
**Evidence:**

- [Chunk 0] "models developed in this study can match the performance of"
- [Chunk 0] "traditional supervised learning models under controlled"


2. With statistical feature vectors, the DQN beats the ANN by about 4% on all metrics, and by about 8% with the tailored reward matrix.
**Evidence:**

- [Chunk 4] "increase of about 4% on all evaluation metrics for the DQN"
- [Chunk 4] "with the updated reward mechanism, with about 8% difference"


3. On raw-signal windows the standard DQN performs very poorly despite close to 11 hours of training, while the alternative timestep-based DQN performs as well as the supervised models.
**Evidence:**

- [Chunk 3] "very poorly even with its immense training time, which is close"
- [Chunk 3] "to create a decent policy and performed as well as the SL models."


4. Reward functions make RL more flexible than class weights in supervised models, because a reward or penalty can be attached to any action.
**Evidence:**

- [Chunk 4] "models are more versatile as rewards or penalties can be applied"


5. RL models are computationally more demanding, especially on high-dimensional raw windows.
**Evidence:**

- [Chunk 0] "demands highlight areas for further improvement."
- [Chunk 5] "input, leading to longer training times and lower performance"


## Method

Four model types are compared: a standard feedforward ANN, an ANN with a reward-based loss, a DQN, and an alternative DQN trained over timesteps with a replay buffer. The networks share the architecture of Table II (fully connected input layer to 128 units, a 128-unit hidden layer, an output layer over the classes), except the alternative DQN, which the paper says has a different architecture. Inputs are either raw vibration windows of 1000 points with 50% overlap or statistical feature vectors. Models 1-4 use windows and models 5-8 use feature vectors; model 8 adds a hand-designed reward matrix.

**Evidence:**

- [Chunk 2] "feedforward ANN that represents a standard SL model, a"
- [Chunk 2] "reward-based loss function, and two types of Deep Q-Networks"
- [Chunk 1] "Input Fully Num_features 128 ReLU"
- [Chunk 2] "vibration data is divided into fixed-length windows of 1000 data"
- [Chunk 3] "kurtosis, crest factor, shape factor, impulse factor, margin factor,"
- [Chunk 3] "to 11 hours. Model 4, which has a different architecture, was able"


## Results

- **Accuracy (%), Model 1: ANN, raw windows** = 67.45
Best of the window-based models; precision, recall and F1 all 0.67.
**Evidence:**

- [Chunk 3] "Model 1: ANN 67.45 0.67 0.67 0.67 6m 17.3s"


- **Accuracy (%), Model 2: modified ANN, raw windows** = 66.24
ANN with reward-based training on windows.
**Evidence:**

- [Chunk 3] "66.24 0.64 0.66 0.65 6m 28.7s"


- **Accuracy (%), Model 3: DQN, raw windows** = 33.06
Standard DQN on windows; F1 0.20.
**Evidence:**

- [Chunk 3] "Model 3: DQN 33.06 0.29 0.33 0.20"


- **Training time, Model 3: DQN, raw windows** = close to 11 hours
The text gives the DQN's training time on windows as close to 11 hours.
**Evidence:**

- [Chunk 3] "to 11 hours. Model 4, which has a different architecture, was able"


- **Accuracy (%), Model 4: alternative DQN, raw windows** = 62.57
Timestep-based DQN with a replay buffer, trained for 160,000 timesteps.
**Evidence:**

- [Chunk 3] "Alternative 62.57 0.62 0.64 0.63"
- [Chunk 3] "160,000 timesteps, enables the DQN to sample from diverse"


- **Accuracy (%), Model 5: ANN, statistical features** = 70.00
Supervised baseline on feature vectors.
**Evidence:**

- [Chunk 4] "Model 5: ANN 70.00 0.69 0.70 0.70 2.5s"


- **Accuracy (%), Model 6: modified ANN, statistical features** = 71.67
ANN with reward-based training on feature vectors.
**Evidence:**

- [Chunk 4] "71.67 0.72 0.72 0.72"


- **Accuracy (%), Model 7: DQN, statistical features** = 73.33
Standard DQN on feature vectors.
**Evidence:**

- [Chunk 4] "Model 7: DQN 73.33 0.74 0.73 0.74 5m 2.0s"


- **Accuracy (%), Model 8: DQN with different rewards, statistical features** = 78.33
Best result in the paper, using the reward matrix of Table VI.
**Evidence:**

- [Chunk 4] "with different 78.33 0.79 0.78 0.77"


## Reproducibility checklist

- **Dataset named, described and cited with a DOI** — present
**Evidence:**

- [Chunk 2] "UORED dataset [10], which was split into 80% training and 20%"
- [Chunk 2] "UORED 2023 42 420,000 Natural FAFNIR"
- [Chunk 5] "Apr. 04, 2023. doi: 10.17632/65d7pmfzvx."


- **Train/test split ratio** — present
**Evidence:**

- [Chunk 2] "UORED dataset [10], which was split into 80% training and 20%"


- **How the split was made (random over windows or by recording) and a validation set** — missing
**Evidence:**

- [Chunk ] "Not found in provided text"


- **Window length and overlap** — present
**Evidence:**

- [Chunk 2] "vibration data is divided into fixed-length windows of 1000 data"
- [Chunk 2] "50% overlap between consecutive windows"


- **Statistical features and the segment length they are computed over** — unclear
**Evidence:**

- [Chunk 3] "mean, standard deviation, peak-to-peak, RMS, skewness,"
- [Chunk 3] "each data file or parts of the data files."


- **Network architecture** — present
**Evidence:**

- [Chunk 1] "Input Fully Num_features 128 ReLU"
- [Chunk 1] "Hidden Fully 128 128 ReLU"


- **Optimizer and core hyperparameters** — present
**Evidence:**

- [Chunk 2] "Learning rate 0.001"
- [Chunk 2] "Discount factor 0.99"
- [Chunk 2] "Decay rate 0.995"


- **Training length (epochs, episodes) for each model** — unclear
**Evidence:**

- [Chunk 3] "160,000 timesteps, enables the DQN to sample from diverse"


- **DQN details: replay buffer size, target network, minimum epsilon** — missing
**Evidence:**

- [Chunk ] "Not found in provided text"


- **Reward matrix for the tailored DQN** — present
**Evidence:**

- [Chunk 4] "Developing fault +1 -1.2 -1"


- **Reward definitions for the default DQN and the reward-based ANN loss** — missing
**Evidence:**

- [Chunk ] "Not found in provided text"


- **Random seeds, repeated runs, variance** — missing
**Evidence:**

- [Chunk ] "Not found in provided text"


- **Hardware used for the reported training times** — missing
**Evidence:**

- [Chunk ] "Not found in provided text"


- **Code release** — missing
**Evidence:**

- [Chunk ] "Not found in provided text"


- **Averaging used for precision, recall and F1 (macro or weighted)** — missing
**Evidence:**

- [Chunk ] "Not found in provided text"


## Missing details

- **How the 80/20 split was drawn: randomly over windows or by recording/file** — impact: Windows overlap by 50%, so a random split puts overlapping halves of the same signal in training and test data and can inflate test accuracy. A replication cannot tell which split to use.
**Evidence:**

- [Chunk 2] "UORED dataset [10], which was split into 80% training and 20%"
- [Chunk 2] "50% overlap between consecutive windows"


- **A validation set for choosing hyperparameters and the reward matrix** — impact: Hyperparameters and the reward matrix were found by trial and error with only a train/test split described, so the reported test scores, including the 78.33% headline, may be tuned on the test data.
**Evidence:**

- [Chunk 2] "achieved by trial and error in the pursuit of optimization."
- [Chunk 2] "samples, and it is achieved by trial and error."


- **Number of epochs for the epoch-trained models and episodes for the DQNs** — impact: Training length drives both accuracy and the reported training times; only model 4's 160,000 timesteps are given.
**Evidence:**

- [Chunk ] "Not found in provided text"


- **DQN internals: replay buffer size, target network use and update rate, minimum epsilon, loss** — impact: DQN results are sensitive to these settings; without them the 33.06% and 73.33% DQN results cannot be rebuilt faithfully.
**Evidence:**

- [Chunk ] "Not found in provided text"


- **The reward used by the default DQN and the reward-based loss of the modified ANN** — impact: Models 2, 3, 6 and 7 depend on these definitions, but only the tailored reward matrix of model 8 is given.
**Evidence:**

- [Chunk 2] "modified ANN with reward-based training that incorporates a"


- **Segment length used to compute the statistical feature vectors, and how many vectors result** — impact: Models 5-8, including the best result, are trained on these vectors; the text only says features are computed per file or per part of a file.
**Evidence:**

- [Chunk 3] "each data file or parts of the data files."


- **Which UORED files were used** — impact: The cited Mendeley record is the spectrogram release, while the method works on raw vibration signals, so a reader may download the wrong data.
**Evidence:**

- [Chunk 5] "conditions – Spectrograms (UORED-VAFCLS-P1) - Mendeley Data."
- [Chunk 2] "vibration data is divided into fixed-length windows of 1000 data"


- **Hardware, seeds and number of runs** — impact: Training times cannot be compared across machines, and the 4% and 8% gaps have no variance estimate to show they exceed run-to-run noise.
**Evidence:**

- [Chunk ] "Not found in provided text"


## Risk breakdown

- **dataset defined & access described** (15 pts) — present
  UORED is named, described in Table I and cited with a Mendeley DOI, although the cited record is the spectrogram release.
**Evidence:**

- [Chunk 2] "UORED 2023 42 420,000 Natural FAFNIR"
- [Chunk 5] "Apr. 04, 2023. doi: 10.17632/65d7pmfzvx."


- **train/val/test split described** (15 pts) — unclear
  An 80/20 train/test ratio is given, but not how the split was drawn, and there is no validation set.
**Evidence:**

- [Chunk 2] "UORED dataset [10], which was split into 80% training and 20%"


- **preprocessing/cleaning described** (10 pts) — unclear
  Windowing is fully specified, but the segment length for the statistical features (used by the best models) is not, and normalization is not mentioned.
**Evidence:**

- [Chunk 2] "50% overlap between consecutive windows"
- [Chunk 3] "each data file or parts of the data files."


- **hyperparameters reported** (10 pts) — present
  Table III gives batch size, learning rate, discount factor, epsilon start and decay rate, and optimizer; epochs and replay buffer settings are missing.
**Evidence:**

- [Chunk 2] "Batch size 64"
- [Chunk 2] "could refine these models to perform even better. Optimizer Adam"


- **compute/hardware reported** (10 pts) — missing
  Training times are reported, but not the hardware they were measured on.
**Evidence:**

- [Chunk ] "Not found in provided text"


- **baselines included** (10 pts) — present
  A standard supervised ANN is the baseline for every RL model, on both input types.
**Evidence:**

- [Chunk 2] "feedforward ANN that represents a standard SL model, a"


- **ablations included** (10 pts) — present
  The paper varies the input representation, the DQN training scheme (model 4) and the reward matrix (model 8).
**Evidence:**

- [Chunk 4] "additional test is done to see if there is a performance boost that"
- [Chunk 3] "to 11 hours. Model 4, which has a different architecture, was able"


- **evaluation protocol clear** (10 pts) — unclear
  Accuracy, precision, recall and F1 on the test split are reported, but not the metric averaging, the number of runs, or whether test windows are independent of training windows.
**Evidence:**

- [Chunk 4] "Model 7: DQN 73.33 0.74 0.73 0.74 5m 2.0s"


- **limitations discussed** (10 pts) — unclear
  Computational cost and possibly suboptimal hyperparameters are acknowledged; the single dataset, single split and missing variance are not discussed.
**Evidence:**

- [Chunk 0] "demands highlight areas for further improvement."
- [Chunk 2] "These hyperparameters may not be optimal, but they are"


### Additional reasons
- Everything is evaluated on one dataset and one train/test split; the method section mentions cross-domain evaluations, but no cross-domain results are reported.
- The best result uses a reward matrix tailored to the DQN's performance on the same dataset, chosen by trial and error.
## Suggested experiments

### Recording-level split
Check whether overlapping windows leak between training and test data.
- Split UORED by recording or bearing so that no recording contributes windows to both sets.
- Retrain models 1-4 on windows and compare accuracy with Table V.
### Seeds and variance
Find out whether the 4% and 8% DQN gains exceed run-to-run noise.
- Train models 5-8 with five seeds each.
- Report mean and standard deviation of accuracy and F1, and a significance test for DQN against ANN.
### Reward matrix against class weights
Test whether the tailored reward helps beyond what a class-weighted supervised loss gives.
- Hold out a validation split and tune the reward matrix and ANN class weights on it.
- Compare model 8 with an ANN trained with the tuned class weights on the untouched test split.
### Cross-domain test
Measure the adaptability the paper attributes to RL.
- Train on one UORED operating condition or bearing and test on another.
- Compare the drop in accuracy for the ANN and the DQN.
## Citations

- [Chunk 2] "UORED dataset [10], which was split into 80% training and 20%"
- [Chunk 2] "50% overlap between consecutive windows"
- [Chunk 4] "with different 78.33 0.79 0.78 0.77"
- [Chunk 4] "increase of about 4% on all evaluation metrics for the DQN"
- [Chunk 4] "performance of the DQN with the UORED dataset outperformed"
- [Chunk 3] "Model 3: DQN 33.06 0.29 0.33 0.20"
