# Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting

**Authors**: Haoyi Zhou<sup>1</sup>, Shanghang Zhang<sup>2</sup>, Jieqi Peng<sup>1</sup>, Shuai Zhang<sup>1</sup>, Jianxin Li<sup>1</sup>, Hui Xiong<sup>3</sup>, Wancai Zhang<sup>4</sup>  
**Affiliations**:  
<sup>1</sup> SCSE and BDBC, Beihang University, Beijing, China  
<sup>2</sup> UC Berkeley, California, US  
<sup>3</sup> Rutgers University, New Jersey, US  
<sup>4</sup> Beijing Guowang Fuda Science & Technology Development Company  
**Emails**: {zhouhy, pengjq, zhangs, lijx}@act.buaa.edu.cn, shz@eecs.berkeley.edu, {xionghui,zhangwancaibuaa}@gmail.com  
**Conference**: The Thirty-Fifth AAAI Conference on Artificial Intelligence (AAAI-21)  
**Copyright**: © 2021, Association for the Advancement of Artificial Intelligence (www.aaai.org). All rights reserved.

## Abstract

Many real-world applications require the prediction of long sequence time-series, such as electricity consumption planning. Long sequence time-series forecasting (LSTF) demands a high prediction capacity of the model, which is the ability to capture precise long-range dependency coupling between output and input efficiently. Recent studies have shown the potential of Transformer to increase the prediction capacity. However, there are several severe issues with Transformer that prevent it from being directly applicable to LSTF, including quadratic time complexity, high memory usage, and inherent limitation of the encoder-decoder architecture. To address these issues, we design an efficient transformer-based model for LSTF, named Informer, with three distinctive characteristics: (i) a ProbSparse self-attention mechanism, which achieves O(L log L) in time complexity and memory usage, and has comparable performance on sequences’ dependency alignment. (ii) the self-attention distilling highlights dominating attention by halving cascading layer input, and efficiently handles extreme long input sequences. (iii) the generative style decoder, while conceptually simple, predicts the long time-series sequences at one forward operation rather than a step-by-step way, which drastically improves the inference speed of long-sequence predictions. Extensive experiments on four large-scale datasets demonstrate that Informer significantly outperforms existing methods and provides a new solution to the LSTF problem.

## Introduction

Time-series forecasting is a critical ingredient across many domains, such as sensor network monitoring (Papadimitriou and Yu 2006), energy and smart grid management, economics and finance (Zhu and Shasha 2002), and disease propagation analysis (Matsubara et al. 2014). In these scenarios, we can leverage a substantial amount of time-series data on past behavior to make a forecast in the long run, namely long sequence time-series forecasting (LSTF). However, existing methods are mostly designed under short-term problem setting, like predicting 48 points or less (Hochreiter and Schmidhuber 1997; Li et al. 2018; Yu et al. 2017; Liu et al. 2019; Qin et al. 2017; Wen et al. 2017). The increasingly long sequences strain the models’ prediction capacity to the point where this trend is holding the research on LSTF. As an empirical example, Fig. 1 shows the forecasting results on a real dataset, where the LSTM network predicts the hourly temperature of an electrical transformer station from the short-term period (12 points, 0.5 days) to the long-term period (480 points, 20 days). The overall performance gap is substantial when the prediction length is greater than 48 points (the solid star in Fig. 1b), where the MSE rises to unsatisfactory performance, the inference speed gets sharp drop, and the LSTM model starts to fail.

The major challenge for LSTF is to enhance the prediction capacity to meet the increasingly long sequence demand, which requires (a) extraordinary long-range alignment ability and (b) efficient operations on long sequence inputs and outputs. Recently, Transformer models have shown superior performance in capturing long-range dependency than RNN models. The self-attention mechanism can reduce the maximum length of network signals traveling paths into the theoretical shortest O(1) and avoid the recurrent structure, whereby Transformer shows great potential for the LSTF problem. Nevertheless, the self-attention mechanism violates requirement (b) due to its L-quadratic computation and memory consumption on L-length inputs/outputs. Some large-scale Transformer models pour resources and yield impressive results on NLP tasks (Brown et al. 2020), but the training on dozens of GPUs and expensive deploying cost make these models unaffordable on real-world LSTF problem. The efficiency of the self-attention mechanism and Transformer architecture becomes the bottleneck of applying them to LSTF problems. Thus, in this paper, we seek to answer the question: can we improve Transformer models to be computation, memory, and architecture efficient, as well as maintaining higher prediction capacity?

Vanilla Transformer (Vaswani et al. 2017) has three significant limitations when solving the LSTF problem:

1. **The quadratic computation of self-attention**. The atom operation of self-attention mechanism, namely canonical dot-product, causes the time complexity and memory usage per layer to be O(L²).
2. **The memory bottleneck in stacking layers for long inputs**. The stack of J encoder/decoder layers makes total memory usage to be O(J · L²), which limits the model scalability in receiving long sequence inputs.
3. **The speed plunge in predicting long outputs**. Dynamic decoding of vanilla Transformer makes the step-by-step inference as slow as RNN-based model (Fig. 1b).

There are some prior works on improving the efficiency of self-attention. The Sparse Transformer (Child et al. 2019), LogSparse Transformer (Li et al. 2019), and Longformer (Beltagy, Peters, and Cohan 2020) all use a heuristic method to tackle limitation 1 and reduce the complexity of self-attention mechanism to O(L log L), where their efficiency gain is limited (Qiu et al. 2019). Reformer (Kitaev, Kaiser, and Levskaya 2019) also achieves O(L log L) with locally-sensitive hashing self-attention, but it only works on extremely long sequences. More recently, Linformer (Wang et al. 2020) claims a linear complexity O(L), but the project matrix cannot be fixed for real-world long sequence input, which may have the risk of degradation to O(L²). Transformer-XL (Dai et al. 2019) and Compressive Transformer (Rae et al. 2019) use auxiliary hidden states to capture long-range dependency, which could amplify limitation 1 and be adverse to break the efficiency bottleneck. All these works mainly focus on limitation 1, and the limitations 2 and 3 remain unsolved in the LSTF problem. To enhance the prediction capacity, we tackle all these limitations and achieve improvement beyond efficiency in the proposed Informer.

To this end, our work delves explicitly into these three issues. We investigate the sparsity in the self-attention mechanism, make improvements of network components, and conduct extensive experiments. The contributions of this paper are summarized as follows:

- We propose Informer to successfully enhance the prediction capacity in the LSTF problem, which validates the Transformer-like model’s potential value to capture individual long-range dependency between long sequence time-series outputs and inputs.
- We propose ProbSparse self-attention mechanism to efficiently replace the canonical self-attention. It achieves the O(L log L) time complexity and O(L log L) memory usage on dependency alignments.
- We propose self-attention distilling operation to privilege dominating attention scores in J-stacking layers and sharply reduce the total space complexity to be O((2 − ε)L log L), which helps receiving long sequence input.
- We propose generative style decoder to acquire long sequence output with only one forward step needed, simultaneously avoiding cumulative error spreading during the inference phase.

## Preliminary

We first provide the LSTF problem definition. Under the rolling forecasting setting with a fixed size window, we have the input \( X^t = \{ x_1^t, \ldots, x_{L_x}^t \mid x_i^t \in \mathbb{R}^{d_x} \} \) at time \( t \), and the output is to predict corresponding sequence \( Y^t = \{ y_1^t, \ldots, y_{L_y}^t \mid y_i^t \in \mathbb{R}^{d_y} \} \). The LSTF problem encourages a longer output’s length \( L_y \) than previous works (Cho et al. 2014; Sutskever, Vinyals, and Le 2014) and the feature dimension is not limited to univariate case (\( d_y \geq 1 \)).

### Encoder-decoder architecture

Many popular models are devised to “encode” the input representations \( X^t \) into a hidden state representations \( H^t \) and “decode” an output representations \( Y^t \) from \( H^t = \{ h_1^t, \ldots, h_{L_h}^t \} \). The inference involves a step-by-step process named “dynamic decoding”, where the decoder computes a new hidden state \( h_{k+1}^t \) from the previous state \( h_k^t \) and other necessary outputs from \( k \)-th step then predict the \( (k + 1) \)-th sequence \( y_{k+1}^t \).

### Input Representation

A uniform input representation is given to enhance the global positional context and local temporal context of the time-series inputs. To avoid trivializing description, we put the details in Appendix B.

## Methodology

Existing methods for time-series forecasting can be roughly grouped into two categories. Classical time-series models serve as a reliable workhorse for time-series forecasting (Box et al. 2015; Ray 1990; Seeger et al. 2017; Seeger, Salinas, and Flunkert 2016), and deep learning techniques mainly develop an encoder-decoder prediction paradigm by using RNN and their variants (Hochreiter and Schmidhuber 1997; Li et al. 2018; Yu et al. 2017). Our proposed Informer holds the encoder-decoder architecture while targeting the LSTF problem. Please refer to Fig. 2 for an overview and the following sections for details.

### Efficient Self-attention Mechanism

The canonical self-attention in (Vaswani et al. 2017) is defined based on the tuple inputs, i.e., query, key, and value, which performs the scaled dot-product as:

\[
A(Q, K, V) = \text{Softmax}\left( \frac{QK^\top}{\sqrt{d}} \right)V
\]

where \( Q \in \mathbb{R}^{L_Q \times d} \), \( K \in \mathbb{R}^{L_K \times d} \), \( V \in \mathbb{R}^{L_V \times d} \), and \( d \) is the input dimension. To further discuss the self-attention mechanism, let \( q_i \), \( k_i \), \( v_i \) stand for the \( i \)-th row in \( Q \), \( K \), \( V \) respectively. Following the formulation in (Tsai et al. 2019), the \( i \)-th query’s attention is defined as a kernel smoother in a probability form:

\[
A(q_i, K, V) = \sum_{j=1}^{L_K} p(k_j | q_i) v_j = \mathbb{E}_{p(k_j | q_i)}[v_j]
\]

where \( p(k_j | q_i) = \frac{k(q_i, k_j)}{\sum_{l=1}^{L_K} k(q_i, k_l)} \) and \( k(q_i, k_j) = \exp\left( \frac{q_i k_j^\top}{\sqrt{d}} \right) \). The self-attention combines the values and acquires outputs based on computing the probability \( p(k_j | q_i) \). It requires the quadratic times dot-product computation and \( O(L_Q L_K) \) memory usage, which is the major drawback when enhancing prediction capacity.

Some previous attempts have revealed that the distribution of self-attention probability has potential sparsity, and they have designed “selective” counting strategies on all \( p(k_j | q_i) \) without significantly affecting the performance. The Sparse Transformer (Child et al. 2019) incorporates both the row outputs and column inputs, in which the sparsity arises from the separated spatial correlation. The LogSparse Transformer (Li et al. 2019) notices the cyclical pattern in self-attention and forces each cell to attend to its previous one by an exponential step size. The Longformer (Beltagy, Peters, and Cohan 2020) extends previous two works to more complicated sparse configuration. However, they are limited to theoretical analysis from following heuristic methods and tackle each multi-head self-attention with the same strategy, which narrows their further improvement.

To motivate our approach, we first perform a qualitative assessment on the learned attention patterns of the canonical self-attention. The “sparsity” self-attention score forms a long tail distribution (see Appendix C for details), i.e., a few dot-product pairs contribute to the major attention, and others generate trivial attention. Then, the next question is how to distinguish them?

### Query Sparsity Measurement

From Eq. (1), the \( i \)-th query’s attention on all the keys are defined as a probability \( p(k_j | q_i) \) and the output is its composition with values \( v \). The dominant dot-product pairs encourage the corresponding query’s attention probability distribution away from the uniform distribution. If \( p(k_j | q_i) \) is close to a uniform distribution \( q(k_j | q_i) = \frac{1}{L_K} \), the self-attention becomes a trivial sum of values \( V \) and is redundant to the residential input. Naturally, the “likeness” between distribution \( p \) and \( q \) can be used to distinguish the “important” queries. We measure the “likeness” through Kullback-Leibler divergence:

\[
KL(q || p) = \ln \sum_{l=1}^{L_K} e^{\frac{q_i k_l^\top}{\sqrt{d}}} - \frac{1}{L_K} \sum_{j=1}^{L_K} \frac{q_i k_j^\top}{\sqrt{d}} - \ln L_K
\]

Dropping the constant, we define the \( i \)-th query’s sparsity measurement as:

\[
M(q_i, K) = \ln \sum_{j=1}^{L_K} e^{\frac{q_i k_j^\top}{\sqrt{d}}} - \frac{1}{L_K} \sum_{j=1}^{L_K} \frac{q_i k_j^\top}{\sqrt{d}}
\]

where the first term is the Log-Sum-Exp (LSE) of \( q_i \) on all the keys, and the second term is the arithmetic mean on them. If the \( i \)-th query gains a larger \( M(q_i, K) \), its attention probability \( p \) is more “diverse” and has a high chance to contain the dominate dot-product pairs in the header field of the long tail self-attention distribution.

### ProbSparse Self-attention

Based on the proposed measurement, we have the ProbSparse self-attention by allowing each key to only attend to the \( u \) dominant queries:

\[
A(Q, K, V) = \text{Softmax}\left( \frac{\bar{Q} K^\top}{\sqrt{d}} \right)V
\]

where \( \bar{Q} \) is a sparse matrix of the same size of \( Q \) and it only contains the Top-\( u \) queries under the sparsity measurement \( M(q, K) \). Controlled by a constant sampling factor \( c \), we set \( u = c \cdot \ln L_Q \), which makes the ProbSparse self-attention only need to calculate \( O(\ln L_Q) \) dot-product for each query-key lookup and the layer memory usage maintains \( O(L_K \ln L_Q) \). Under the multi-head perspective, this attention generates different sparse query-key pairs for each head, which avoids severe information loss in return.

However, the traversing of all the queries for the measurement \( M(q_i, K) \) requires calculating each dot-product pairs, i.e., quadratically \( O(L_Q L_K) \), besides the LSE operation has the potential numerical stability issue. Motivated by this, we propose an empirical approximation for the efficient acquisition of the query sparsity measurement.

**Lemma 1**. For each query \( q_i \in \mathbb{R}^d \) and \( k_j \in \mathbb{R}^d \) in the keys set \( K \), we have the bound as:

\[
\ln L_K \leq M(q_i, K) \leq \max_j \left\{ \frac{q_i k_j^\top}{\sqrt{d}} \right\} - \frac{1}{L_K} \sum_{j=1}^{L_K} \frac{q_i k_j^\top}{\sqrt{d}} + \ln L_K
\]

When \( q_i \in K \), it also holds. (Proof is given in Appendix D.1)

From Lemma 1, we propose the max-mean measurement as:

\[
M(q_i, K) = \max_j \left\{ \frac{q_i k_j^\top}{\sqrt{d}} \right\} - \frac{1}{L_K} \sum_{j=1}^{L_K} \frac{q_i k_j^\top}{\sqrt{d}}
\]

The range of Top-\( u \) approximately holds in the boundary relaxation with Proposition 1 (refers in Appendix D.2). Under the long tail distribution, we only need to randomly sample \( U = L_K \ln L_Q \) dot-product pairs to calculate the \( M(q_i, K) \), i.e., filling other pairs with zero. Then, we select sparse Top-\( u \) from them as \( \bar{Q} \). The max-operator in \( M(q_i, K) \) is less sensitive to zero values and is numerical stable. In practice, the input length of queries and keys are typically equivalent in the self-attention computation, i.e., \( L_Q = L_K = L \) such that the total ProbSparse self-attention time complexity and space complexity are \( O(L \ln L) \).

### Encoder: Allowing for Processing Longer Sequential Inputs under the Memory Usage Limitation

The encoder is designed to extract the robust long-range dependency of the long sequential inputs. After the input representation, the \( t \)-th sequence input \( X^t \) has been shaped into a matrix \( X_{en}^t \in \mathbb{R}^{L_x \times d_{\text{model}}} \). We give a sketch of the encoder in Fig. 3 for clarity.

#### Self-attention Distilling

As the natural consequence of the ProbSparse self-attention mechanism, the encoder’s feature map has redundant combinations of value \( V \). We use the distilling operation to privilege the superior ones with dominating features and make a focused self-attention feature map in the next layer. It trims the input’s time dimension sharply, seeing the n-heads weights matrix (overlapping red squares) of Attention blocks in Fig. 3. Inspired by the dilated convolution (Yu, Koltun, and Funkhouser 2017; Gupta and Rush 2017), our “distilling” procedure forwards from \( j \)-th layer into \( (j + 1) \)-th layer as:

\[
X_{j+1}^t = \text{MaxPool} \left( \text{ELU} \left( \text{Conv1d} \left( [X_j^t]_{AB} \right) \right) \right)
\]

where \( [\cdot]_{AB} \) represents the attention block. It contains the Multi-head ProbSparse self-attention and the essential operations, where \( \text{Conv1d}(\cdot) \) performs an 1-D convolutional filters (kernel width=3) on time dimension with the \( \text{ELU}(\cdot) \) activation function (Clevert, Unterthiner, and Hochreiter 2016). We add a max-pooling layer with stride 2 and down-sample \( X^t \) into its half slice after stacking a layer, which reduces the whole memory usage to be \( O((2 − \epsilon)L \log L) \), where \( \epsilon \) is a small number. To enhance the robustness of the distilling operation, we build replicas of the main stack with halving inputs, and progressively decrease the number of self-attention distilling layers by dropping one layer at a time, like a pyramid in Fig. 2, such that their output dimension is aligned. Thus, we concatenate all the stacks’ outputs and have the final hidden representation of encoder.

### Decoder: Generating Long Sequential Outputs Through One Forward Procedure

We use a standard decoder structure (Vaswani et al. 2017) in Fig. 2, and it is composed of a stack of two identical multi-head attention layers. However, the generative inference is employed to alleviate the speed plunge in long prediction. We feed the decoder with the following vectors as:

\[
X_{de}^t = \text{Concat}(X_{\text{token}}, X_0^t) \in \mathbb{R}^{(L_{\text{token}} + L_y) \times d_{\text{model}}}
\]

where \( X_{\text{token}} \in \mathbb{R}^{L_{\text{token}} \times d_{\text{model}}} \) is the start token, \( X_0^t \in \mathbb{R}^{L_y \times d_{\text{model}}} \) is a placeholder for the target sequence (set scalar as 0). Masked multi-head attention is applied in the ProbSparse self-attention computing by setting masked dot-products to \(-\infty\). It prevents each position from attending to coming positions, which avoids auto-regressive. A fully connected layer acquires the final output, and its outsize \( d_y \) depends on whether we are performing a univariate forecasting or a multivariate one.

#### Generative Inference

Start token is efficiently applied in NLP’s “dynamic decoding” (Devlin et al. 2018), and we extend it into a generative way. Instead of choosing specific flags as the token, we sample a \( L_{\text{token}} \) long sequence in the input sequence, such as an earlier slice before the output sequence. Take predicting 168 points as an example (7-day temperature prediction in the experiment section), we will take the known 5 days before the target sequence as “start-token”, and feed the generative-style inference decoder with \( X_{de} = \{ X_{5d}, X_0 \} \). The \( X_0 \) contains target sequence’s time stamp, i.e., the context at the target week. Then our proposed decoder predicts outputs by one forward procedure rather than the time-consuming “dynamic decoding” in the conventional encoder-decoder architecture. A detailed performance comparison is given in the computation efficiency section.

#### Loss function

We choose the MSE loss function on prediction w.r.t the target sequences, and the loss is propagated back from the decoder’s outputs across the entire model.

## Experiment

We extensively perform experiments on four datasets, including 2 collected real-world datasets for LSTF and 2 public benchmark datasets.

- **ETT (Electricity Transformer Temperature)**: The ETT is a crucial indicator in the electric power long-term deployment. We collected 2-year data from two separated counties in China. To explore the granularity on the LSTF problem, we create separate datasets as \{ETTh1, ETTh2\} for 1-hour-level and ETTm1 for 15-minute-level. Each data point consists of the target value “oil temperature” and 6 power load features. The train/val/test is 12/4/4 months.
- **ECL (Electricity Consuming Load)**: It collects the electricity consumption (Kwh) of 321 clients. Due to the missing data (Li et al. 2019), we convert the dataset into hourly consumption of 2 years and set ‘MT_320’ as the target value. The train/val/test is 15/3/4 months.
- **Weather**: This dataset contains local climatological data for nearly 1,600 U.S. locations, 4 years from 2010 to 2013, where data points are collected every 1 hour. Each data point consists of the target value “wet bulb” and 11 climate features. The train/val/test is 28/10/10 months.

### Experimental Details

We briefly summarize basics, and more information on network components and setups are given in Appendix E.

**Baselines**: We have selected five time-series forecasting methods as comparison, including ARIMA (Ariyo, Adewumi, and Ayo 2014), Prophet (Taylor and Letham 2018), LSTMa (Bahdanau, Cho, and Bengio 2015), LSTnet (Lai et al. 2018), and DeepAR (Flunkert, Salinas, and Gasthaus 2017). To better explore the ProbSparse self-attention’s performance in our proposed Informer, we incorporate the canonical self-attention variant (Informer†), the efficient variant Reformer (Kitaev, Kaiser, and Levskaya 2019), and the most related work LogSparse self-attention (Li et al. 2019) in the experiments. The details of network components are given in Appendix E.1.

**Hyper-parameter tuning**: We conduct grid search over the hyper-parameters, and detailed ranges are given in Appendix E.3. Informer contains a 3-layer stack and a 1-layer stack (1/4 input) in the encoder, and a 2-layer decoder. Our proposed methods are optimized with Adam optimizer, and its learning rate starts from 1e−4, decaying 0.5 times smaller every epoch. The total number of epochs is 8 with proper early stopping. We set the comparison methods as recommended, and the batch size is 32.

**Setup**: The input of each dataset is zero-mean normalized. Under the LSTF settings, we prolong the prediction windows size \( L_y \) progressively, i.e., \{1d, 2d, 7d, 14d, 30d, 40d\} in \{ETTh, ECL, Weather\}, \{6h, 12h, 24h, 72h, 168h\} in ETTm1. **Metrics**: We use two evaluation metrics, including MSE = \( \frac{1}{n} \sum_{i=1}^n (y - \hat{y})^2 \) and MAE = \( \frac{1}{n} \sum_{i=1}^n |y - \hat{y}| \) on each prediction window (averaging for multivariate prediction), and roll the whole set with stride = 1. **Platform**: All the models were trained/tested on a single Nvidia V100 32GB GPU. The source code is available at [https://github.com/zhouhaoyi/Informer2020](https://github.com/zhouhaoyi/Informer2020).

## Results and Analysis

Table 1 and Table 2 summarize the univariate/multivariate evaluation results of all the methods on 4 datasets. We gradually prolong the prediction horizon as a higher requirement of prediction capacity, where the LSTF problem setting is precisely controlled to be tractable on one single GPU for each method. The best results are highlighted in **bold**.

### Univariate Time-series Forecasting

Under this setting, each method attains predictions as a single variable over time series. From Table 1, we can observe that:

1. The proposed model Informer significantly improves the inference performance (winning-counts in the last column) across all datasets, and their predict error rises smoothly and slowly within the growing prediction horizon, which demonstrates the success of Informer in enhancing the prediction capacity in the LSTF problem.
2. The Informer beats its canonical degradation Informer† mostly in winning-counts, i.e., 32>12, which supports the query sparsity assumption in providing a comparable attention feature map. Our proposed method also outperforms the most related work LogTrans and Reformer. We note that the Reformer keeps dynamic decoding and performs poorly in LSTF, while other methods benefit from the generative style decoder as nonautoregressive predictors.
3. The Informer model shows significantly better results than recurrent neural networks LSTMa. Our method has a MSE decrease of 26.8% (at 168), 52.4% (at 336), and 60.1% (at 720). This reveals a shorter network path in the self-attention mechanism acquires better prediction capacity than the RNN-based models.
4. The proposed method outperforms DeepAR, ARIMA, and Prophet on MSE by decreasing 49.3% (at 168), 61.1% (at 336), and 65.1% (at 720) in average. On the ECL dataset, DeepAR performs better on shorter horizons (≤ 336), and our method surpasses on longer horizons. We attribute this to a specific example, in which the effectiveness of prediction capacity is reflected with the problem scalability.

#### Table 1: Univariate Long Sequence Time-series Forecasting Results

| **Dataset** | **Metric** | **Prediction Length** | **Informer** | **Informer†** | **LogTrans** | **Reformer** | **LSTMa** | **DeepAR** | **ARIMA** | **Prophet** |
|-------------|------------|-----------------------|--------------|---------------|--------------|--------------|------------|------------|-----------|-------------|
| **ECL**     | MSE        | 24                    | **0.098**    | 0.099         | 0.103        | 0.222        | 0.114      | 0.107      | 0.108     | 0.115       |
|             |            | 48                    | **0.158**    | 0.159         | 0.167        | 0.284        | 0.193      | 0.162      | 0.175     | 0.168       |
|             |            | 168                   | **0.183**    | 0.235         | 0.207        | 1.522        | 0.236      | 0.239      | 0.396     | 1.224       |
|             |            | 336                   | **0.222**    | 0.258         | 0.230        | 1.860        | 0.590      | 0.445      | 0.468     | 1.549       |
|             |            | 720                   | **0.269**    | 0.285         | 0.273        | 2.112        | 0.683      | 0.658      | 0.659     | 2.735       |
|             | MAE        | 24                    | **0.247**    | 0.241         | 0.259        | 0.389        | 0.272      | 0.280      | 0.284     | 0.275       |
|             |            | 48                    | **0.319**    | 0.317         | 0.328        | 0.445        | 0.358      | 0.327      | 0.424     | 0.330       |
|             |            | 168                   | **0.346**    | 0.390         | 0.375        | 1.191        | 0.392      | 0.422      | 0.504     | 0.763       |
|             |            | 336                   | **0.387**    | 0.423         | 0.398        | 1.124        | 0.698      | 0.552      | 0.593     | 1.820       |
|             |            | 720                   | **0.435**    | 0.442         | 0.463        | 1.436        | 0.768      | 0.707      | 0.766     | 3.253       |
| **Weather** | MSE        | 24                    | **0.030**    | 0.034         | 0.065        | 0.095        | 0.121      | 0.091      | 0.090     | 0.120       |
|             |            | 48                    | **0.069**    | 0.066         | 0.078        | 0.249        | 0.305      | 0.219      | 0.179     | 0.133       |
|             |            | 96                    | **0.194**    | 0.187         | 0.199        | 0.920        | 0.287      | 0.364      | 0.272     | 0.194       |
|             |            | 288                   | **0.401**    | 0.409         | 0.411        | 1.108        | 0.524      | 0.948      | 0.462     | 0.452       |
|             |            | 672                   | **0.512**    | 0.519         | 0.598        | 1.793        | 1.064      | 2.437      | 0.639     | 2.747       |
|             | MAE        | 24                    | **0.137**    | 0.160         | 0.202        | 0.228        | 0.233      | 0.243      | 0.206     | 0.290       |
|             |            | 48                    | **0.203**    | 0.194         | 0.220        | 0.390        | 0.411      | 0.362      | 0.306     | 0.305       |
|             |            | 96                    | **0.372**    | 0.384         | 0.386        | 0.767        | 0.420      | 0.496      | 0.399     | 0.396       |
|             |            | 288                   | **0.554**    | 0.548         | 0.572        | 1.245        | 0.584      | 0.795      | 0.558     | 0.574       |
|             |            | 672                   | **0.644**    | 0.665         | 0.702        | 1.528        | 0.873      | 1.352      | 0.697     | 1.174       |
| **ETTm1**   | MSE        | 24                    | **0.117**    | 0.119         | 0.136        | 0.231        | 0.131      | 0.128      | 0.219     | 0.302       |
|             |            | 48                    | **0.178**    | 0.185         | 0.206        | 0.328        | 0.190      | 0.203      | 0.273     | 0.445       |
|             |            | 168                   | **0.266**    | 0.269         | 0.309        | 0.654        | 0.341      | 0.293      | 0.503     | 2.441       |
|             |            | 336                   | **0.297**    | 0.310         | 0.359        | 1.792        | 0.456      | 0.585      | 0.728     | 1.987       |
|             |            | 720                   | **0.359**    | 0.361         | 0.388        | 2.087        | 0.866      | 0.499      | 1.062     | 3.859       |
|             | MAE        | 24                    | **0.251**    | 0.256         | 0.279        | 0.401        | 0.254      | 0.274      | 0.355     | 0.433       |
|             |            | 48                    | **0.318**    | 0.316         | 0.356        | 0.423        | 0.334      | 0.353      | 0.409     | 0.536       |
|             |            | 168                   | **0.398**    | 0.404         | 0.439        | 0.634        | 0.448      | 0.451      | 0.599     | 1.142       |
|             |            | 336                   | **0.416**    | 0.422         | 0.484        | 1.093        | 0.554      | 0.644      | 0.730     | 2.468       |
|             |            | 720                   | **0.466**    | 0.471         | 0.499        | 1.534        | 0.809      | 0.596      | 0.943     | 1.144       |
| **ETTh1**   | MSE        | 48                    | **0.239**    | 0.238         | 0.280        | 0.971        | 0.493      | 0.204      | 0.879     | 0.524       |
|             |            | 168                   | **0.447**    | 0.442         | 0.454        | 1.671        | 0.723      | 0.315      | 1.032     | 2.725       |
|             |            | 336                   | **0.489**    | 0.501         | 0.514        | 3.528        | 1.212      | 0.414      | 1.136     | 2.246       |
|             |            | 720                   | **0.540**    | 0.543         | 0.558        | 4.891        | 1.511      | 0.563      | 1.251     | 4.243       |
|             |            | 960                   | **0.582**    | 0.594         | 0.624        | 7.019        | 1.545      | 0.657      | 1.370     | 6.901       |
|             | MAE        | 48                    | **0.359**    | 0.368         | 0.429        | 0.884        | 0.539      | 0.357      | 0.764     | 0.595       |
|             |            | 168                   | **0.503**    | 0.514         | 0.529        | 1.587        | 0.655      | 0.436      | 0.833     | 1.273       |
|             |            | 336                   | **0.528**    | 0.552         | 0.563        | 2.196        | 0.898      | 0.519      | 0.876     | 3.077       |
|             |            | 720                   | **0.571**    | 0.578         | 0.609        | 4.047        | 0.966      | 0.595      | 0.933     | 1.415       |
|             |            | 960                   | **0.608**    | 0.638         | 0.645        | 5.105        | 1.006      | 0.683      | 0.982     | 4.264       |
| **Count**   |            |                       | **32**       | 12            | 0            | 0            | 0          | 0          | 0         | 0           |

### Multivariate Time-series Forecasting

Within this setting, some univariate methods are inappropriate, and LSTnet is the state-of-the-art baseline. On the contrary, our proposed Informer is easy to change from univariate prediction to multivariate one by adjusting the final FCN layer. From Table 2, we observe that:

1. The proposed model Informer greatly outperforms other methods and the findings 1 & 2 in the univariate settings still hold for the multivariate time-series.
2. The Informer model shows better results than RNN-based LSTMa and CNN-based LSTnet, and the MSE decreases 26.6% (at 168), 28.2% (at 336), 34.3% (at 720) in average. Compared with the univariate results, the overwhelming performance is reduced, and such phenomena can be caused by the anisotropy of feature dimensions’ prediction capacity. It is beyond the scope of this paper, and we will explore it in the future work.

#### Table 2: Multivariate Long Sequence Time-series Forecasting Results

| **Dataset** | **Metric** | **Prediction Length** | **Informer** | **Informer†** | **LogTrans** | **Reformer** | **LSTMa** | **LSTnet** |
|-------------|------------|-----------------------|--------------|---------------|--------------|--------------|------------|------------|
| **ECL**     | MSE        | 24                    | **0.577**    | 0.620         | 0.686        | 0.991        | 0.650      | 1.293      |
|             |            | 48                    | **0.685**    | 0.692         | 0.766        | 1.313        | 0.702      | 1.456      |
|             |            | 168                   | **0.931**    | 0.947         | 1.002        | 1.824        | 1.212      | 1.997      |
|             |            | 336                   | 1.128        | **1.094**     | 1.362        | 2.117        | 1.424      | 2.655      |
|             |            | 720                   | **1.215**    | 1.241         | 1.397        | 2.415        | 1.960      | 2.143      |
|             | MAE        | 24                    | **0.549**    | 0.577         | 0.604        | 0.754        | 0.624      | 0.901      |
|             |            | 48                    | **0.625**    | 0.671         | 0.757        | 0.906        | 0.675      | 0.960      |
|             |            | 168                   | **0.752**    | 0.797         | 0.846        | 1.138        | 0.867      | 1.214      |
|             |            | 336                   | 0.873        | **0.813**     | 0.952        | 1.280        | 0.994      | 1.369      |
|             |            | 720                   | **0.896**    | 0.917         | 1.291        | 1.520        | 1.322      | 1.380      |
| **Weather** | MSE        | 24                    | 0.720        | **0.753**     | 0.828        | 1.531        | 1.143      | 2.742      |
|             |            | 48                    | **1.457**    | 1.461         | 1.806        | 1.871        | 1.671      | 3.567      |
|             |            | 168                   | 3.489        | **3.485**     | 4.070        | 4.660        | 4.117      | 3.242      |
|             |            | 336                   | 2.723        | **2.626**     | 3.875        | 4.028        | 3.434      | 2.544      |
|             |            | 720                   | **3.467**    | 3.548         | 3.913        | 5.381        | 3.963      | 4.625      |
|             | MAE        | 24                    | **0.665**    | 0.727         | 0.750        | 1.613        | 0.813      | 1.457      |
|             |            | 48                    | **1.001**    | 1.077         | 1.034        | 1.735        | 1.221      | 1.687      |
|             |            | 168                   | **1.515**    | 1.612         | 1.681        | 1.846        | 1.674      | 2.513      |
|             |            | 336                   | **1.340**    | 1.285         | 1.763        | 1.688        | 1.549      | 2.591      |
|             |            | 720                   | **1.473**    | 1.495         | 1.552        | 2.015        | 1.788      | 3.709      |
| **ETTm1**   | MSE        | 24                    | **0.323**    | 0.306         | 0.419        | 0.724        | 0.621      | 1.968      |
|             |            | 48                    | **0.494**    | 0.465         | 0.507        | 1.098        | 1.392      | 1.999      |
|             |            | 96                    | **0.678**    | 0.681         | 0.768        | 1.433        | 1.339      | 2.762      |
|             |            | 288                   | **1.056**    | 1.162         | 1.462        | 1.820        | 1.740      | 1.257      |
|             |            | 672                   | **1.192**    | 1.231         | 1.669        | 2.187        | 2.736      | 1.917      |
|             | MAE        | 24                    | **0.369**    | 0.371         | 0.412        | 0.607        | 0.629      | 1.170      |
|             |            | 48                    | 0.503        | **0.470**     | 0.583        | 0.777        | 0.939      | 1.215      |
|             |            | 96                    | **0.614**    | 0.612         | 0.792        | 0.945        | 0.913      | 1.542      |
|             |            | 288                   | **0.786**    | 0.879         | 1.320        | 1.094        | 1.124      | 2.076      |
|             |            | 672                   | **0.926**    | 1.103         | 1.461        | 1.232        | 1.555      | 2.941      |
| **ETTh2**   | MSE        | 24                    | **0.335**    | 0.349         | 0.435        | 0.655        | 0.546      | 0.615      |
|             |            | 48                    | 0.395        | **0.386**     | 0.426        | 0.729        | 0.829      | 0.660      |
|             |            | 168                   | **0.608**    | 0.613         | 0.727        | 1.318        | 1.038      | 0.748      |
|             |            | 336                   | **0.702**    | 0.707         | 0.754        | 1.930        | 1.657      | 0.782      |
|             |            | 720                   | **0.831**    | 0.834         | 0.885        | 2.726        | 1.536      | 0.851      |
|             | MAE        | 24                    | **0.381**    | 0.397         | 0.477        | 0.583        | 0.570      | 0.545      |
|             |            | 48                    | 0.459        | **0.433**     | 0.495        | 0.666        | 0.677      | 0.589      |
|             |            | 168                   | **0.567**    | 0.582         | 0.671        | 0.855        | 0.835      | 0.647      |
|             |            | 336                   | **0.620**    | 0.634         | 0.670        | 1.167        | 1.059      | 0.683      |
|             |            | 720                   | **0.731**    | 0.741         | 0.773        | 1.575        | 1.109      | 0.757      |
| **ETTh1**   | MSE        | 48                    | 0.344        | **0.334**     | 0.355        | 1.404        | 0.486      | 0.369      |
|             |            | 168                   | 0.368        | **0.353**     | 0.368        | 1.515        | 0.574      | 0.394      |
|             |            | 336                   | **0.381**    | 0.381         | 0.373        | 1.601        | 0.886      | 0.419      |
|             |            | 720                   | **0.406**    | 0.391         | 0.409        | 2.009        | 1.676      | 0.556      |
|             |            | 960                   | **0.460**    | 0.492         | 0.477        | 2.141        | 1.591      | 0.605      |
|             | MAE        | 48                    | **0.393**    | 0.399         | 0.418        | 0.999        | 0.572      | 0.445      |
|             |            | 168                   | 0.424        | **0.420**     | 0.432        | 1.069        | 0.602      | 0.476      |
|             |            | 336                   | **0.431**    | 0.439         | 0.439        | 1.104        | 0.795      | 0.477      |
|             |            | 720                   | 0.443        | **0.438**     | 0.454        | 1.170        | 1.095      | 0.565      |
|             |            | 960                   | **0.548**    | 0.550         | 0.589        | 1.387        | 1.128      | 0.599      |
| **Count**   |            |                       | **33**       | 14            | 1            | 0            | 0          | 2          |

### LSTF with Granularity Consideration

We perform an additional comparison to explore the performance with various granularities. The sequences \{96, 288, 672\} of ETTm1 (minutes-level) are aligned with \{24, 48, 168\} of ETTh1 (hour-level). The Informer outperforms other baselines even if the sequences are at different granularity levels.

### Parameter Sensitivity

We perform the sensitivity analysis of the proposed Informer model on ETTh1 under the univariate setting.

**Input Length**: In Fig. 4a, when predicting short sequences (like 48), initially increasing input length of encoder/decoder degrades performance, but further increasing causes the MSE to drop because it brings repeat short-term patterns. However, the MSE gets lower with longer inputs in predicting long sequences (like 168). Because the longer encoder input may contain more dependencies, and the longer decoder token has rich local information.

**Sampling Factor**: The sampling factor controls the information bandwidth of ProbSparse self-attention in Eq. (3). We start from the small factor (=3) to large ones, and the general performance increases a little and stabilizes at last in Fig. 4b. It verifies our query sparsity assumption that there are redundant dot-product pairs in the self-attention mechanism. We set the sample factor \( c = 5 \) (the red line) in practice.

**The Combination of Layer Stacking**: The replica of Layers is complementary for the self-attention distilling, and we investigate each stack \{L, L/2, L/4\}’s behavior in Fig. 4c. The longer stack is more sensitive to the inputs, partly due to receiving more long-term information. Our method’s selection (the red line), i.e., joining L and L/4, is the most robust strategy.

### Ablation Study: How well Informer works?

We also conducted additional experiments on ETTh1 with ablation consideration.

#### The performance of ProbSparse self-attention mechanism

In the overall results Table 1 & 2, we limited the problem setting to make the memory usage feasible for the canonical self-attention. In this study, we compare our methods with LogTrans and Reformer, and thoroughly explore their extreme performance. To isolate the memory efficient problem, we first reduce settings as \{batch size=8, heads=8, dim=64\}, and maintain other setups in the univariate case. In Table 3, the ProbSparse self-attention shows better performance than the counterparts. The LogTrans gets OOM in extreme cases because its public implementation is the mask of the full-attention, which still has \( O(L^2) \) memory usage. Our proposed ProbSparse self-attention avoids this from the simplicity brought by the query sparsity assumption in Eq. (4), referring to the pseudo-code in Appendix E.2, and reaches smaller memory usage.

#### Table 3: Ablation Study of the ProbSparse Self-attention Mechanism

| **Prediction length** | **Encoder’s input** | **Informer** | **Informer†** | **LogTrans** | **Reformer** |
|-----------------------|---------------------|--------------|---------------|--------------|--------------|
| **336**               | 336                 | MSE: **0.249**<br>MAE: **0.393** | MSE: 0.225<br>MAE: 0.384 | MSE: 0.216<br>MAE: 0.376 | MSE: 1.875<br>MAE: 1.144 |
|                       | 720                 | MSE: **0.241**<br>MAE: **0.383** | MSE: 0.214<br>MAE: 0.371 | MSE: -<br>MAE: - | MSE: 1.865<br>MAE: 1.129 |
|                       | 1440                | MSE: **0.263**<br>MAE: **0.418** | MSE: 0.231<br>MAE: 0.398 | MSE: -<br>MAE: - | MSE: 1.861<br>MAE: 1.125 |
| **720**               | 720                 | MSE: **0.259**<br>MAE: **0.423** | MSE: -<br>MAE: - | MSE: -<br>MAE: - | MSE: 1.536<br>MAE: 1.497 |
|                       | 1440                | MSE: **0.273**<br>MAE: **0.463** | MSE: -<br>MAE: - | MSE: -<br>MAE: - | MSE: 1.434<br>MAE: 1.434 |

*Note*: Informer† uses the canonical self-attention mechanism. The ‘-’ indicates failure for the out-of-memory.

#### The performance of self-attention distilling

In this study, we use Informer† as the benchmark to eliminate additional effects of ProbSparse self-attention. The other experimental setup is aligned with the settings of univariate Time-series. From Table 5, Informer† has fulfilled all the experiments and achieves better performance after taking advantage of long sequence inputs. The comparison method Informer‡ removes the distilling operation and reaches OOM with longer inputs (> 720). Regarding the benefits of long sequence inputs in the LSTF problem, we conclude that the self-attention distilling is worth adopting, especially when a longer prediction is required.

#### Table 5: Ablation Study of the Self-attention Distilling

| **Prediction length** | **Encoder’s input** | **Informer** | **Informer†** | **Informer‡** |
|-----------------------|---------------------|--------------|---------------|---------------|
| **336**               | 336                 | MSE: **0.249**<br>MAE: **0.393** | MSE: 0.229<br>MAE: 0.391 | MSE: 0.215<br>MAE: 0.387 |
|                       | 480                 | MSE: **0.225**<br>MAE: **0.384** | MSE: 0.204<br>MAE: 0.377 | MSE: -<br>MAE: - |
|                       | 720                 | MSE: **0.216**<br>MAE: **0.376** | MSE: -<br>MAE: - | MSE: -<br>MAE: - |
|                       | 960                 | MSE: **0.199**<br>MAE: **0.371** | MSE: -<br>MAE: - | MSE: -<br>MAE: - |
|                       | 1200                | MSE: **0.186**<br>MAE: **0.365** | MSE: -<br>MAE: - | MSE: -<br>MAE: - |
| **480**               | 336                 | MSE: **0.208**<br>MAE: **0.385** | MSE: 0.224<br>MAE: 0.381 | MSE: 0.243<br>MAE: 0.392 |
|                       | 480                 | MSE: **0.197**<br>MAE: **0.388** | MSE: 0.208<br>MAE: 0.376 | MSE: -<br>MAE: - |
|                       | 720                 | MSE: **0.213**<br>MAE: **0.383** | MSE: -<br>MAE: - | MSE: -<br>MAE: - |
|                       | 960                 | MSE: **0.192**<br>MAE: **0.377** | MSE: -<br>MAE: - | MSE: -<br>MAE: - |
|                       | 1200                | MSE: **0.174**<br>MAE: **0.362** | MSE: -<br>MAE: - | MSE: -<br>MAE: - |

*Note*: Informer‡ removes the self-attention distilling from Informer†. The ‘-’ indicates failure for the out-of-memory.

#### The performance of generative style decoder

In this study, we testify the potential value of our decoder in acquiring a “generative” results. Unlike the existing methods, the labels and outputs are forced to be aligned in the training and inference, our proposed decoder’s predicting relies solely on the time stamp, which can predict with offsets. From Table 6, we can see that the general prediction performance of Informer‡ resists with the offset increasing, while the counterpart fails for the dynamic decoding. It proves the decoder’s ability to capture individual long-range dependency between arbitrary outputs and avoid error accumulation.

#### Table 6: Ablation Study of the Generative Style Decoder

| **Prediction length** | **Prediction offset** | **Informer** | **Informer‡** | **Informer§** |
|-----------------------|----------------------|--------------|---------------|---------------|
| **336**               | +0                   | MSE: **0.207**<br>MAE: **0.385** | MSE: 0.201<br>MAE: 0.393 | MSE: -<br>MAE: - |
|                       | +12                  | MSE: **0.209**<br>MAE: **0.387** | MSE: -<br>MAE: - | MSE: -<br>MAE: - |
|                       | +24                  | MSE: **0.211**<br>MAE: **0.391** | MSE: -<br>MAE: - | MSE: -<br>MAE: - |
|                       | +48                  | MSE: **0.211**<br>MAE: **0.393** | MSE: -<br>MAE: - | MSE: -<br>MAE: - |
|                       | +72                  | MSE: **0.216**<br>MAE: **0.397** | MSE: -<br>MAE: - | MSE: -<br>MAE: - |
| **480**               | +0                   | MSE: **0.198**<br>MAE: **0.390** | MSE: 0.392<br>MAE: 0.484 | MSE: -<br>MAE: - |
|                       | +48                  | MSE: **0.203**<br>MAE: **0.392** | MSE: -<br>MAE: - | MSE: -<br>MAE: - |
|                       | +96                  | MSE: **0.208**<br>MAE: **0.393** | MSE: -<br>MAE: - | MSE: -<br>MAE: - |
|                       | +144                 | MSE: **0.208**<br>MAE: **0.401** | MSE: -<br>MAE: - | MSE: -<br>MAE: - |
|                       | +168                 | MSE: **0.208**<br>MAE: **0.403** | MSE: -<br>MAE: - | MSE: -<br>MAE: - |

*Note*: Informer§ replaces our decoder with dynamic decoding one in Informer‡. The ‘-’ indicates failure for the unacceptable metric results.

### Computation Efficiency

With the multivariate setting and all the methods’ current finest implement, we perform a rigorous runtime comparison in Fig. 5. During the training phase, the Informer (red line) achieves the best training efficiency among Transformer-based methods. During the testing phase, our methods are much faster than others with the generative style decoding. The comparisons of theoretical time complexity and memory usage are summarized in Table 4.

#### Table 4: L-related Computation Statics of Each Layer

| **Methods** | **Training** | **Memory** | **Testing Steps** |
|-------------|--------------|------------|-------------------|
| Informer    | O(L log L)   | O(L log L) | 1                 |
| Transformer | O(L²)        | O(L²)      | L                 |
| LogTrans    | O(L log L)   | O(L log L) | 1*                |
| Reformer    | O(L log L)   | O(L log L) | L                 |
| LSTM        | O(L)         | O(L)       | L                 |

*Note*: The * denotes applying our proposed decoder. The LSTnet is hard to present in a closed form.

## Conclusion

In this paper, we studied the long-sequence time-series forecasting problem and proposed Informer to predict long sequences. Specifically, we designed the ProbSparse self-attention mechanism and distilling operation to handle the challenges of quadratic time complexity and quadratic memory usage in vanilla Transformer. Also, the carefully designed generative decoder alleviates the limitation of traditional encoder-decoder architecture. The experiments on real-world data demonstrated the effectiveness of Informer for enhancing the prediction capacity in LSTF problem.

## Acknowledgments

This work was supported by grants from the Natural Science Foundation of China (U20B2053, 61872022, and 61421003) and State Key Laboratory of Software Development Environment (SKLSDE-2020ZX-12). Thanks for computing infrastructure provided by Beijing Advanced Innovation Center for Big Data and Brain Computing. This work was also sponsored by CAAI-Huawei MindSpore Open Fund. The corresponding author is Jianxin Li.

## Ethics Statement

The proposed Informer can process long inputs and make efficient long sequence inference, which can be applied to the challenging long sequence time-series forecasting (LSTF) problem. The significant real-world applications include sensor network monitoring (Papadimitriou and Yu 2006), energy and smart grid management, disease propagation analysis (Matsubara et al. 2014), economics and finance forecasting (Zhu and Shasha 2002), evolution of agri-ecosystems, climate change forecasting, and variations in air pollution. As a specific example, online sellers can predict the monthly product supply, which helps to optimize long-term inventory management. The distinct difference from other time series problems is its requirement on a high degree of prediction capacity. Our contributions are not limited to the LSTF problem. In addition to acquiring long sequences, our method can bring substantial benefits to other domains, such as long sequence generation of text, music, image, and video.

Under the ethical considerations, any time-series forecasting application that learns from the history data runs the risk of producing biased predictions. It may cause irreparable losses to the real owners of the property/asset. Domain experts should guide the usage of our methods, while the long sequence forecasting can also benefit the work of the domain experts. Taking applying our methods to electrical transformer temperature prediction as an example, the manager will examine the results and decide the future power deployment. If a long enough prediction is available, it will be helpful for the manager to prevent irreversible failure in the early stage. In addition to identifying the bias data, one promising method is to adopt transfer learning. We have donated the collected data (ETT dataset) for further research on related topics, such as water supply management and 5G network deployment. Another drawback is that our method requires high-performance GPU, which limits its application in the underdevelopment regions.

## References

- Ariyo, A. A.; Adewumi, A. O.; and Ayo, C. K. 2014. Stock price prediction using the ARIMA model. In *The 16th International Conference on Computer Modelling and Simulation*, 106–112. IEEE.
- Bahdanau, D.; Cho, K.; and Bengio, Y. 2015. Neural Machine Translation by Jointly Learning to Align and Translate. In *ICLR 2015*.
- Beltagy, I.; Peters, M. E.; and Cohan, A. 2020. Longformer: The Long-Document Transformer. *CoRR abs/2004.05150*.
- Box, G. E.; Jenkins, G. M.; Reinsel, G. C.; and Ljung, G. M. 2015. *Time series analysis: forecasting and control*. John Wiley & Sons.
- Brown, T. B.; Mann, B.; Ryder, N.; Subbiah, M.; Kaplan, J.; Dhariwal, P.; Neelakantan, A.; Shyam, P.; Sastry, G.; Askell, A.; Agarwal, S.; Herbert-Voss, A.; Krueger, G.; Henighan, T.; Child, R.; Ramesh, A.; Ziegler, D. M.; Wu, J.; Winter, C.; Hesse, C.; Chen, M.; Sigler, E.; Litwin, M.; Gray, S.; Chess, B.; Clark, J.; Berner, C.; McCandlish, S.; Radford, A.; Sutskever, I.; and Amodei, D. 2020. Language Models are Few-Shot Learners. *CoRR abs/2005.14165*.
- Child, R.; Gray, S.; Radford, A.; and Sutskever, I. 2019. Generating Long Sequences with Sparse Transformers. *arXiv:1904.10509*.
- Cho, K.; van Merrienboer, B.; Bahdanau, D.; and Bengio, Y. 2014. On the Properties of Neural Machine Translation: Encoder-Decoder Approaches. In *