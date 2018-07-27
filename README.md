
## Machine Learning

- ### [meProp](https://github.com/lancopku/meProp)

  论文 “meProp: Sparsified Back Propagation for Accelerated Deep Learning with Reduced Overfitting”[[pdf]](http://proceedings.mlr.press/v70/sun17c/sun17c.pdf)相关代码。此项工作在后向传播中仅使用一小部分梯度来更新模型参数，计算成本呈线性减少，训练轮数并未增加，而模型准确度却有所提高。


- ### [meSimp](https://github.com/lancopku/meSimp)

  论文 "Training Simplification and Model Simplification for Deep Learning: A Minimal Effort Back Propagation Method" [[pdf]](https://arxiv.org/pdf/1711.06528.pdf)相关代码。此项工作在后向传播中仅使用一小部分梯度更新参数并消除了参数矩阵中一些很少被更新到的行或列。实验显示模型通常可被简化9倍左右，准确度并未受损甚至有所上升。


- ### [Label embedding](https://github.com/lancopku/label-embedding-network)

  论文 “paper Label Embedding Network: Learning Label Representation for Soft Training of Deep Networks”[[pdf]](https://arxiv.org/pdf/1710.10393.pdf)相关代码。这项工作在训练过程中学习标签表示，并让以往并无关联的标签彼此之间产生了交互。模型收敛加快且极大地提高了准确度。同时，学习到的标签表示也更具合理性与可解释性。


## Machine Translation

- ### [Deconv Dec](https://github.com/lancopku/DeconvDec)

  论文"Deconvolution-Based Global Decoding for Neural Machine Translation”[[pdf]](https://arxiv.org/pdf/1806.03692.pdf)相关代码。这项工作提出了一个新的神经机器翻译模型，以对目标序列上下文的结构预测为指导来生成序列，模型获得了极具竞争力的结果，对于不同长度的序列鲁棒性更强，且减轻了生成序列中的重复现象。

- ### [bag-of-words](https://github.com/lancopku/bag-of-words)

  论文“Bag-of-Words as Target for Neural Machine Translation”[[pdf]](https://arxiv.org/pdf/1805.04871.pdf)相关代码。这项工作将目标语句与目标的词袋都作为训练目标，使得模型能够生成出有可能正确却不在训练集中的句子。实验显示模型BLEU值大幅优于基线模型。

- ### [ACA4NMT](https://github.com/lancopku/ACA4NMT)

  论文“Decoding History Based Adaptive Control of Attention for Neural Machine Translation”[[pdf]](https://arxiv.org/pdf/1802.01812.pdf)相关代码。该模型通过追踪解码历史来控制注意力机制。模型能够较少重复地生成翻译且精度更高。



## Summarization 


- ### [LancoSum](https://github.com/lancopku/LancoSum) (toolkit)
  此项目提供了一个针对生成式摘要的工具包，包含通用的基线模型——基于注意力机制的序列到序列模型以及LancoPKU组近期提出的三个高质量摘要模型。通过修改配置文件或命令行，研究者可方便地将其应用至自己的工作。

- ### [Global-Encoding](https://github.com/lancopku/Global-Encoding)

  论文“Global Encoding for Abstractive Summarization” [[pdf]](https://arxiv.org/pdf/1805.03989.pdf)相关代码。这项工作提出了一个基于全局源语言信息控制编码段到解码端信息流的框架，模型优于多个基线模型且能够减少重复输出。

- ### [HSSC](https://github.com/lancopku/HSSC)

  论文“A Hierarchical End-to-End Model for Jointly Improving Text Summarization and Sentiment Classification”[[pdf]](https://arxiv.org/pdf/1805.01089.pdf)相关代码。这项工作提出了一个联合学习摘要和情感分类的任务的模型。实验显示所提出模型在两项任务上都取得了相较强基线模型更好的效果。


- ### [WEAN](https://github.com/lancopku/WEAN)

  论文“Query and Output: Generating Words by Querying Distributed Word Representations for Paraphrase Generation”[[pdf]](https://arxiv.org/pdf/1803.01465.pdf)相关代码。在生成摘要时，该模型通过查询单词表示（词嵌入）来产生新的单词。模型大幅度优于基线模型且在三个基准数据集上达到了最优表现。


- ### [SRB](https://github.com/lancopku/SRB)

  论文“Improving Semantic Relevance for Sequence-to-Sequence Learning of Chinese Social Media Text Summarization”[[pdf]](https://arxiv.org/pdf/1706.02459.pdf)相关代码。这项工作使源文本与摘要的表示获得尽可能高的相似度，而极大地提高了源文本与生成摘要的语义关联。




- ### [superAE](https://github.com/lancopku/superAE)

  论文“Autoencoder as Assistant Supervisor: Improving Text Representation for Chinese Social Media Text Summarization”[[pdf]](https://arxiv.org/pdf/1805.04869.pdf)相关代码。这项工作将摘要自编码器作为给序列到序列模型的一个监督信号，来获得更具信息量的源文本表示。实验结果显示模型在基准数据集上获得了最优效果。


## Text Generation

- ### [Unpaired-Sentiment-Translation](https://github.com/lancopku/Unpaired-Sentiment-Translation)

  论文“Unpaired Sentiment-to-Sentiment Translation: A Cycled Reinforcement Learning Approach" [[pdf]](https://arxiv.org/pdf/1805.05181.pdf)相关代码。这项工作提出了一个循环强化学习的方式来实现情感转换。所提出方法不依赖任何平行语料，且在内容保留程度上，显著地优于现有的最优模型。



- ### [DPGAN](https://github.com/lancopku/DPGAN)

  论文“DP-GAN: Diversity-Promoting Generative Adversarial Network for Generating Informative and Diversified Text” [[pdf]](https://arxiv.org/pdf/1802.01345.pdf)相关代码。这项工作在生成对抗网络中创新地引入了一个基于语言模型的判别器。所提出模型能够生成显著得优于基线方法的更具多样性和信息量的文本。


## Dependency Parsing

- ### [nndep](https://github.com/lancopku/nndep)

  论文“Hybrid Oracle: Making Use of Ambiguity in Transition-based Chinese Dependency Parsing”[[pdf]](https://arxiv.org/pdf/1711.10163.pdf)相关代码。此项工作利用一个分析状态的所有正确转移对损失函数提供更好的监督信号。新的分析器在中文依存句法分析上优于使用传统策略的分析器，同时，此句法分析器也可用于对同一句子生成多个转移序列。

## Sequence Labeling

- ### [PKUSeg](https://github.com/lancopku/PKUSeg) (toolkit)

  此项目提供了一个针对中文分词的工具包。PKUSeg简单易用，支持多领域分词，在不同领域的数据上都大幅提高了分词的准确率。

- ### [ChineseNER](https://github.com/lancopku/ChineseNER)

  论文“Cross-Domain and Semi-Supervised Named Entity Recognition in Chinese Social Media: A Unified Model”相关代码。[[pdf]](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8411523). 此项工作将域外数据与域内未标注数据结合起来提高社交媒体中的命名实体识别表现。模型表现在强基线模型的基础上获得了明显提升。
  

- ### [Multi-Order-LSTM](https://github.com/lancopku/Multi-Order-LSTM)

  论文"Does Higher Order LSTM Have Better Accuracy for Segmenting and Labeling Sequence Data?”[[pdf]](https://arxiv.org/pdf/1711.08231.pdf)相关代码。此项工作合并了低阶LSTM模型和高阶LSTM模型，考虑到了标签之间的长距离依赖。模型保持了对更高阶的模型的扩展性，尤其对于长实体的识别表现优异。


- ### [Decode-CRF](https://github.com/lancopku/Decode-CRF)

  论文“Conditional Random Fields with Decode-based Learning: Simpler and Faster”[[pdf]](https://arxiv.org/pdf/1503.08381.pdf)相关代码。此项工作提出了一个基于解码的概率化在线学习方法。该方法训练很快，易于实现，准确率高，且理论可收敛。


## Text Classification

- ###  [SGM](https://github.com/lancopku/SGM)

  论文“SGM: Sequence Generation Model for Multi-label Classification”[[pdf]](https://arxiv.org/pdf/1806.04822.pdf)相关代码。此项工作将多标签分类任务看做序列生成任务。所提出方法不仅能捕捉标签之间的关联，还能在预测不同标签时自动选择出最具信息量的单词。



## Applied Tasks

- ### [AAPR](https://github.com/lancopku/AAPR)

  论文“Automatic Academic Paper Rating Based on Modularized Hierarchical Convolutional Neural Network”
[[pdf]](https://arxiv.org/pdf/1805.03977.pdf)相关代码。此项工作建立了一个自动评估学术论文的的数据集，并提出了一个适用用此任务的模块化的层级卷积网络。


- ### [tcm_prescription_generation](https://github.com/lancopku/tcm_prescription_generation)

  论文“Exploration on Generating Traditional ChineseMedicine Prescriptions from Symptoms with an End-to-End Approach”[[pdf]](https://arxiv.org/pdf/1801.09030.pdf)相关代码。此项工作利用序列到序列模型，探索了传统中医的药方生成任务。



## Datasets

- ### [Chinese-Literature-NER-RE-Dataset](https://github.com/lancopku/Chinese-Literature-NER-RE-Dataset)

  论文“A Discourse-Level Named Entity Recognition and Relation Extraction Dataset for Chinese Literature Text” [[pdf]](https://arxiv.org/pdf/1711.07010.pdf)相关数据。本篇工作从数百篇中文散文中建立了一个篇章级别的数据集，旨在提高命名实体识别和关系抽取任务在散文上的表现。

- ### [Chinese-Dependency-Treebank-with-Ellipsis](https://github.com/lancopku/Chinese-Dependency-Treebank-with-Ellipsis)

  论文“Building an Ellipsis-aware Chinese Dependency Treebank for Web Text”[[pdf]](https://arxiv.org/pdf/1801.06613.pdf)相关数据。本篇工作建立了一个中文微博依存树库，包含572个在保留语义的情况下还原了省略语的句子，旨在提高依存句法分析在存在省略的文本上的表现。

- ### [Chinese-abbreviation-dataset](https://github.com/lancopku/Chinese-abbreviation-dataset)

  论文“A Chinese Dataset with Negative Full Forms for General Abbreviation Prediction” [[pdf]](https://arxiv.org/pdf/1712.06289.pdf)相关数据。本篇工作建立了一个通用的中文缩略语预测数据集，该数据集涵盖了无缩略语的完全短语，旨在促进这一领域的研究。




