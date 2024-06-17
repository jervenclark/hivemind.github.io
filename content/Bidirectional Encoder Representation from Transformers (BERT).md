BERT is a language model based on the [[Transformer Architecture]], notable for its dramatic improvement over previous state of the art models. It was introduced in October 2018 by researchers at Google. A 2020 literature survey concluded that in a little over a year, BERT has become a ubiquitous baseline in [[Natural Language Processing (NLP)]] experiments counting over 150 research publications analyzing and improving the model.

BERT was originally implemented in the English language at two model sizes BERT$_{BASE}$ (12 encoders with 12 bidirectional self-attention heads totaling 110 million parameters) and BERT$_{LARGE}$ (24 encoders with 16 bidirectional self-attention heads totaling 340 million parameters). Both models were pre-trained on the [[Toronto BookCorpus]] and English Wikipedia.

## Design
BERT is an "encoder-only" transformer architecture.
On a high level, BERT consists of three modules:
- **embedding**: his module converts an array of one-hot encoded tokens into an array of vectors representing the tokens.
- **a stack of encoders**: these encoders are the transformer encoders. They perform transformations over the array representation vectors
- **un-embedding**: this module converts the final representation vectors into one-hot encoded tokens again

The un-embedding module is necessary for pretraining, but it is often unnecessary for downstream tasks. Instead, one would take the representation vectors output at the end of the stack of encoders, and use those as a vector representation of the text input, and train a smaller model on top of that.

BERT uses [[WordPiece]] to convert each English word into an integer code. Its vocabulary has size 30,000. Any token not appearing in its vocabulary is replaced by UNK for "unknown".

### Pretraining
BERT was pretrained simultaneously on two tasks.
#### Language Modeling
15% of tokens were selected for prediction, and the training objective was to predict the selected token given its context. The selected token is
- replaced with a MASK token with 80% probability
- replaced with a random word token with 10% probability
- not replaced with 10% probability

For example, the sentence "my dog is cute" may have the 4th token selected for prediction. The model would have input text
- my dog is MASK
- my dog is happy
- my dog is cute

After processing the input text, the models 4th output vector is passed to a separate neural network, which outputs a probability distribution over its 30k-large vocabulary.

#### Next sentence prediction
Given two spans of text, the model predicts if these two spans appeared sequentially in the training corpus, outputting either IsNext or NotNext. The first span starts with a special token CLS for "classify". The two spans are separated by a special token SEP for separate. After processing the two spans, the 1st output vector is passed to a separate neural network for the binary classification into IsNext or NotNext.
- For example, given "CLS my dog is cute SEP he likes playing" the model should output the token IsNext
- Given "CLS my dog is cute SEP how do magnets work" the model should output token NotNext

As a result of this training process, BERT learns [[Latent Representations]] of words and sentences in context. After pretraining, BERT can be fine-tuned with fewer resources on smaller datasets to optimize its performance on specific NLP tasks such as language inference, text classifications, etc, and sequence-to-sequence based language generation tasks like question-answering, conversational response generation. The pretraining stage is significantly more computationally expensive than fine-tuning.

## Performance
When BERT was published, it achieved state-of-the-art performance on a number of natural language understanding tasks:
- [[General Language Understanding Evaluation (GLUE)]]
- [[Stanford Question Answering Dataset (SQuAD)]]
- [[Situations With Adversarial Generations (SWAG)]]

## Analysis
The reason for BERT's state-of-the-art performance on these [[Natural Language Understanding]] tasks are not yet well understood. Current research has focused on investigating the relationship behind BERT's output as a result of carefully chosen input sequences, analysis of internal vector representations through probing classifiers, and the relationships represented by attention weights. The high performance of the BERT model could also be attributed to the fact that it is bidirectionally trained. This means that BERT, based on the Transformer model architecture, applies its self-attention mechanism to learn information from a text from the left and right side during training, and consequently gains a deep understanding of context.

For example, the word "fine" can have two different meanings depending on context "I feel fine" and "She has fine blond hair". BERT considers the words surrounding the target word "fine" from left and right side. 

However it comes at a cost, due to its encoder-only architecture lacking a decoder, BERT cant be prompted and cant generate text, while bidirectional models in general do not work effectively without the right side, thus being difficult to prompt, with even short text generation requiring sophisticated computationally expensive techniques.

In contrast to deep learning neural networks which require very large amounts of data, BERT has already been pretrained which means that it has learnt the representations of the words and sentences as well as the underlying semantic relations that they are connected with. BERT can then be fine-tuned on smaller datasets for specific tasks such as sentiment classification. The pretrained models are chosen according to the content of the given dataset one uses but also the goal of the task. For example, if the task is a sentiment classification task on financial data, a pretrained model for the analysis of sentiment of financial text should be chosen. The weights of the original pretrained models were released on github.

## History
BERT was orignally published by Google researchers Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. The design has its origins from pretraining contextual representations, including [[Semi-supervised Sequence Learning]], [[Generative Pretraining]], [[ELMo]], and [[ULMFit]]. Unlike previous models, BERT is a deeply bidirectional, unsupervised language representation, pretrained using only a plain text corpus. Context-free models such as [[Word2Vec]] or [[Global Vectors (GloVe)]] generate a single word embedding representation for each word int eh vocabulary, where as BERT takes into account the context for each occurrence of a given word. For instance, whereas the vector for "running" will have the same word2vec vector representation for both of its occurrences in the sentences "He is running a company" and "He is running a marathon", BERT will provide a contextualized embedding that will be different according to the sentence.

On October 25, 2019, Google announced that they had started applying BERT models for English language search queries within the US. On December 9, 2019, it was reported that BERT had been adopted by Google Search for over 70 languages. In October 2020, almost every single English-based query was processed by a BERT model.

A later paper proposes [[Robustly Optimized BERT Pretraining Approach (RoBERTa)]], which preserves BERT's architecture, but improves its training, changing key hyper-parameters, removing the next-sentence prediction task, and using much larger mini-batch sizes. 

## Notes
### Resources
- [[BERT꞉ Pre-training of Deep Bidirectional Transformers for Language Understanding.pdf]]
### Further Readings
- 