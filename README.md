# KCL-SSM
## Contrastive Learning Based Sentence Semantic Matching with Knowledge Graph Augmentation

A sentence semantic matching framework for domain-specific question matching, integrating **Teacher-Student learning, Contrastive Learning, Knowledge Graph-enhanced candidate retrieval, and Hybrid Scoring**.

<img width="2401" height="988" alt="KCL-SSM Overall Architecture" src="https://github.com/user-attachments/assets/58e27f52-f101-4c60-8242-bfcbb03ec929" />

## Overview

KCL-SSM is a sentence semantic matching framework designed to match diverse user queries with predefined standard questions. It addresses two key challenges in domain-specific applications: **limited labeled data** and **lexical variation in user expressions**.

The framework consists of three main phases:

### Phase I: Model Construction

LLM-generated synthetic queries are first used to enrich the training data with diverse expressions. A **Teacher-Student architecture with Contrastive Learning** is then developed to learn discriminative semantic representations.

The Teacher model is pretrained with the expanded synthetic data, while the Student model learns from real-world user queries through knowledge distillation. In parallel, a domain-specific **Knowledge Graph** is constructed from keywords extracted from standard questions to capture relationships among domain concepts.

### Phase II: Candidate Set Generation

Given a user query, **Aho-Corasick (AC)** keyword matching is integrated with the domain-specific Knowledge Graph to retrieve relevant candidate standard questions.

This candidate retrieval stage reduces the search space for semantic matching and improves the efficiency of subsequent ranking.

### Phase III: Query Sentence Matching

The fine-tuned Student model is used as the sentence encoder to obtain semantic representations of the user query and candidate standard questions.

Semantic similarity is combined with graph-based scores through a **Hybrid Scoring** strategy to improve the final ranking, particularly for complex or ambiguous queries.

## Key Features

- **Teacher-Student Architecture**  
  Uses LLM-generated synthetic queries to pretrain the Teacher model and transfers semantic knowledge to the Student model through knowledge distillation.

- **Contrastive Learning**  
  Learns discriminative sentence representations for fine-grained semantic matching.

- **Knowledge Graph-Enhanced Candidate Retrieval**  
  Combines Aho-Corasick keyword matching with a domain-specific Knowledge Graph to reduce the candidate search space.

- **Hybrid Scoring**  
  Combines semantic similarity and graph-based scores to improve candidate ranking.

## Methodology

### Phase I: Model Construction

The Teacher-Student architecture combines LLM-based data augmentation, Contrastive Learning, and Knowledge Distillation to improve semantic representation learning under limited labeled data.

<img width="1550" height="725" alt="Teacher-Student Architecture" src="https://github.com/user-attachments/assets/dbcd0c3f-da4d-4350-8d83-6223aae426aa" />

In parallel, a domain-specific Knowledge Graph is constructed from keywords extracted from standard questions. The graph captures relationships among domain concepts and provides structured information for subsequent candidate retrieval.

<img width="1745" height="406" alt="image" src="https://github.com/user-attachments/assets/d91a42e5-a853-42cd-9729-41414af207ca" />


### Phase II: Candidate Set Generation

Aho-Corasick keyword matching and the domain-specific Knowledge Graph are used to retrieve candidate standard questions before semantic matching.

<img width="2204" height="403" alt="Knowledge Graph Candidate Retrieval" src="https://github.com/user-attachments/assets/9278ce4c-b617-402c-8e4b-0ba4d44ea2c1" />

### Phase III: Query Sentence Matching

The Student model encodes the user query and candidate standard questions. Semantic similarity and graph-based scores are then combined through Hybrid Scoring for final ranking.

<img width="1575" height="444" alt="Hybrid Scoring" src="https://github.com/user-attachments/assets/d526cf14-1913-4306-afb4-ae0a0099d551" />

## Application Scenario

The framework was evaluated in a real-world **traffic adjudication customer service** scenario, where colloquial user queries are matched to **251 predefined standard questions**.

Example:

> User query: 「我剛收到一張罰單，我想問一下這個可以怎麼申訴？」

↓

> Standard question: 「我想要了解如何申訴？」

The application demonstrates how semantic matching can bridge the lexical gap between colloquial user expressions and predefined knowledge-base questions.

## Dataset

KCL-SSM was evaluated on four sentence matching datasets, including three public intent classification benchmarks and one real-world domain-specific dataset.

| Dataset | Train Queries | Test Queries | Standard Questions |
|---|---:|---:|---:|
| Banking77 | 10,003 | 3,080 | 77 |
| HWU64 | 8,954 | 1,076 | 64 |
| CLINC150 | 18,000 | 4,500 | 150 |
| **Foxconn** | **1,809** | **84** | **251** |

### Foxconn Dataset

The Foxconn dataset was collected through an industry–academia collaboration with a publicly listed company and originates from the Kaohsiung City Traffic Adjudication Office. It consists of real customer service queries from a transportation-related scenario and is designed for standard question matching.

The dataset contains 251 standard questions, 1,809 training utterances, and 84 test utterances. As the data were collected from real customer service interactions, they reflect practical challenges in enterprise applications, including limited training data, highly colloquial user expressions, and overlapping fine-grained intents.

> **Note:** Raw datasets and user data are not included in this repository.

## Experimental Results

KCL-SSM was evaluated on three public intent classification benchmarks and one real-world domain-specific dataset.

| Dataset   |      Acc@1 |  Acc@3 |  Acc@5 |        MRR | nDCG@3 | nDCG@5 |
| --------- | ---------: | -----: | -----: | ---------: | -----: | -----: |
| Banking77 |     93.21% | 97.6% | 98.28% | 95.5% | 95.82% | 96.1% |
| HWU64     |     94.8% | 97.86% | 98.88% |     96.52% | 96.52% | 96.67% |
| CLINC150  |     97.42% | 99.29% | 99.44% |     98.37% | 98.56% | 98.6% |
| Foxconn   | 73.81% | 86.9% | 88.1% |     80.61% | 81.5% | 80.2% |

The results demonstrate the effectiveness of KCL-SSM for semantic matching across both public benchmark datasets and real-world domain-specific queries.

## Project Structure

The repository contains the core Teacher and Student implementations, training pipelines, preprocessing modules, and inference components.

```text
KCL-SSM-Portfolio/
├── README.md
├── teacher/
│   ├── model.py
│   ├── pipeline.py
│   ├── pipeline_params.py
│   ├── utils.py
│   ├── requirements.txt
│   └── Train_model/
│       ├── dataset.py
│       ├── hyperparameter.py
│       ├── loss.py
│       ├── model_train.py
│       ├── run_train.py
│       ├── trainer.py
│       ├── utils_train.py
│       └── preprocess_generate/
└── student/
    ├── model.py
    ├── pipeline.py
    ├── pipeline_params.py
    ├── predict.py
    ├── requirements.txt
    ├── utils.py
    ├── AC_KG_CL_Ranker/
    │   ├── build_graph.py
    │   ├── inference_metrics.py
    │   └── query_system.py
    └── Train_model/
        ├── dataset.py
        ├── hyperparameter.py
        ├── loss.py
        ├── model_train.py
        ├── run_train.py
        ├── trainer.py
        ├── utils_train.py
        └── preprocess_generate/
```

## Implementation

### Model

- Python
- PyTorch
- Contrastive Learning
- Knowledge Distillation
- Teacher-Student Architecture

### Retrieval and Ranking

- Aho-Corasick keyword matching
- Domain-specific Knowledge Graph
- Cosine Similarity
- Hybrid Scoring
- Candidate retrieval and reranking

### Data and LLM

- LLM-generated synthetic data
- Prompt Engineering
- Data augmentation
- Domain-specific question matching

## Environment

The original experiments were conducted using:

- Ubuntu 22.04.4
- Python 3.9
- PyTorch
- Hugging Face Transformers

The repository focuses on the core implementation and model architecture. Original datasets, trained model checkpoints, and deployment-specific files are not included.

## Notes

This repository is provided as a **research and portfolio implementation** of KCL-SSM. The included code focuses on the core modeling, preprocessing, candidate retrieval, and ranking components.

For privacy and project restrictions, raw real-world user data and other non-public project materials are excluded.

## Technologies

**Python · PyTorch · Transformer · Contrastive Learning · Knowledge Distillation · Knowledge Graph · Aho-Corasick · LLM Data Augmentation · Sentence Semantic Matching**
