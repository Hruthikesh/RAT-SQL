# RAT-SQL | Relation-Aware Text-to-SQL

This is my research-oriented implementation of **RAT-SQL (Relation-Aware Transformer for Text-to-SQL)**.

The main goal of the project is to understand how a model can take a natural-language question, understand the structure of a database, and generate the corresponding SQL query.

I worked mainly on understanding and implementing the different parts of the RAT-SQL pipeline, including schema encoding, relation-aware attention, SQL decoding, preprocessing, training, and evaluation.

## What this project does

Given a question such as:

> "How many employees are there in each department?"

and a database schema containing tables, columns, and relationships, the model tries to generate the corresponding SQL query.

The important part is that the model does not treat the database schema as plain text. It uses relationships between tables and columns while encoding the input.

## Main Components

- **Schema representation**
  - Tables and columns
  - Primary-key relationships
  - Foreign-key relationships
  - Column and table types

- **Relation-aware encoding**
  - Relation-aware attention
  - Schema linking
  - Question-to-schema interactions

- **Text-to-SQL decoding**
  - Structured SQL generation
  - Grammar-aware decoding
  - Pointer-based selection from schema elements

- **Training and evaluation**
  - BERT-based encoder
  - PyTorch implementation
  - Exact Match evaluation
  - Execution Accuracy evaluation
  - Controlled experiments with different model configurations

## Experiments

I also experimented with different model sizes and training settings rather than treating the model as a single black-box implementation.

Some of the results obtained during the experiments were:

| Model | Split | Exact Match | Execution Accuracy |
|---|---|---:|---:|
| BERT-medium, 40 epochs | Dev | 64.9% | 66.9% |
| BERT-medium | Test | 59.3% | 63.4% |
| BERT-small, seed 42 | Dev | 61.4% | 63.7% |
| BERT-small, seed 43 | Dev | 59.1% | 61.1% |

These numbers are from the experiments already run for this project. The repository keeps the configurations and experiment-related results so the work can be inspected instead of only showing the final numbers.

## Project Structure

```text
RAT-SQL/
│
├── configs/          # Model and experiment configurations
├── data/             # Data preparation and schema handling
├── experiments/      # Experiment configurations and results
├── notebooks/        # Exploratory work
├── reports/          # Experiment/research notes
├── results/          # Evaluation results
├── scripts/           # Utility and experiment scripts
├── src/               # Main RAT-SQL implementation
├── tests/             # Offline test suite
│
├── PROJECT_STATUS.md
├── README.md
├── requirements.txt
├── pyproject.toml
├── Makefile
└── LICENSE
```

## Tech Stack

**Languages**
- Python

**Deep Learning**
- PyTorch
- Transformers

**NLP / Text-to-SQL**
- BERT
- RAT-SQL
- Schema linking
- Grammar-aware SQL decoding

**Data & Evaluation**
- Spider dataset
- NumPy
- Exact Match
- Execution Accuracy

**Development**
- Git / GitHub
- pytest
- Jupyter Notebook

## Testing

The repository includes an offline test suite for the main components.

The current test suite passes completely:

```text
61 passed
```

The tests use small synthetic data and lightweight model configurations where possible, so the core implementation can be checked without downloading large models or datasets.

## Why I built this

I wanted to go beyond simply using an existing Text-to-SQL model through an API.

This project helped me understand how a research model is actually put together — from database schema representation and attention mechanisms to decoding, training, evaluation, and experiment tracking.

It is also part of my broader work around **LLMs, RAG, Text-to-SQL and model evaluation**, where I am interested in understanding when different approaches actually work rather than only building applications around them.

## References

This project is based on the ideas introduced in:

**RAT-SQL: Relation-Aware Schema Encoding and Linking for Text-to-SQL Parsers**

The original research paper and the Spider benchmark are the main references for the implementation and evaluation.

## Why I Built This

I built this project to understand how Text-to-SQL systems work beyond just calling an existing model or API.

I wanted to learn how a model represents database schemas, uses relationships between tables and columns, connects a natural-language question with the schema, and finally generates a SQL query.

I also used the project to experiment with different model configurations and compare their performance using Exact Match and Execution Accuracy.

This project gave me hands-on experience with research-oriented deep learning, NLP, model training, and evaluation.

## Author

**Thalla Hruthikesh**  
B.Tech, IIT Kharagpur

Student research/project implementation created for learning and experimentation.