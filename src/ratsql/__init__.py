"""RAT-SQL: a research re-implementation of relation-aware schema encoding and
linking for Text-to-SQL (Wang et al., ACL 2020).

Sub-packages
------------
schema      Schema model, relational schema graph, relation vocabulary/matrix,
            n-gram and value-based schema linking.
sql         Spider-style SQL tokenizer/parser, ASDL-like grammar, AST,
            transition system (action sequences + grammar masks), serializer.
data        Spider loading, synthetic data, deterministic preprocessing,
            torch datasets / collation.
models      BERT / BiLSTM input encoders, relation-aware transformer,
            LSTM AST decoder, full Text-to-SQL model.
training    Trainer (AMP, accumulation, clipping, scheduling, checkpoints).
evaluation  Exact match (Spider exact-set-match re-implementation),
            execution accuracy, SQL validity, hardness, complexity, errors.
inference   Checkpoint loading and prediction helpers.
analysis    Attention and schema-linking analysis utilities.
"""

__version__ = "1.0.0"
