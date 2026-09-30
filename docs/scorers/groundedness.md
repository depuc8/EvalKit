# Groundedness

Checks that each **claim** in the answer is supported by the context, and lists the ones that
are not. Score = supported claims / total claims.

```python
GroundednessScorer()                      # lexical heuristic
GroundednessScorer(judge=my_judge)        # claim-level LLM check
```

## Where the context comes from

1. `case.context`, if the dataset provides it (a *golden* context), otherwise
2. what the agent actually retrieved: LangChain retriever steps, plus results of tools you declared
   with `LangGraphAdapter(graph, context_tools={"search_docs"})`.

If neither exists the scorer is skipped. Use `use_case_context=False` to always judge against
what was retrieved - the right choice when you want to catch retrieval failures rather than
generation failures.

## Heuristic mode

The answer is split into sentences (fragments under three words are dropped). A claim is
**supported** when at least `support_threshold` (default 0.6) of its content words - words that
are not stopwords - appear in the context **or in the user's question**. Words the user supplied
need no evidence. Numbers count as content words, so an invented figure is flagged.

Answers that abstain ("I don't know", "not enough information", ...) make no claims and score 1.0.

!!! warning "Limits of the heuristic"
    It is lenient: it cannot tell that two true facts were combined into a false statement, and
    it will not catch a negation. It is a fast first line of defence for regressions; use a judge
    when you need real entailment checking.

## Judge mode

The judge splits the answer into claims and marks each supported or not, with evidence, using
only the context. The reply format is
`{"claims": [{"claim": "...", "supported": true, "evidence": "..."}]}`.

## Reading the result

```python
score.details["unsupported_claims"]   # ["Refunds are issued within 90 days."]
score.details["claims"]               # every claim with its verdict
```

The HTML report prints unsupported claims under the case's score.
