# Data handling

`data/fixtures` is synthetic/de-identified development data.

`data/active_policies.jsonl` contains reviewed active policy sets. Each set can
contain one or more simultaneously active rules.

`data/content_samples.jsonl` contains English text conversations and the tools
that Gemini may call while producing supervision. Bundled smoke rows also carry
hidden semantic expectations and mark selected tool sequences as required;
neither expectation field is exposed to Gemini or the exported SFT examples.

`data/tool_env` is a deterministic synthetic lookup environment for real tool
trajectory generation. It is not an oracle or production moderation evidence.

`data/oracle` is never supplied to retrieval or model prompts.

Production evaluation requires a separately approved/de-identified business holdout and must not be committed.

Generated `events.jsonl` files contain operational metadata only. They must not
be extended with raw prompts, candidate text, tool inputs/outputs, credentials,
provider response bodies, or service-account details.
