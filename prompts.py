question_prompt_with_knowledge = """Based on the given facts and your own knowledge, answer the question and return all possible answers in a numbered list.

facts: {}
question: {}
"""

readout_header_template = "\nHere are some chunk-preserving facts about topic {} that may be related to the question."

llm_system_prompt = "You are an AI assistant that helps people find information."

sample_relations_prompt = """Given the question, we have the topic of the question and its relations.

question: {1}
topic: {2}

Based on the question, please select top {0} relations from the options below to explore about the topic to answer the question and just return top {0} selected relations in a numbered list without explanation.
options: {3}
"""

sample_relations_distant_prompt = """Given the question and its topic, we have explored some relations in the previous hop.

question: {1}
topic: {2}

For each numbered block below:
- "Already explored" shows a relation already visited and its discovered entities.
- "Next-hop candidates" lists NEW next-hop relation names you can explore further from those entities.
- Each numbered block corresponds to one previous-hop relation.

Select the top {0} next-hop candidates (from the "Next-hop candidates" lists only) that are most useful to answer the question.
Return only a numbered list of the selected next-hop relation names without explanation.
"""

sample_relations_option_guard = "\n\nOnly return relations from the ones in the options given."

sample_relations_distant_option_guard = '\n\nOnly return next-hop relation names copied from the "Next-hop candidates" lists above.'

sample_relations_retry_prompt = "Selected relations do not exist in the options I provide. Please try again."

sample_relations_distant_block_template = "\n{index}.\nAlready explored: {fact}\nNext-hop candidates: {candidates}\n"

sample_relations_fallback_prompt = """Question: {question}
Topic: {topic}

Using your own knowledge of the relation semantics, select the top {target} relations from the options below that are most useful to answer the question.
Return only a numbered list of relation names copied exactly from the options.

options: {options}
"""

sample_relations_fallback_retry_prompt = "You must return only relation names copied exactly from the options. Return the top {target} relations."
