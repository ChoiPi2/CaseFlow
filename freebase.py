from SPARQLWrapper import SPARQLWrapper, JSON
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from prompts import (
    sample_relations_distant_block_template,
    sample_relations_distant_option_guard,
    sample_relations_distant_prompt,
    sample_relations_fallback_prompt,
    sample_relations_fallback_retry_prompt,
    sample_relations_option_guard,
    sample_relations_prompt,
    sample_relations_retry_prompt,
)
from utils import run_llm, token_count, get_list_str
import time

SPARQLPATH = os.environ.get("FREEBASE_SPARQL_ENDPOINT", "").strip()

SPARQL_TIMEOUT = int(os.environ.get("FREEBASE_SPARQL_TIMEOUT", "180"))
SPARQL_MAX_RETRY = int(os.environ.get("FREEBASE_SPARQL_MAX_RETRY", "2"))
SPARQL_RETRY_SLEEP = float(os.environ.get("FREEBASE_SPARQL_RETRY_SLEEP", "10"))

def execute_sparql(sparql_query: str) -> list:
    if not SPARQLPATH:
        raise ValueError(
            "Missing FREEBASE_SPARQL_ENDPOINT. Export the SPARQL endpoint URL before running Freebase queries."
        )
    sparql = SPARQLWrapper(SPARQLPATH)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)

    try:
        sparql.setTimeout(SPARQL_TIMEOUT)
    except Exception:
        pass

    results = None
    for _ in range(SPARQL_MAX_RETRY):
        try:
            results = sparql.query().convert()
            break
        except Exception:
            time.sleep(SPARQL_RETRY_SLEEP)

    if results is None:
        return []

    return results["results"]["bindings"]

def _fallback_sample_relations_by_name(question: str, topic_name: str, relations: list, args, distant: bool = False) -> list:
    target = min(args.width, len(relations))
    if target <= 0:
        return []

    prompt = sample_relations_fallback_prompt.format(
        question=question,
        topic=topic_name,
        target=target,
        options=", ".join(relations),
    )

    sampled_relations = []
    history = []
    retry_prompt = sample_relations_fallback_retry_prompt.format(target=target)

    for _ in range(args.max_retry):
        response = run_llm(prompt, args) if not history else run_llm(prompt, args, history, retry_prompt)
        sampled_relations = get_sampled_relations(
            response, relations, allow_suffix_match=not distant
        )
        if len(sampled_relations) >= target:
            break
        history.append(response)

    tag = " (distant)" if distant else ""
    print(f"[Fallback] LLM top-k fallback{tag}: {len(sampled_relations)} relations selected", flush=True)
    return sampled_relations

def sample_relations(question: str, topic_name: str, relations: list, args) -> list:
    prompt = sample_relations_prompt.format(args.width, question, topic_name, ', '.join(relations))
    while token_count(prompt) > args.limit_llm_in:
        relations = relations[:-1]
        prompt = sample_relations_prompt.format(args.width, question, topic_name, ', '.join(relations))
    prompt += sample_relations_option_guard
    response = run_llm(prompt, args)
    sampled_relations = get_sampled_relations(response, relations)
    target = min(args.width, len(relations))
    minimum = target
    history = []
    retry_prompt = sample_relations_retry_prompt
    while minimum > 0 and len(sampled_relations) < minimum and len(history) < args.max_retry:
        print('Sampling failed, Retrying.')
        minimum = max(minimum - 1, 0)
        history.append(response)
        response = run_llm(prompt, args, history, retry_prompt)
        sampled_relations = list(dict.fromkeys(sampled_relations + get_sampled_relations(response, relations)))

    if target > 0 and len(sampled_relations) < target and (len(history) >= args.max_retry or minimum <= 0):
        sampled_relations = _fallback_sample_relations_by_name(question, topic_name, relations, args)

    return sampled_relations

def sample_relations_distant(question: str, topic_name: str, relations: dict, args) -> list:
    prompt = sample_relations_distant_prompt.format(args.width, question, topic_name)
    canonical_relations = []
    for i, r in enumerate(relations):
        canonical_relations.extend([f"{r}->{r2}" for r2 in relations[r]['relation']])
        prompt += sample_relations_distant_block_template.format(
            index=i + 1,
            fact=relations[r]['fact'],
            candidates=', '.join(relations[r]['relation']),
        )
    while token_count(prompt) > args.limit_llm_in:
        prompt = sample_relations_distant_prompt.format(args.width, question, topic_name)
        max_count = max([token_count(relations[r]['relation']) for r in relations])
        if max_count == 0:
            raise ValueError("The facts exceed LLM token limit.")
        for i, r in enumerate(relations):
            if token_count(relations[r]['relation']) == max_count:
                relations[r].update({'relation': relations[r]['relation'][:-1]})
            prompt += sample_relations_distant_block_template.format(
                index=i + 1,
                fact=relations[r]['fact'],
                candidates=', '.join(relations[r]['relation']),
            )
    canonical_relations = []
    for r in relations:
        for i in relations[r]['relation']:
            canonical_relations.append(r + '->' + i)
    prompt += sample_relations_distant_option_guard
    response = run_llm(prompt, args)
    sampled_relations = get_sampled_relations(
        response, canonical_relations, allow_suffix_match=True
    )
    target = min(args.width, len(canonical_relations))
    minimum = target
    history = []
    retry_prompt = sample_relations_retry_prompt
    while minimum > 0 and len(sampled_relations) < minimum and len(history) < args.max_retry:
        minimum = max(minimum - 1, 0)
        print('Sampling failed, Retrying.')
        history.append(response)
        response = run_llm(prompt, args, history, retry_prompt)
        sampled_relations = list(dict.fromkeys(
            sampled_relations + get_sampled_relations(
                response, canonical_relations, allow_suffix_match=True
            )
        ))

    if target > 0 and len(sampled_relations) < target and (len(history) >= args.max_retry or minimum <= 0):
        sampled_relations = _fallback_sample_relations_by_name(
            question, topic_name, canonical_relations, args, distant=True
        )

    return sampled_relations

def get_sampled_relations(response: str, relations: list, allow_suffix_match: bool = True) -> list:
    def _normalize_relation(text: str) -> str:
        text = text.strip()
        text = re.sub(r'^\s*\d+[\.)]?\s*', '', text)
        text = text.strip()
        text = re.sub(r'\s*->\s*', '->', text)
        text = re.sub(r'[\"\'`]', '', text)
        text = text.strip(" .,:;()[]{}")
        return text

    raw_items = list(get_list_str(response))
    raw_items.extend(line.strip() for line in response.splitlines() if line.strip())
    response_set = set()
    for item in raw_items:
        norm = _normalize_relation(item)
        if not norm:
            continue
        response_set.add(norm)
        if '->' in norm:
            response_set.add(norm.rsplit('->', 1)[-1])

    sampled_relations = []
    for relation in relations:
        norm_relation = _normalize_relation(relation)
        last_hop = norm_relation.rsplit('->', 1)[-1]
        spaced_relation = norm_relation.replace('->', ' -> ')
        if (
            norm_relation in response_set
            or spaced_relation in response_set
            or (allow_suffix_match and last_hop in response_set)
        ):
            sampled_relations.append(relation)

    return sampled_relations
