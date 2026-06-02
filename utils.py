import json
from collections import defaultdict
from openai import OpenAI, APITimeoutError, APIConnectionError
import time
import os
import re

from prompts import llm_system_prompt, readout_header_template


_MODEL_METADATA_CACHE = {}


def _infer_chat_capability(client, engine: str, llm_name: str) -> bool:
    cache_key = (getattr(client, "base_url", None), engine, llm_name)
    if cache_key in _MODEL_METADATA_CACHE:
        return _MODEL_METADATA_CACHE[cache_key]

    chat_like = any(tok in llm_name for tok in ["chat", "instruct", "gpt", "o1", "o3", "o4"])

    try:
        model_list = client.models.list()
        for model in getattr(model_list, "data", []):
            model_id = str(getattr(model, "id", "") or "")
            if model_id != engine:
                continue
            root = str(getattr(model, "root", "") or "").lower()
            if any(tok in root for tok in ["chat", "instruct"]):
                chat_like = True
            break
    except Exception:
        pass

    _MODEL_METADATA_CACHE[cache_key] = chat_like
    return chat_like


def run_llm(prompt: str, args, history: list = None, retry_prompt: str = None) -> str:
    llm_name = args.llm.lower()
    if "llama" in args.llm.lower():
        base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
        if not base_url:
            raise ValueError(
                "Missing OPENAI_BASE_URL for local/open-source LLM backends."
            )
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"), base_url=base_url)
        engine = args.llm
    else:
        api_key = getattr(args, "api_key", "") or os.environ.get("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Missing OpenAI API key. Pass --api_key or export OPENAI_API_KEY."
            )
        client = OpenAI(api_key=api_key)
        engine = args.llm
    use_plain_completion = ("llama-2" in llm_name) and not _infer_chat_capability(client, engine, llm_name)
    temperature = args.temperature
    max_tokens = args.limit_llm_out
    messages = [{"role": "system", "content": llm_system_prompt}]
    messages.append({"role": "user", "content": prompt})

    if history is not None:
        temperature = min(1, args.temperature + 0.2 * len(history))
        max_tokens = args.limit_llm_out * 2
        input_budget = int(getattr(args, "limit_llm_in", 8192))
        if use_plain_completion:
            input_budget = min(input_budget, max(512, 4096 - int(max_tokens) - 128))
        if token_count(prompt + retry_prompt + history[-1]) < input_budget:
            messages.append({"role": "assistant", "content": history[-1]})
        messages.append({"role": "user", "content": retry_prompt})

    def _build_completion_prompt(messages_: list[dict]) -> str:
        role_map = {
            "system": "SYSTEM",
            "user": "USER",
            "assistant": "ASSISTANT",
        }
        rendered = []
        for msg in messages_:
            role = role_map.get(msg.get("role", "user"), "USER")
            content = str(msg.get("content", "")).strip()
            rendered.append(f"{role}:\n{content}")
        rendered.append("ASSISTANT:\n")
        return "\n\n".join(rendered)

    def _call_llm(messages_, temperature_, max_tokens_):
        if use_plain_completion:
            request_kwargs = dict(
                model=engine,
                prompt=_build_completion_prompt(messages_),
                temperature=temperature_,
                max_tokens=max_tokens_,
                frequency_penalty=0,
                presence_penalty=0,
                stop=[
                    "\nSYSTEM:",
                    "\nUSER:",
                    "\nASSISTANT:",
                ],
            )
            return client.completions.create(**request_kwargs)

        request_kwargs = dict(
            model=engine,
            messages=messages_,
            temperature=temperature_,
            max_tokens=max_tokens_,
            frequency_penalty=0,
            presence_penalty=0,
        )
        return client.chat.completions.create(**request_kwargs)

    try:
        response = _call_llm(messages, temperature, max_tokens)
    except (APITimeoutError, APIConnectionError):
        time.sleep(10)
        response = _call_llm(messages, temperature, max_tokens)
               
    if use_plain_completion:
        result = response.choices[0].text
    else:
        result = response.choices[0].message.content

    return result


def _fmt_chunk(values: list[str]) -> str:
    vals = [str(v).strip() for v in values if str(v).strip()]
    if not vals:
        return "chunk()"
    if len(vals) == 1:
        return vals[0]
    return f"chunk({', '.join(vals)})"


def _collect_chunks(chunk_info_list: list, final_state) -> list[dict]:
    chunks = []
    for info in chunk_info_list:
        for local_idx in range(info["K_act"]):
            global_idx = info["global_start"] + local_idx
            tails = info["tail_names_per_chunk"][local_idx]
            if not tails:
                continue
            chunks.append({
                "hop": info["hop"],
                "relation": info["relation"],
                "rel_short": info["relation"].rsplit("->", 1)[-1],
                "tails": tails,
                "global_idx": global_idx,
                "active": bool(final_state.active[global_idx]),
            })
    return chunks


def build_chunk_readout(
    chunk_info_list: list,
    final_state,
    topic_name: str,
    graph_triples: list,
    limit_llm_in: int,
) -> str:
    active = [c for c in _collect_chunks(chunk_info_list, final_state) if c["active"]]
    hop1 = [c for c in active if c["hop"] == 1]
    hop2 = [c for c in active if c["hop"] == 2]
    hop3 = [c for c in active if c["hop"] == 3]

    rtd: dict[tuple[str, str], list[str]] = defaultdict(list)
    for triple in graph_triples:
        if len(triple) < 3:
            continue
        h = str(triple[0]).strip()
        r = str(triple[1]).strip()
        t = str(triple[2]).strip()
        if not h or not r or not t:
            continue
        key = (r.lower(), t.lower())
        if h not in rtd[key]:
            rtd[key].append(h)

    def _group_by_head(rel_short: str, tails: list[str], allowed_heads_lower: set[str]):
        h2t: dict[str, list[str]] = defaultdict(list)
        for tail in tails:
            heads = rtd.get((rel_short.lower(), str(tail).lower()), [])
            for head in heads:
                if head.lower() in allowed_heads_lower and tail not in h2t[head]:
                    h2t[head].append(tail)
        return dict(h2t)

    def _emit_grouped(
        h2t: dict[str, list[str]],
        rel_short: str,
        parent_label: str,
        indent: str,
        start_idx: int,
    ):
        lines = []
        entries = []
        idx = start_idx
        if not h2t:
            return lines, entries, idx

        grouped: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for head, tails in h2t.items():
            grouped[tuple(tails)].append(head)

        for tails_tuple, heads in grouped.items():
            label = f"{parent_label}.{idx}"
            lines.append(
                f"\n{indent}{label}. {_fmt_chunk(heads)} -> {rel_short} -> {_fmt_chunk(list(tails_tuple))}"
            )
            entries.append((label, {t.lower() for t in tails_tuple}))
            idx += 1
        return lines, entries, idx

    lines = [readout_header_template.format(topic_name)]

    for i, c1 in enumerate(hop1):
        lines.append(f"\n{i+1}. {topic_name} -> {c1['rel_short']} -> {_fmt_chunk(c1['tails'])}")
        hop1_tails_lower = {str(t).lower() for t in c1["tails"]}
        j = 1

        for c2 in hop2:
            if not c2["relation"].startswith(c1["relation"] + "->"):
                continue
            h2t_2 = _group_by_head(c2["rel_short"], c2["tails"], hop1_tails_lower)
            l2, entries_2, j = _emit_grouped(h2t_2, c2["rel_short"], str(i + 1), "\t", j)
            lines.extend(l2)

            for parent_label_2, hop2_tails_lower in entries_2:
                k = 1
                for c3 in hop3:
                    if not c3["relation"].startswith(c2["relation"] + "->"):
                        continue
                    h2t_3 = _group_by_head(c3["rel_short"], c3["tails"], hop2_tails_lower)
                    l3, _, k = _emit_grouped(h2t_3, c3["rel_short"], parent_label_2, "\t\t", k)
                    lines.extend(l3)

        lines.append("\n")

    text = "".join(lines)
    while token_count(text) > limit_llm_in:
        text = text.rsplit("\n", 1)[0]
    return text
    
def save_2_jsonl(file_name: str, output: dict):
    with open(file_name, "a") as outfile:
        json_str = json.dumps(output)
        outfile.write(json_str + "\n")

def get_list_str(string: str) -> list:
    text = str(string or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return []

    lines = [ln.strip() for ln in text.replace('\n\t', ' ').split('\n')]
    items = []
    for line in lines:
        if not line:
            continue
        if re.match(r"^(?:\d+[\.\)]|[-*])\s+", line):
            item = re.sub(r"^(?:\d+[\.\)]|[-*])\s+", "", line).strip()
            item = re.sub(r"^\d+[\.\)]\s*", "", item).strip()
            if not item:
                continue
            lower = item.lower()
            if lower.startswith(("certainly", "sure", "here are", "based on the", "the answer is")) and len(item.split()) > 6:
                continue
            if re.fullmatch(r"\d+[\.\)]?", item):
                continue
            items.append(item)

    if items:
        return items

    matches = re.findall(r'(?:^|\n)\s*(?:\d+[\.\)]|[-*])\s+(.*?)(?=\n\s*(?:\d+[\.\)]|[-*])\s+|$)', text, re.DOTALL)
    items = []
    for match in matches:
        item = re.sub(r"\s+", " ", match).strip()
        item = re.sub(r"^\d+[\.\)]\s*", "", item).strip()
        if not item or re.fullmatch(r"\d+[\.\)]?", item):
            continue
        items.append(item)
    return items


def token_count(text: str) -> float:
    punctuation = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
    number = set('0123456789')

    n_tokens = len("".join(i for i in text if i in punctuation)) 
    text = "".join(i for i in text if i not in punctuation)

    n_tokens += len("".join(i for i in text if i in number)) / 2
    text  = "".join(i for i in text if i not in number)

    n_tokens += len(text) / 4

    return n_tokens
