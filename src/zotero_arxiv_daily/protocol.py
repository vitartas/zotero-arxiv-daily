from dataclasses import dataclass
from typing import Optional, TypeVar
from datetime import datetime
import re
import tiktoken
from openai import OpenAI
from loguru import logger
import json
import os
import subprocess
from tempfile import TemporaryDirectory
RawPaperItem = TypeVar('RawPaperItem')

# A leading "TLDR:" / "**TL;DR:**" label added by the model; the email template already labels the TLDR.
_TLDR_LABEL = re.compile(r"^\s*(?:\*\*|__)?\s*TL;?\s?DR\s*(?:\*\*|__)?\s*[:\-–—]?\s*(?:\*\*|__)?\s*", flags=re.IGNORECASE)


def _request_claude_code(llm_params: dict, messages: list[dict]) -> str:
    """Query Claude through the Claude Code CLI (`claude -p`), billed to the logged-in Claude subscription.

    Locally this uses the existing `claude` login; in CI it uses CLAUDE_CODE_OAUTH_TOKEN (from `claude setup-token`).
    """
    cc_params = llm_params.get("claude_code") or {}
    model = (llm_params.get("generation_kwargs") or {}).get("model")
    system_prompt = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    user_prompt = "\n\n".join(m["content"] for m in messages if m["role"] != "system")

    cmd = [
        cc_params.get("cli_path") or "claude", "-p",
        "--output-format", "json",
        "--tools", "",
        "--no-session-persistence",
        "--strict-mcp-config",
        "--safe-mode",
    ]
    if model:
        cmd += ["--model", model]
    if system_prompt:
        cmd += ["--system-prompt", system_prompt]
    if cc_params.get("fallback_model"):
        cmd += ["--fallback-model", cc_params["fallback_model"]]
    if cc_params.get("effort"):
        cmd += ["--effort", cc_params["effort"]]

    # An API key would take precedence over the subscription login, so never pass one on.
    env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
    # Run outside the repo so no project CLAUDE.md or settings get loaded into the context.
    with TemporaryDirectory() as cwd:
        proc = subprocess.run(
            cmd,
            input=user_prompt,
            capture_output=True,
            text=True,
            env=env,
            cwd=cwd,
            timeout=cc_params.get("timeout") or 600,
        )
    try:
        output = json.loads(proc.stdout)
    except json.JSONDecodeError:
        raise RuntimeError(f"claude exited with code {proc.returncode}: {(proc.stderr or proc.stdout).strip()}")
    if proc.returncode != 0 or output.get("is_error"):
        raise RuntimeError(f"claude returned an error: {output.get('result') or proc.stderr.strip()}")
    return output["result"]


def _request_llm(openai_client: OpenAI | None, llm_params: dict, messages: list[dict]) -> str:
    api_mode = llm_params.get("api_mode", "chat_completion")

    if api_mode == "claude_code":
        return _request_claude_code(llm_params, messages)

    generation_kwargs = dict(llm_params.get("generation_kwargs", {}))

    if api_mode == "chat_completion":
        response = openai_client.chat.completions.create(
            messages=messages,
            **generation_kwargs,
        )
        return response.choices[0].message.content

    if api_mode == "response":
        max_tokens = generation_kwargs.pop("max_tokens", None)
        if max_tokens is not None and "max_output_tokens" not in generation_kwargs:
            generation_kwargs["max_output_tokens"] = max_tokens
        response = openai_client.responses.create(
            input=messages,
            **generation_kwargs,
        )
        return response.output_text

    raise ValueError(
        f"Unsupported llm.api_mode: {api_mode}. "
        "Expected 'chat_completion', 'response' or 'claude_code'."
    )


@dataclass
class Paper:
    source: str
    title: str
    authors: list[str]
    abstract: str
    url: str
    pdf_url: Optional[str] = None
    full_text: Optional[str] = None
    tldr: Optional[str] = None
    affiliations: Optional[list[str]] = None
    score: Optional[float] = None

    def _generate_tldr_with_llm(self, openai_client:OpenAI,llm_params:dict) -> str:
        lang = llm_params.get('language', 'English')
        prompt = f"Given the following information of a paper, generate a one-sentence TLDR summary in {lang}:\n\n"
        if self.title:
            prompt += f"Title:\n {self.title}\n\n"

        if self.abstract:
            prompt += f"Abstract: {self.abstract}\n\n"

        if self.full_text:
            prompt += f"Preview of main content:\n {self.full_text}\n\n"

        if not self.full_text and not self.abstract:
            logger.warning(f"Neither full text nor abstract is provided for {self.url}")
            return "Failed to generate TLDR. Neither full text nor abstract is provided"
        
        # use gpt-4o tokenizer for estimation
        enc = tiktoken.encoding_for_model("gpt-4o")
        prompt_tokens = enc.encode(prompt)
        prompt_tokens = prompt_tokens[:4000]  # truncate to 4000 tokens
        prompt = enc.decode(prompt_tokens)
        
        tldr = _request_llm(
            openai_client,
            llm_params,
            [
                {
                    "role": "system",
                    "content": f"You are an assistant who perfectly summarizes scientific paper, and gives the core idea of the paper to the user. Your answer should be in {lang}. Reply with the summary only, as plain text: no title, label or markdown.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return _TLDR_LABEL.sub("", tldr, count=1).strip()
    
    def generate_tldr(self, openai_client:OpenAI,llm_params:dict) -> str:
        try:
            tldr = self._generate_tldr_with_llm(openai_client,llm_params)
            self.tldr = tldr
            return tldr
        except Exception as e:
            logger.warning(f"Failed to generate tldr of {self.url}: {e}")
            tldr = self.abstract
            self.tldr = tldr
            return tldr

    def _generate_affiliations_with_llm(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        if self.full_text is not None:
            prompt = f"Given the beginning of a paper, extract the affiliations of the authors in a python list format, which is sorted by the author order. If there is no affiliation found, return an empty list '[]':\n\n{self.full_text}"
            # use gpt-4o tokenizer for estimation
            enc = tiktoken.encoding_for_model("gpt-4o")
            prompt_tokens = enc.encode(prompt)
            prompt_tokens = prompt_tokens[:2000]  # truncate to 2000 tokens
            prompt = enc.decode(prompt_tokens)
            affiliations = _request_llm(
                openai_client,
                llm_params,
                [
                    {
                        "role": "system",
                        "content": "You are an assistant who perfectly extracts affiliations of authors from a paper. You should return a python list of affiliations sorted by the author order, like [\"TsingHua University\",\"Peking University\"]. If an affiliation is consisted of multi-level affiliations, like 'Department of Computer Science, TsingHua University', you should return the top-level affiliation 'TsingHua University' only. Do not contain duplicated affiliations. If there is no affiliation found, you should return an empty list [ ]. You should only return the final list of affiliations, and do not return any intermediate results.",
                    },
                    {"role": "user", "content": prompt},
                ],
            )

            affiliations = re.search(r'\[.*?\]', affiliations, flags=re.DOTALL).group(0)
            affiliations = json.loads(affiliations)
            affiliations = list(set(affiliations))
            affiliations = [str(a) for a in affiliations]

            return affiliations
    
    def generate_affiliations(self, openai_client:OpenAI,llm_params:dict) -> Optional[list[str]]:
        try:
            affiliations = self._generate_affiliations_with_llm(openai_client,llm_params)
            self.affiliations = affiliations
            return affiliations
        except Exception as e:
            logger.warning(f"Failed to generate affiliations of {self.url}: {e}")
            self.affiliations = None
            return None
@dataclass
class CorpusPaper:
    title: str
    abstract: str
    added_date: datetime
    paths: list[str]