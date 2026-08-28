#!/usr/bin/env python3
"""Run Friday's held-out voice and text output-quality scorecard."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from openai import DefaultHttpxClient, OpenAI


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from friday_core.conversation_evals import ConversationQualityEvalRunner
from friday_core.conversation import (
    FAST_CONVERSATION_TEMPERATURE,
    FAST_CONVERSATION_TOP_P,
)
from friday_core.live_runtime import (
    read_live_runtime,
    read_local_model_credential,
)


def main() -> int:
    runtime = read_live_runtime(REPO)
    client = OpenAI(
        base_url=runtime.base_url,
        api_key=read_local_model_credential(REPO), timeout=90,
        http_client=DefaultHttpxClient(
            trust_env=False, follow_redirects=False))

    def complete(system: str, user: str) -> str:
        response = client.chat.completions.create(
            model=runtime.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=FAST_CONVERSATION_TEMPERATURE,
            top_p=FAST_CONVERSATION_TOP_P, max_tokens=360,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        return response.choices[0].message.content or ""

    result = ConversationQualityEvalRunner(
        complete, model=runtime.model,
        runtime_fingerprint=runtime.fingerprint).run(
            REPO / "evals" / "conversation-quality-v1.json")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
