# Customer Support Agent

A customer support agent built with LangChain that changes its own instructions and available tools as a conversation progresses through a support workflow, based on the state machine pattern.

## Workflow

The agent walks a customer through three stages:

1. **Warranty verification** — asks whether the device is under warranty.
2. **Issue classification** — asks the customer to describe the problem and classifies it as hardware or software.
3. **Resolution** — provides troubleshooting steps or repair instructions, or escalates to a human for paid repairs when the device is out of warranty with a hardware issue.

```
Customer reports an issue
        |
Is the device under warranty?
   /                    \
 Yes                     No
  |                       |
What type of issue?    What type of issue?
  /        \               /        \
Hardware  Software     Hardware   Software
  |          |             |          |
Warranty   Troubleshoot  Escalate   Troubleshoot
repair                   to human
  |          |             |          |
   \_________|_____________|_________/
              |
        Issue resolved
```

Conversation state persists across turns, and the customer can correct earlier answers (for example, changing warranty status) partway through, which sends the workflow back to the relevant earlier step.

## How it works

Rather than using separate agents for each stage, this is a single agent whose system prompt and available tools change depending on where the conversation currently stands. Each stage is really just a different configuration of the same agent.

### State

A custom state schema tracks which step is active plus the information collected so far:

```python
class SupportState(AgentState):
    current_step: NotRequired[SupportStep]  # warranty_collector, issue_classifier, resolution_specialist
    warranty_status: NotRequired[Literal["in_warranty", "out_of_warranty"]]
    issue_type: NotRequired[Literal["hardware", "software"]]
```

### Tools drive the workflow

Tools do not just perform an action, they also move the conversation to the next step by returning a `Command` that updates state:

- `record_warranty_status` — records the warranty answer and advances to issue classification.
- `record_issue_type` — records the issue type and advances to resolution.
- `provide_solution` — gives the customer a solution or repair instructions.
- `escalate_to_human` — hands the case to a human agent with a stated reason.
- `go_back_to_warranty` / `go_back_to_classification` — let the customer correct an earlier answer, sending the workflow back a step.

### Step configuration

Each step has its own prompt and its own set of tools, defined in a single lookup table:

```python
STEP_CONFIG = {
    "warranty_collector": {"prompt": WARRANTY_COLLECTOR_PROMPT, "tools": [record_warranty_status], "requires": []},
    "issue_classifier": {"prompt": ISSUE_CLASSIFIER_PROMPT, "tools": [record_issue_type], "requires": ["warranty_status"]},
    "resolution_specialist": {"prompt": RESOLUTION_SPECIALIST_PROMPT, "tools": [provide_solution, escalate_to_human, go_back_to_warranty, go_back_to_classification], "requires": ["warranty_status", "issue_type"]},
}
```

The `requires` field guards against reaching a step before the information it depends on has been collected.

### Middleware applies the right configuration each turn

A `wrap_model_call` middleware function runs before every model call. It reads `current_step` from state, looks up that step's prompt and tools, checks that any required information has already been collected, formats the prompt with the collected state values, and overrides the request with that configuration before it reaches the model.

### Persistence

The agent is created with an `InMemorySaver` checkpointer, which is what allows `current_step` and the collected answers to survive between separate calls to `agent.invoke`. Without it, the workflow would reset on every message.

### Message history

Because a support conversation can run long, a `SummarizationMiddleware` is included to compress older messages once the conversation passes a token threshold, while keeping the most recent messages in full.

## Setup

```bash
pip install langchain
```

Set the API key environment variable for whichever model provider is used, for example:

```bash
export OPENAI_API_KEY="sk-..."
```

## Running it

The agent is invoked per conversation turn, with a `thread_id` used to keep state tied to a specific conversation:

```python
from langchain_core.utils.uuid import uuid7

thread_id = str(uuid7())
config = {"configurable": {"thread_id": thread_id}}

result = agent.invoke(
    {"messages": [HumanMessage("Hi, my phone screen is cracked")]},
    config,
)
```

Each subsequent call reuses the same `config` so the agent picks up where the conversation left off.

## Files

| File | Responsibility |
|---|---|
| `state.py` | Defines `SupportState`, tracking the current step and collected answers |
| `tools.py` | The workflow tools, including the state transition and correction tools |
| `configurations.py` | Per step prompts and the `STEP_CONFIG` lookup table |
| `main.py` | Middleware (`apply_step_config`) and agent setup, creating the agent with the state schema, middleware, and checkpointer |
| `test.py` | Runs through the workflow across multiple turns to test it end-to-end |
| `requirements.txt` | Python dependencies |
