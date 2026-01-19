# agent/planner/planner.py

import time
import json
import re
from typing import Any, Dict, List, Union, Tuple
import jsonschema
from agent.models.llm import ask
from collections import deque

class Planner:
    def __init__(self, tools: Dict[str, Dict[str, Any]], max_retries: int = 3):
        self.tools = {name: props for name, props in tools.items() if name != "none"}
        self.max_retries = max_retries

    def _extract_json_plan(self, response: str) -> List[Dict[str, Any]]:
        """LLM çıktısından JSON nesneleri listesini güvenli bir şekilde çıkarır."""
        match = re.search(r"```json\n(.*?)\n```", response, re.DOTALL)
        if match:
            json_str = match.group(1)
        else:
            start = response.find('[')
            end = response.rfind(']')
            if start != -1 and end != -1 and end > start:
                json_str = response[start:end+1]
            else:
                raise json.JSONDecodeError("Yanıt içinde geçerli bir JSON listesi bulunamadı.", response, 0)

        try:
            plan = json.loads(json_str)
            if isinstance(plan, list) and all(isinstance(item, dict) and 'tool_name' in item and 'args' in item for item in plan):
                return plan
            else:
                raise TypeError(f"JSON verisi, her biri 'tool_name' ve 'args' içeren bir nesne listesi olmalıdır. Alınan plan: {plan}")
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(f"Ayıklanan '{json_str}' string'i JSON olarak ayrıştırılamadı. Hata: {e.msg}", response, e.pos)

    def _is_tool_creation_needed(self, goal: str) -> bool:
        """
        Kullanıcının hedefinin yeni bir araç oluşturmayı gerektirip gerektirmediğini belirlemek için basit bir LLM çağrısı kullanır.
        """
        if "tool_creator" not in self.tools:
            return False

        prompt = f"""
        You are a decision-making AI. Your task is to determine if a user's request requires creating a new tool.
        A new tool is needed if the request is a specific, reusable coding task, like "write a Python script to do X," "create a function for Y," or "use library Z to analyze data."
        A new tool is NOT needed for general questions, research, file editing, or simple, one-off commands.

        Analyze the user's goal below. Respond with only "true" or "false".

        User Goal: "{goal}"

        Does this goal require creating a new tool? (true/false):
        """
        try:
            response = ask(prompt, max_new_tokens=10).strip().lower()
            print(f"[Planner] Araç oluşturma kontrolü. Hedef: '{goal}'. Yanıt: '{response}'")
            return "true" in response
        except Exception as e:
            print(f"[Planner] Araç oluşturma kontrolü sırasında LLM hatası: {e}")
            return False

    def plan(self, goal: str) -> List[Dict[str, Any]]:
        """
        Kullanıcı hedefine göre bir araç zinciri planı oluşturur.
        """
        # Adım 1: Görevin doğrudan bir araç oluşturma isteği olup olmadığını kontrol et
        if self._is_tool_creation_needed(goal):
            print("[Planner] Karar: Yeni araç oluşturulması gerekiyor. `tool_creator` planı oluşturuluyor.")
            return [{
                "tool_name": "tool_creator",
                "args": {
                    "task_description": goal
                }
            }]

        # Adım 2: Araç oluşturma gerekmiyorsa, genel planlama mantığına devam et
        print("[Planner] Karar: Genel planlama mantığı kullanılıyor.")
        tools_string = "\n".join([
            f'- `{name}`: {props["description"]} (Argümanlar: {props.get("args_schema", "Bilinmiyor")})'
            for name, props in self.tools.items()
        ])

        prompt = f"""
You are an expert planner AI. Your job is to create a step-by-step plan to achieve a user's goal.
Your response MUST be a JSON list of objects, where each object represents a tool to be executed.

**CRITICAL RULES - FOLLOW THESE EXACTLY:**

1.  **Analyze the Goal:**
    *   First, understand the user's core intent. What are they trying to achieve?
    *   Break down complex goals into a logical sequence of tool calls.

2.  **Use the Scratchpad (Working Memory):**
    *   For any multi-step task that requires passing data between steps, you MUST use the `working_memory` tool.
    *   Use `working_memory` with `action: "set"` to store intermediate results.
    *   Use `working_memory` with `action: "get"` to retrieve those results in a later step.
    *   **DO NOT** use `code_editor` to create temporary files. The `code_editor` tool is ONLY for when the user explicitly asks to create or modify a permanent file.

3.  **Financial Analysis Rule:**
    *   If the user's goal is to find, discover, research, or get suggestions for financial assets (like stocks, cryptocurrencies, coins, etc.), you **MUST** use the `find_assets` tool as the first step.
    *   When using `find_assets`, its `query` argument **MUST** be the user's full, original request. Use the `"{{user_goal}}"` placeholder for this. **DO NOT** summarize, shorten, or extract keywords from the user's goal for this tool.
    *   The output of `find_assets` will be a list of asset tickers.
    *   You can then pass these tickers to other tools like `comprehensive_financial_analyst` or `price_forecaster`.
    *   **DO NOT** invent or hallucinate asset tickers (e.g., 'coin1', 'stock_abc'). Always discover them first with `find_assets`.

4.  **The `chat` Tool is ONLY for Clarification:**
    *   You can use the `chat` tool, but ONLY as the VERY FIRST STEP, and ONLY if the user's goal is ambiguous or missing critical information.
    *   **NEVER** use the `chat` tool to ask a question that is a rephrasing of the user's goal. Your purpose is to find the answer, not to ask the user for it.

5.  **Construct the Plan:**
    *   Your output MUST be a valid JSON list of objects.
    *   Each object must have `"tool_name"` (string) and `"args"` (object).
    *   **You MUST provide all required arguments for a tool.**
    *   Use `"{{previous_tool_output}}"` to pass the result of the previous step.
    *   Use `"{{user_goal}}"` to refer to the original user request.
    *   **NEVER** invent file paths. If you need to write a file, use a descriptive name based on the user's goal (e.g., `cybersecurity_summary.txt`).

6.  **Self-Correction:**
    *   Before outputting the final JSON, review your plan. Does it violate any of the rules?
    *   If the plan is bad, scrap it and generate a new one that follows the rules.

**EXAMPLE of a GOOD PLAN (using working_memory):**
User Goal: "Research <topic> and write a summary to a file named <filename>.txt"
```json
[
    {{
        "tool_name": "internet_search",
        "args": {{
            "query": "<topic>"
        }}
    }},
    {{
        "tool_name": "working_memory",
        "args": {{
            "action": "set",
            "key": "summary_key",
            "value": "{{previous_tool_output}}"
        }}
    }},
    {{
        "tool_name": "code_editor",
        "args": {{
            "action": "rewrite_file",
            "file_path": "<filename>.txt",
            "new_content": "{{working_memory.get('summary_key')}}"
        }}
    }}
]
```

User Goal: "{goal}"

Available Tools:
{tools_string}

Plan (You must respond with ONLY a valid JSON list of objects, with no other text or explanation):
"""
        response = ask(prompt, max_new_tokens=512)

        try:
            plan = self._extract_json_plan(response)
            print(f"[Planner] LLM tabanlı plan oluşturuldu: {json.dumps(plan, indent=2)}")
            return plan
        except (json.JSONDecodeError, TypeError) as e:
            print(f"[Planner] LLM planı başarısız: {e}. Ham yanıt: {response}")
            raise RuntimeError(f"LLM'den geçerli bir plan alınamadı: {response}")

    def validate_plan(self, plan: List[Dict[str, Any]]):
        """
        Oluşturulan planı, araçların varlığı, argüman şeması ve yer tutucu kullanımı açısından doğrular.
        """
        if not plan:
            raise ValueError("Plan boş olamaz.")

        for i, step in enumerate(plan):
            tool_name = step.get("tool_name")
            if not tool_name:
                raise ValueError(f"Planın {i}. adımı 'tool_name' içermiyor.")

            if tool_name not in self.tools:
                raise ValueError(f"Plandaki '{tool_name}' aracı bulunamadı.")

            tool = self.tools[tool_name]
            args = step.get("args", {})

            # İlk adımda yer tutucu kullanımını kontrol et
            if i == 0:
                for key, value in args.items():
                    if isinstance(value, str) and "{{previous_tool_output}}" in value:
                        raise ValueError("İlk adımda '{{previous_tool_output}}' yer tutucusu kullanılamaz.")

            # Argümanları şemaya göre doğrula
            schema = tool.get("args_schema")
            if schema:
                try:
                    # Pydantic modelleri için .schema() çağrısını ele al (varsayımsal)
                    if hasattr(schema, 'schema') and callable(schema.schema):
                        schema_dict = schema.schema()
                    else:
                        schema_dict = schema

                    # Yer tutucu değerlerini doğrulamadan önce geçici olarak kaldır
                    args_to_validate = {}
                    for k, v in args.items():
                        if isinstance(v, str) and "{{" in v and "}}" in v:
                            # Bu bir yer tutucu, şimdilik atla
                            pass
                        else:
                            args_to_validate[k] = v

                    jsonschema.validate(instance=args_to_validate, schema=schema_dict)

                except jsonschema.exceptions.ValidationError as e:
                    raise ValueError(f"'{tool_name}' aracı için geçersiz argümanlar: {e.message}")
                except Exception as e:
                    raise ValueError(f"'{tool_name}' aracı için argüman şeması doğrulanırken bir hata oluştu: {e}")

    def execute(self, plan: List[Dict[str, Any]], input_data: str, agent_instance=None) -> Dict[str, Any]:
        """
        Executes a plan step by step, replacing placeholders and managing outputs.
        """
        last_result: Dict[str, Any] = {}
        previous_tool_output: Any = None

        for i, step in enumerate(plan):
            tool_name = step.get("tool_name")
            args = step.get("args", {})

            if not tool_name:
                raise RuntimeError(f"Step {i} of the plan is missing a 'tool_name'.")

            print(f"[Planner] Step {i+1}: Executing tool '{tool_name}'...")

            if tool_name not in self.tools:
                raise RuntimeError(f"Unknown tool: '{tool_name}'.")

            tool_func = self.tools[tool_name]["func"]

            # Agent örneğini araca iletmek için kontrol
            tool_kwargs = {}
            if agent_instance:
                tool_kwargs['agent_instance'] = agent_instance

            processed_args = {}
            for key, value in args.items():
                if isinstance(value, str):
                    # Handle {{previous_tool_output}}
                    if value in ("{{previous_tool_output}}", "{previous_tool_output}"):
                        if i > 0 and previous_tool_output is None:
                            print(f"[Planner Warning] Argument '{key}' for tool '{tool_name}' expected the output of the previous tool, but it was None.")
                        processed_args[key] = previous_tool_output
                    # Handle {{user_goal}}
                    elif value in ("{{user_goal}}", "{user_goal}"):
                        processed_args[key] = input_data
                    # Handle {{working_memory.get('some_key')}}
                    elif '{{working_memory.get' in value:
                        match = re.search(r"working_memory\.get\('([^']+)'\)", value) # Bu kısım şimdilik korunuyor, ancak agent hafızasına geçilebilir.
                        if match and agent_instance and hasattr(agent_instance, 'working_memory'):
                            wm_key = match.group(1)
                            retrieved_value = agent_instance.working_memory.get(wm_key, "")
                            # Yer tutucuyu dizede değiştir
                            processed_args[key] = value.replace(f"{{{{working_memory.get('{wm_key}')}}}}", retrieved_value)
                        else:
                            print(f"[Planner Warning] Could not parse working_memory key from '{value}' or working_memory tool is not available.")
                            processed_args[key] = value
                    else:
                        processed_args[key] = value
                else:
                    processed_args[key] = value

            try:
                print(f"[Planner] Arguments: {processed_args}")
                # Araca hem argümanları hem de agent örneğini gönder
                if isinstance(processed_args, dict):
                    current_result = tool_func(**processed_args, **tool_kwargs)
                else: # Eğer argümanlar bir string ise
                    current_result = tool_func(processed_args, **tool_kwargs)

                if not isinstance(current_result, dict) or "status" not in current_result:
                     raise TypeError(f"Tool '{tool_name}' did not return a standard response format ({{'status': '...', ...}}). Got: {current_result}")

                if current_result.get("status") == "clarification_needed":
                    print(f"[Planner] Plan halted: Clarification needed from user.")
                    return current_result

                if current_result.get("status") != "success":
                    message = current_result.get("message", "Unknown execution error.")
                    raise RuntimeError(message)

                last_result = current_result
                previous_tool_output = last_result.get("result")
                print(f"[Planner] '{tool_name}' executed successfully.")

            except Exception as e:
                raise RuntimeError(f"Error executing tool '{tool_name}': {str(e)}")

        return last_result if last_result else {"status": "success", "message": "Plan executed successfully, but no tools returned a result."}

    def plan_and_execute(self, goal: str) -> Dict[str, Any]:
        """
        Hedef için plan oluşturur ve çalıştırır. Hata durumunda yeniden planlamayı dener.
        """
        plan: List[Dict[str, Any]] = []
        retries = 0
        current_goal = goal
        last_error = ""

        while retries <= self.max_retries:
            try:
                if not plan:
                    plan = self.plan(current_goal)
                    self.validate_plan(plan) # Planı oluşturduktan hemen sonra doğrula

                print(f"[Planner] Deneme {retries + 1}/{self.max_retries + 1} - Plan: {json.dumps(plan, indent=2)}")
                result = self.execute(plan, goal)

                if isinstance(result, dict) and result.get("status") == "clarification_needed":
                    return result

                if isinstance(result, dict) and result.get("status") == "success":
                    result["retries"] = retries
                    return result
                else:
                    last_error = result.get("message", "Bilinmeyen bir yürütme hatası.")
                    raise RuntimeError(last_error)

            except Exception as e:
                last_error = str(e)
                print(f"[Planner Retry {retries + 1}] Planda veya yürütmede hata: {last_error}")

                if retries == self.max_retries:
                    break

                current_goal = f"'{goal}' hedefine ulaşmaya çalışırken bir önceki denemede şu hatayı aldım: '{last_error}'. Lütfen bu hatayı düzeltecek farklı bir plan oluştur."
                plan = []
                retries += 1
                time.sleep(1)

        return {"status": "error", "message": f"Planner tüm denemelerde başarısız oldu. Son hata: {last_error}"}