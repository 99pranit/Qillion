import os
import json
import re
from openai import OpenAI


class learningPath:
    def __init__(self, api_provider='groq', query=None, model_name=None, cognitive_level=None, user_purpose=None):
        self.api_provider = api_provider
        self.query = query
        self.model_name = model_name
        self.cognitive_level = cognitive_level
        self.user_purpose = user_purpose
        self.client = None
        self._topic_data = None      # cached result from get_purpose()
        self._learning_path = None   # cached result from get_learning_path()

    def setup_api(self):
        """Initializes the Groq OpenAI client."""
        self.api_key = os.environ.get(self.api_provider.upper() + "_API_KEY")
        if not self.api_key:
            raise ValueError("API Key is missing. Set GROQ_API_KEY env var or pass it to __init__.")

        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )
        print("API setup successful")

    # ── helpers ───────────────────────────────────────────────────────────────

    def _parse_json_reply(self, raw: str) -> dict:
        """Strip markdown fences and parse JSON safely."""
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Fallback: return raw string wrapped in a dict so callers never crash
            return {"raw": cleaned}

    # ── Step 1: identify purpose / extract topic ──────────────────────────────

    def get_path(self) -> str:
        """
        Calls the LLM to extract the core topic, subtopics, domain,
        difficulty hint and keywords from the user query.

        Returns a human-readable string that describes the identified
        learning intent (used in the Streamlit UI confirmation widget).
        """
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    You are an expert educational content analyzer.

                    Given the following inputs, extract and return a structured JSON object containing:
                    - "core_topic": the main subject being asked about
                    - "subtopics": a list of related subtopics or concepts implied by the query
                    - "domain": the broader academic/professional domain
                    - "difficulty_hint": inferred difficulty level from the query phrasing (beginner/intermediate/advanced)
                    - "keywords": key terms to anchor the learning path
                    - "identified_purpose": one sentence describing what the user wants to learn or achieve

                    Inputs:
                    query: {self.query}
                    blooms_taxonomy_level: {self.cognitive_level}
                    user_purpose: {self.user_purpose}

                    Respond ONLY with a valid JSON object. No explanation, no markdown fences.
                    """,
                }
            ],
            model="openai/gpt-oss-120b",
        )

        raw = chat_completion.choices[0].message.content
        self._topic_data = self._parse_json_reply(raw)
        print("[topic_data]", json.dumps(self._topic_data, indent=2))

        # Return the human-readable purpose string for the UI confirmation widget
        return self._topic_data.get(
            "identified_purpose",
            f"Learn about {self._topic_data.get('core_topic', self.query)}"
        )

    # ── Step 2: generate learning path ────────────────────────────────────────

    def get_learning_path(self) -> dict:
        """
        Uses the topic data extracted by get_purpose() to build a structured
        learning path aligned with the user's Bloom's taxonomy level.

        Must be called AFTER get_purpose().

        Returns a dict with keys:
            learning_path_title  – str
            stages               – list of stage dicts, each with:
                stage_number, title, blooms_level, objectives,
                resources, duration, milestone
        """
        if self._topic_data is None:
            raise RuntimeError("Call get_purpose() before get_learning_path().")

        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    You are an expert curriculum designer specializing in personalized learning paths.

                    Using the extracted topic data below, generate a structured learning path that:
                    1. Aligns with the specified Bloom's Taxonomy level ({self.cognitive_level})
                    2. Serves the user's purpose ({self.user_purpose})
                    3. Progresses logically from foundational to advanced concepts

                    Extracted topic data:
                    {json.dumps(self._topic_data, indent=2)}

                    Return a JSON object with:
                    - "learning_path_title": a concise title for this learning journey
                    - "stages": an array of stages, each containing:
                        - "stage_number": integer
                        - "title": stage name
                        - "objectives": list of learning objectives

                    Respond ONLY with a valid JSON object. No explanation, no markdown fences.
                    """,
                }
            ],
            model="openai/gpt-oss-120b",
        )

        raw = chat_completion.choices[0].message.content
        self._learning_path = self._parse_json_reply(raw)
        print("[learning_path]", json.dumps(self._learning_path, indent=2))
        return self._learning_path

    # ── convenience: run both steps in one call ───────────────────────────────

    def run(self) -> dict:
        """
        Convenience method that runs get_purpose() then get_learning_path()
        and returns both results as a dict.

        Returns:
            {
                "purpose":       str,   # human-readable intent string
                "topic_data":    dict,  # raw extraction JSON
                "learning_path": dict   # structured learning path JSON
            }
        """
        purpose = self.get_purpose()
        learning_path = self.get_learning_path()
        return {
            "purpose": purpose,
            "topic_data": self._topic_data,
            "learning_path": learning_path
        }