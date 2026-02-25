import os
from openai import OpenAI

class identifyPurpose:
    def __init__(self, api_provider='groq', query=None, model_name=None, cognitive_level=None):
        self.api_provider = api_provider
        self.query=query
        self.model_name=model_name
        self.cognitive_level=cognitive_level
        self.intent_guidance = {
            "knowledge": "Intents often involve: learning facts, preparing for tests, satisfying curiosity, building foundational understanding",
            "comprehension": "Intents often involve: making informed decisions, explaining to others, understanding implications, grasping meaning",
            "application": "Intents often involve: solving specific problems, completing tasks, implementing solutions, fixing issues",
            "analysis": "Intents often involve: comparing options, identifying patterns, finding root causes, planning strategies",
            "synthesis": "Intents often involve: designing new solutions, creating original work, developing plans, combining ideas",
            "evaluation": "Intents often involve: making judgments, validating decisions, assessing quality, determining best options"
            }
        
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
        
    def get_topics(self):
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"""
                    Analyze what the user wants to DO with this knowledge.
                    Query: {self.query}
                    Cognitive Level: {self.cognitive_level}
                    {self.intent_guidance.get(self.cognitive_level, "Intents often involve: learning, solving problems, or making decisions")}

                    Consider:
                    - Why are they asking this question?
                    - What will they do with the answer?
                    - What is their ultimate goal?

                    Provide a brief, specific statement of their intent (1-2 sentences max).

                    Format: [action verb] + [purpose]
                    Example: "Apply formula to solve homework problem" or "Understand concept to prepare for certification exam"

                    User Intent:""",
                }
            ],
            model=self.model_name,
        )
        reply = chat_completion.choices[0].message.content.lower()

        return reply
