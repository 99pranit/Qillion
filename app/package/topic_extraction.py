import os
from openai import OpenAI

class identifyPurpose:
    def __init__(self, api_provider='groq', query=None, model_name=None, cognitive_level=None, user_purpose=None):
        self.api_provider = api_provider
        self.query=query
        self.model_name=model_name
        self.cognitive_level=cognitive_level
        self.user_purpose=user_purpose
        
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
        
    def get_purpose(self):
        # Topic Extraction
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"""
            You are an expert educational content analyzer.

            Given the following inputs, extract and return a structured JSON object containing:
            - "core_topic": the main subject being asked about
            - "subtopics": a list of 3-5 related subtopics or concepts implied by the query
            - "domain": the broader academic/professional domain (e.g., "Computer Science", "Biology")
            - "difficulty_hint": inferred difficulty level from the query phrasing (beginner/intermediate/advanced)
            - "keywords": key terms to anchor the learning path

            Inputs:
            query: {self.query}
            blooms_taxonomy_level: {self.cognitive_level}
            user_purpose: {self.user_purpose}

            Respond ONLY with a valid JSON object. No explanation.
        """,
                }
            ],
            model="openai/gpt-oss-120b",
        )

        reply = chat_completion.choices[0].message.content.lower()
        print(reply)

        # Learning Path Generator
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
            {reply}

            Return a JSON object with:
            - "learning_path_title": a concise title for this learning journey
            - "stages": an array of stages, each containing:
                - "stage_number": integer
                - "title": stage name
                - "objectives": list of learning objectives

            Respond ONLY with a valid JSON object. No explanation.
        """,
                    }
                ],
                model="openai/gpt-oss-120b",
            )
        return reply
