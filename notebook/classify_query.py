import os
from openai import OpenAI
from tqdm import tqdm
from collections import Counter

class classify_query:
    def __init__(self, api_key=None, persona='multi'):
        # Use provided key or fall back to environment variable
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.client = None
        self.valid_labels = [
            'knowledge', 'comprehension', 'application', 
            'analysis', 'synthesis', 'evaluation'
        ]
        self.persona = persona
        
        # Static prompt blocks
        self.definitions = """
                    1. Knowledge: Recalling facts, terms, basic concepts, or answers without necessarily understanding them
                    2. Comprehension: Demonstrating understanding of facts by interpreting, translating, summarizing, or explaining
                    3. Application: Using learned information in new concrete situations to solve problems
                    4. Analysis: Breaking down information into parts, examining relationships, distinguishing facts from inferences
                    5. Synthesis: Combining elements to form a new whole, proposing solutions, or designing new approaches
                    6. Evaluation: Making judgments based on criteria and standards through checking and critiquing
        """
        
        self.examples = """
                    1. What is the capital of France? → Knowledge
                    2. Use Ohm’s law to calculate the current in a circuit. → Application
                    3. Critique the author’s argument in the article. → Evaluation
                    4. Summarize the main idea of the passage. → Comprehension             
                    5. Examine the causes of World War I. → Analysis                
                    6. Assess the validity of the research study’s conclusions. → Evaluation
                    7. Propose a plan to reduce plastic pollution in cities. → Synthesis
                    8. Define photosynthesis. → Knowledge
                    9. Apply the concept of supply and demand to predict price changes. → Application              
                    10. Create a new ending for the story. → Synthesis
                    11. Explain in your own words what Newton’s First Law means. → Comprehension
                    12. Identify the relationship between exercise and mental health in the study. → Analysis
        """
        
        self.counterexamples = """
                    1. “List the steps to apply gradient descent.” → is not Knowledge
                    2. “Explain how you would implement this feature in code.” → is not Comprehension
                    3. “Compare quicksort and mergesort on large datasets.” → is not Application
                    4. “Design an experiment to compare two models.” → is not Analysis
                    5. “Evaluate three proposed architectures and pick one.” → is not Synthesis
                    6. “Describe the algorithm’s steps.” → is not Evaluation
        """

    def setup_api(self):
        """Initializes the Groq OpenAI client."""
        if not self.api_key:
            raise ValueError("API Key is missing. Set GROQ_API_KEY env var or pass it to __init__.")
        
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=self.api_key
        )
        print("API setup successful")

    def _generate_prompt_(self, persona, query):
        """Helper to generate specific prompts based on the persona."""
        base_labels = f"Labels: [{', '.join(self.valid_labels)}]"
        persona = persona.lower()

        if persona == 'professor':
            return f"""You are a university professor labeling question based on Bloom's Taxonomy Level.
                    {base_labels}

                    Definations:
                    {self.definitions}

                    Counterexamples:
                    {self.counterexamples}

                    Now classify the following question:
                
                query : {query}"""

        elif persona == 'student':
            return f"""You are a student labeling question based on Bloom's Taxonomy Level.
                    {base_labels}
                
                query : {query}"""

        elif persona in ['psychologist', 'psychiatrist']:
            return f"""You are a psychiatrist labeling question based on Bloom's Taxonomy Level.
                    {base_labels}

                    Examples:
                    {self.examples}

                    Counterexamples:
                    {self.counterexamples}

                    Now classify the question into one Bloom’s Taxonomy level:
                
                query : {query}"""

        elif persona == 'engineer':
            return f"""You are a engineer labeling question based on Bloom's Taxonomy Level.
                    {base_labels}

                    Definations:
                    {self.definitions}

                    Examples:
                    {self.examples}

                    Now classify the question into one Bloom’s Taxonomy level:
                
                query : {query}"""

        elif persona == 'examiner':
            return f"""You are a examiner labeling question.
                    {base_labels}

                    Definations:
                    {self.definitions}

                    Now classify the following question:
                
                query : {query}"""
        
        else:
            return f"""Classify the following question based on Bloom's Taxonomy Level: {base_labels}. Query: {query}"""
        
    def get_ensemble_label(self, query, model_name="openai/gpt-oss-120b"):
        """
        Classifies a query using multiple personas and returns the majority vote label.
        Based on the MPET (Multi-Persona Ensemble Technique) from the notebook.
        """
        personas = ['professor', 'student', 'psychologist', 'engineer', 'examiner']
        
        votes = []
        # Gather votes from all personas
        for persona in personas:
            label = self.get_label(query, model_name=model_name, persona=persona)
            if label not in ["unknown", "error"]:
                votes.append(label)
        
        if not votes:
            return "unknown"

        # Apply Majority Vote (MPET Logic)
        vote_counts = Counter(votes)
        top_label, count = vote_counts.most_common(1)[0]
        
        return {
            "final_label": top_label,
            "confidence": count / len(votes)
        }

    def get_label(self, query, model_name="openai/gpt-oss-120b", persona="student"):
        """Classifies a single query using a specific persona."""
        if not self.client:
            self.setup_api()

        prompt_content = self._generate_prompt_(persona, query)

        try:
            chat_completion = self.client.chat.completions.create(
                messages=[{"role": "user", "content": prompt_content}],
                model=model_name,
            )
            reply = chat_completion.choices[0].message.content.lower().strip()

            # Self-correction loop
            retries = 0
            max_retries = 3
            correction_model = "openai/gpt-oss-20b"

            while reply not in self.valid_labels and retries < max_retries:
                extraction_prompt = f"""Extract answer from previous reponse only in one word without punctuation from: 
                        [{' , '.join(self.valid_labels)}]
                    previous response : {reply}"""
                
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": extraction_prompt}],
                    model=correction_model,
                )
                
                reply = chat_completion.choices[0].message.content.lower().strip()
                import string
                reply = reply.translate(str.maketrans('', '', string.punctuation))
                retries += 1

            return reply if reply in self.valid_labels else "unknown"

        except Exception as e:
            print(f"Error processing query: {e}")
            return "error"

if __name__ == "__main__":
    # 1. Setup and Initialization
    # Ensure you have your API key set in your environment variables: export GROQ_API_KEY="your_key_here"
    api_key = os.environ.get("GROQ_API_KEY")
    
    if not api_key:
        print("Please set the GROQ_API_KEY environment variable.")
    else:
        classifier = classify_query(api_key=api_key)

        sample_queries = input('Enter query to classify')
        persona = input('Enter persona of classification')

        if persona == 'multi':
            print("--- Single Persona Classification (Student) ---")
            results = classifier.get_label(sample_queries, persona="student")
            for q, r in zip(sample_queries, results):
                print(f"Q: {q}\nLabel: {r}\n")
        else:
            print("\n--- MPET Ensemble Classification (Majority Vote) ---")
            test_query = "Propose a plan to reduce plastic pollution in cities."
            ensemble_result = classifier.get_ensemble_label(test_query)
        
            print(f"Q: {test_query}")
            print(f"Consensus Label: {ensemble_result['final_label'].upper()}")
            print(f"Confidence: {ensemble_result['confidence']:.2%}")