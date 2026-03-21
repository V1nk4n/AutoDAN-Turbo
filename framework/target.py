import os
import json

class Target():
    '''
    Temporary implementation of Target class
    '''
    def __init__(self, model):
        self.model = model

    def respond(self, prompt):
        # ✅ Giảm max_length để tăng tốc
        response = self.model.generate("You are a helpful assistant.", prompt, max_length=2000, do_sample=True, temperature=0.6, top_p=0.9)
        return response

    def respond_batch(self, prompts, batch_size=2):
        responses = []
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start+batch_size]
            systems = ["You are a helpful assistant."] * len(chunk)
            users = [p["prompt"] for p in chunk]
            chunk_responses = self.model.generate_batch(systems, users, max_length=2000, do_sample=True, temperature=0.6, top_p=0.9)
            responses.extend(chunk_responses)
        return responses

        