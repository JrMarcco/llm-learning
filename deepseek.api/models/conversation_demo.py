import openai
import os

openai.api_key = os.getenv("DEEPSEEK_API_KEY")

class Conversation:
    def __init__(self, prompt, num_of_round):
        self.prompt = prompt
        self.num_of_round = num_of_round
        self.messages = []
        self.messages.append({"role": "system", "content": prompt})
        
    def ask(self, question):
        try:
            self.messages.append({"role": "user", "content": question})
            response = openai.chat.completions.create(
                model="deepseek-chat",
                messages=self.messages,
                max_tokens=2048,
                temperature=0.7,
                stream=False
            )
        except Exception as e:
            print(e)
            return e
        
        message = response.choices[0].message.content
        self.messages.append({"role": "assistant", "content": message})

        if len(self.messages) > self.num_of_round*2 + 1:
            del self.messages[1:3] # remove the first round conversation left.

        return message
    