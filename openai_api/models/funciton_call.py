import json
import requests
import os
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored

GPT_MODEL = "gpt-4-turbo"


@retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, function_call=None, model=GPT_MODEL):

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.getenv("OPENAI_API_KEY")
    }

    json_data = {
        "model": model,
        "messages": messages
    }

    if functions is not None:
        json_data.update({"functions": functions})

    if function_call is not None:
        json_data.update({"function_call": function_call})

    try:
        rsp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data
        )

        return rsp

    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta"
    }

    for message in messages:
        if message["role"] == "system":
            print(colored(f"system: {message['content']}\n", role_to_color[message["role"]]))

        elif message["role"] == "user":
            print(colored(f"user: {message['content']}\n", role_to_color[message["role"]]))

        elif message["role"] == "assistant" and message.get("function_call"):
            print(colored(f"assistant[function_call]: {message['function_call']}\n", role_to_color[message["role"]]))

        elif message["role"] == "assistant" and not message.get("function_call"):
            print(colored(f"assistant[content]: {message['content']}\n", role_to_color[message["role"]]))

        elif message["role"] == "function":
            print(colored(f"function ({message['name']}): {message['content']}\n", role_to_color[message["role"]]))


funcs = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "format": {
                    "type": "string",
                    "description": "The temperature unit to use. Infer this from the users location.",
                }
            }
        },
        "required": ["location", "format"]
    }, {
        "name": "get_n_day_weather_forecast",
        "description": "Get an N-day weather forecast",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA"
                },
                "format": {
                    "type": "string",
                    "description": "The temperature unit to use. Infer this from the users location.",
                },
                "num_days": {
                    "type": "integer",
                    "description": "The number of days to forecast.",
                }
            }
        },
        "required": ["location", "format", "num_days"]
    }
]


msgs = [{
    "role": "system",
    "content": "Don't make assumptions about what values to plug into functions. "
               "Ask for clarification if a user request is ambiguous."
}, {
    "role": "user",
    "content": "What's the weather like today."
}]
chat_response = chat_completion_request(msgs, functions=funcs)
assistant_message = chat_response.json()["choices"][0]["message"]
msgs.append(assistant_message)
pretty_print_conversation(msgs)


msgs.append({
    "role": "user",
    "content": "What is the weather like in Xiamen, China?"
})
chat_response = chat_completion_request(msgs, functions=funcs)
assistant_message = chat_response.json()["choices"][0]["message"]
msgs.append(assistant_message)
pretty_print_conversation(msgs)


msgs.append({
    "role": "user",
    "content": "What about over the next 7 days?"
})
chat_response = chat_completion_request(msgs, functions=funcs)
assistant_message = chat_response.json()["choices"][0]["message"]
msgs.append(assistant_message)
pretty_print_conversation(msgs)

