import google.generativeai as genai
import os
from mistralai import Mistral
import functools
import json


GOOGLE_API_KEY=os.environ['GOOGLE_API_KEY']
MISTRAL_API_KEY = os.environ["MISTRAL_API_KEY"]


genai.configure(api_key=GOOGLE_API_KEY)



def add_nums(a: int, b: int):
    """adds numbers a and b together"""
    return a + b

def subtract_nums(a: int, b: int):
    """subtracts number b from number a"""
    return a - b

def multiply_nums(a: int, b: int):
    """multiplies numbers a and b together"""
    return a * b




def gemini_call(input: str):
    instruction = "You are a helpful calculator bot. You can add, subtract, and multiply numbers. Do not perform any other task."

    tool_list = [add_nums, subtract_nums, multiply_nums]
    gemini_model = genai.GenerativeModel(
        "models/gemini-1.5-pro", tools=tool_list, system_instruction=instruction
    )

    # gemini_chat = gemini_model.start_chat() #This will just return information on what function to run
    gemini_chat = gemini_model.start_chat(enable_automatic_function_calling=True) # use this to automatically run the function that gemini decides

    response = gemini_chat.send_message(input)
    print(response.parts)

def mistral_call(input: str):
    model = "mistral-large-latest"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "add_nums",
                "description": "add two numbers together",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "integer",
                            "description": "The first number",
                        },
                        "b": {
                            "type": "integer",
                            "description": "The second number",
                        }
                    },
                    "required": ["a", "b"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "subtract_nums",
                "description": "subtract two numbers together",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "integer",
                            "description": "The first number",
                        },
                        "b": {
                            "type": "integer",
                            "description": "The second number",
                        }
                    },
                    "required": ["a", "b"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "multiply_nums",
                "description": "multiply two numbers together",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "a": {
                            "type": "integer",
                            "description": "The first number",
                        },
                        "b": {
                            "type": "integer",
                            "description": "The second number",
                        }
                    },
                    "required": ["a", "b"],
                },
            },
        },
    ]

    function_names = {
        "add_nums": functools.partial(add_nums),
        "subtract_nums": functools.partial(subtract_nums),
        "multiply_nums": functools.partial(multiply_nums),


    }

    messages = [{"role": "user", "content": input}]

    client = Mistral(api_key=MISTRAL_API_KEY)
    response = client.chat.complete(
        model = model,
        messages = messages,
        tools = tools,
        tool_choice = "any",
    )
    messages.append(response.choices[0].message)
    tool_call = response.choices[0].message.tool_calls[0]
    function_name = tool_call.function.name
    function_params = json.loads(tool_call.function.arguments)
    function_result = function_names[function_name](**function_params)

    messages.append({"role":"tool", "name":function_name, "content":str(function_result), "tool_call_id":tool_call.id})
    response = client.chat.complete(
        model = model, 
        messages = messages
    )
    print(response.choices[0].message.content)



if __name__ == '__main__':
    gemini_call("what is 5 - 7")
    mistral_call("what is 5 * 7")