"""
Provide a restful server for chat completion.

Usage:
python3 -m fastchat.serve.restful_server --model ~/model_weights/llama-7b
"""

import argparse

import uvicorn
from fastapi import FastAPI
from typing import List, Optional
from pydantic import BaseModel

from fastchat.serve.inference import load_model, generate_stream
from fastchat.serve.serve_chatglm import  chatglm_generate_stream


app = FastAPI()

models = {}
tokenizers = {}
model_devices = {}


class CompletionRequest(BaseModel):
    model: str
    messages: List[dict]
    temperature: Optional[float] = 1.0
    n: Optional[int] = 1
    max_tokens: Optional[int] = None


@app.post("/v0/chat/completions")
async def create_chat_completion(request: CompletionRequest):
    # You should implement the actual functionality here by sending a request to OpenAI's API
    # For now, we'll return the input data as is
    return request.dict()


def chat_completion(model_name: str, messages: List[dict], temperature: float, max_tokens: int, debug: bool):
    model = models[model_name]
    tokenizer = tokenizers[model_name]
    device = model_devices[model_name]
    is_chatglm = "chatglm" in str(type(model)).lower()

    # Chat
    conv = conv_templates[conv_template].copy()
    while True:
        try:
            inp = chatio.prompt_for_input(conv.roles[0])
        except EOFError:
            inp = ""
        if not inp:
            print("exit...")
            break

        conv.append_message(conv.roles[0], inp)
        conv.append_message(conv.roles[1], None)

        if is_chatglm:
            prompt = conv.messages[conv.offset:]
            generate_stream_func = chatglm_generate_stream
            skip_echo_len = len(conv.messages[-2][1]) + 1
        else:
            generate_stream_func = generate_stream
            prompt = conv.get_prompt()
            skip_echo_len = len(prompt.replace("</s>", " ")) + 1

        params = {
            "model": model_path,
            "prompt": prompt,
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
        }

        chatio.prompt_for_output(conv.roles[1])
        output_stream = generate_stream_func(model, tokenizer, params, device)
        outputs = chatio.stream_output(output_stream, skip_echo_len)
        # NOTE: strip is important to align with the training data.
        conv.messages[-1][-1] = outputs.strip()

        if debug:
            print("\n", {"prompt": prompt, "outputs": outputs}, "\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m",
        help="The path to the weights")
    parser.add_argument("--device", type=str, choices=["cpu", "cuda", "mps"], default="cuda")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--load-8bit", action="store_true", help="Use 8-bit quantization.")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    main(args)

    uvicorn.run("restful_server:app", host=args.host, port=args.port, reload=True)
