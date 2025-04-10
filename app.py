# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import vertexai
import os
import logging
import traceback

from google import genai
from google.genai import types, chats

import firebase_admin
from firebase_admin import credentials, firestore

from flask import Flask, request, jsonify #, render_template

from common import audiostream, config as configuration, function_calling
from services.user import User as UserService

# Environment variables

PROJECT_ID = os.environ.get("PROJECT_ID", "dn-demos")
REGION = os.environ.get("REGION", "us-central1")
FAKE_USER_ID = "7608dc3f-d239-405c-a097-b152ab38a354"

DEFAULT_SAFETY_SETTINGS = [
                types.SafetySetting(
                    category='HARM_CATEGORY_UNSPECIFIED',
                    threshold='BLOCK_ONLY_HIGH',
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_DANGEROUS_CONTENT',
                    threshold='BLOCK_ONLY_HIGH',
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_SEXUALLY_EXPLICIT',
                    threshold='BLOCK_ONLY_HIGH',
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_HARASSMENT',
                    threshold='BLOCK_ONLY_HIGH',
                ),
                types.SafetySetting(
                    category='HARM_CATEGORY_HATE_SPEECH',
                    threshold='BLOCK_ONLY_HIGH',
                )                
            ]

logging.basicConfig()
logging.getLogger().setLevel(logging.DEBUG)

vertexai.init(project=PROJECT_ID, location=REGION)

# Config file loader
config = configuration.Config.get_instance()

# Our main chat config with system instructions
chat_config = types.GenerateContentConfig(
    system_instruction=config.get_property('chatbot', 'llm_system_instruction') + config.get_property('chatbot', 'llm_response_type'),
    tools=[UserService.get_function_declarations()],
    safety_settings=DEFAULT_SAFETY_SETTINGS,        
    automatic_function_calling=types.AutomaticFunctionCallingConfig(
        disable=True
    ),
    tool_config=types.ToolConfig(
        function_calling_config=types.FunctionCallingConfig(mode='AUTO'),
    ),
)

firebase_admin.initialize_app(credentials.ApplicationDefault())

def init_client() -> genai.Client:
    client = genai.Client(
        vertexai=True, project=PROJECT_ID, location=REGION
    )
    
    return client

# Chat initialization per tenant (cleanup needed after timeout/logout)
def init_client_chat(client: genai.Client, user_id) -> chats.Chat:
    if user_id in client_sessions and client_sessions[user_id] != None:
        logging.debug("Re-using existing session")
        return client_sessions[user_id]

    logging.debug("Creating new chat session for user %s", user_id)

    gemini_client = client.chats.create(
        model=config.get_property('general', 'llm_gemini_version'), config=chat_config, 
    )

    client_sessions[user_id] = gemini_client
    return client_sessions[user_id]

app = Flask(
    __name__,
    instance_relative_config=True,
    template_folder="templates",
)

gemini_client = init_client()
db_client = firestore.client()
user_service = UserService(db_client, config, gemini_client)

# Init our session handling variables
client_sessions = {}

# Our main chat handler
@app.route("/chat", methods=["POST"])
def chat():
    # chat = init_chat(chat_model, FAKE_USER_ID)
    chat = init_client_chat(gemini_client, FAKE_USER_ID)
    audio = audiostream.get_audio_stream(request)

    # If we got audio stream input, let's convert it first to text via gemini flash
    if audio != None: 
        transcribed_audio_response = gemini_client.models.generate_content(
            model=config.get_property('general', 'llm_gemini_version'),
            contents=[
                config.get_property('chatbot', 'audio_transcription_instruction'),
                types.Part.from_bytes(data=audio, mime_type='audio/mpeg')
            ]
        )
        
        prompt = transcribed_audio_response.text
        logging.debug(f"Transcribed audio: %s", prompt)
        
        # Now that we have the audio in text for, so we can send it further to our pipeline
        response = chat.send_message(message=prompt, config=chat_config)
    else:        
        prompt = types.Part.from_text(text=request.form.get("prompt"))
        response = chat.send_message(message=prompt, config=chat_config)

    if response.function_calls is not None:
        try:
            function_call_part = response.function_calls[0]
            function_call_name = function_call_part.name
            function_call_args = function_call_part.args
            function_call_content = response.candidates[0].content

            logging.info("Calling " + function_call_name)
            logging.info(function_call_args)
            logging.info(function_call_content)
            
            function_call_args['user_id'] = FAKE_USER_ID
            function_result, html_response = function_calling.call_function(user_service, function_call_name, function_call_args)

            function_response_part = types.Part.from_function_response(
                name=function_call_part.name,
                response={
                    'result': function_result
                }
            )

            response = chat.send_message(message=function_response_part, config=chat_config)
            text_response = function_calling.extract_text(response) + html_response

        except TypeError as e:
            logging.error("%s, %s", traceback.format_exc(), e)
            text_response = config.get_property('chatbot', 'generic_error_message')

        except Exception as e:
            logging.error("%s, %s", traceback.format_exc(), e)
            text_response = config.get_property('chatbot', 'generic_error_message')
    else:
        text_response = function_calling.extract_text(response)

    if len(text_response) == 0:
        text_response = config.get_property('chatbot', 'generic_error_message')
        
    return function_calling.gemini_response_to_template_html(text_response)

@app.route("/", methods=["GET"])
def home():
#    if os.environ.get("DEV_MODE") == "true":
    with open("templates/index.html", mode='r') as file: #
        data = file.read()

    return data
    
#    return render_template("index.html")

@app.route("/version", methods=["GET"])
def version():
    return jsonify({
        "version": config.get_property('general', 'version')
        })

# Get character color
@app.route("/get_model", methods=["GET"])
def get_model():
    model = user_service.get_model(FAKE_USER_ID)

    if(model is not None) :
        response = jsonify(model.to_dict())
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response
        
    return 'Character was not found. Double-check the name and try again.', 404

@app.route("/reset", methods=["GET"])
def reset():
    for uid in client_sessions:
        client_sessions[uid] = None

    return jsonify({'status': 'ok'}), 200

if __name__ == "__main__":
    os.makedirs('uploads', exist_ok=True)
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8888)))
