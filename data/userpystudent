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

import traceback
import random
import logging
import requests, json
import time
import threading
import base64

from common.function_calling import extract_text
from common.rag import RAG
from models import model, user

from google.cloud.firestore_v1.base_query import FieldFilter
from google import genai
from google.genai import types
from vertexai.preview.vision_models import ImageGenerationModel

class User:
    def __init__(self, db, config_service, gemini_client: genai.Client):
        """
        Initializes the User service.

        Args:
            db: Firestore client instance.
            config_service: Service to get configuration values.
            rag_config: The RAG model instance.
        """
        self.db = db
        self.config_service = config_service
        self.gemini_client = gemini_client


    @staticmethod
    def get_function_declarations():
        fc_show_my_model = types.FunctionDeclaration(
            name='fc_show_my_model',
            description='Show user\'s model / character on the screen.',
            parameters=types.Schema(
                type='OBJECT',
                properties={},            
            )
        )
        
        fc_show_my_avatar = types.FunctionDeclaration(
            name='fc_show_my_avatar',
            description='Show user\'s current avatar.',
            parameters=types.Schema(
                type='OBJECT',
                properties={}
            )
        )


        return types.Tool(function_declarations=[
            fc_show_my_model,
            fc_show_my_avatar,
        ])
    
    def fc_show_my_model(self, user_id):
        logging.info(f"Showing user's ({user_id}) character")
        return '''Reply something like "there you go"''', '''<script>$("#modelWindow").show();</script>'''

    def fc_show_my_avatar(self, user_id):
        logging.info(f"Showing user's ({user_id}) avatar")
        return '''Reply something like "There you go."''', '''
            <div>
                <br>
                <img class="avatar" src="/static/avatars/%s.png?rand=%s">
            </div>''' % (user_id, str(random.randint(0, 1000000)))

    def get_model(self, user_id):
        """
        Retrieves the color of a character from Firestore.

        Args:
            user_id: The ID of the user.

        Returns:
            A dictionary containing the character's color information, or None if not found.
        """
        try:
            # Get the models collection reference
            models_ref = self.db.collection("models")

            # Query for the model with the given user_id
            query = models_ref.where("user_id", "==", user_id)
            results = query.get()

            if not results:
                logging.warning(f"No character found for '{user_id}'.")
                return None

            # In our current setup, there should only be one matching model
            # model_doc = results[0]
            # model_data = model_doc.to_dict()
            
            return model.Model.from_dict(results[0].to_dict())

            # we return the full object since it is now already a dict
            # return model_data

        except Exception as e:
            logging.error("%s, %s", traceback.format_exc(), e)
            return None
        