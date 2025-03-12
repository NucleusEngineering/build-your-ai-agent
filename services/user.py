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
from models import model, user

from google.cloud.firestore_v1.base_query import FieldFilter
from google.genai import types
from vertexai.preview.vision_models import ImageGenerationModel

class User:
    def __init__(self, db, config_service, rag_model):
        """
        Initializes the User service.

        Args:
            db: Firestore client instance.
            config_service: Service to get configuration values.
            rag_model: The RAG model instance.
        """
        self.db = db
        self.config_service = config_service
        self.rag_model = rag_model


    @staticmethod
    def get_function_declarations():

        fc_generate_avatar = types.FunctionDeclaration(
            name='fc_generate_avatar',
            description='Create new avatar or avatar. Inject the description into the function that is being called.',
            parameters=types.Schema(
                type='OBJECT',
                properties={
                    'description': types.Schema(
                        type='string', 
                        description='Description of the picture or avatar',
                    ),
                },
                required=[
                    'description',
                ]
            ),
        )

        fc_rag_retrieval = types.FunctionDeclaration(
            name='fc_rag_retrieval',
            description='Function to be invoked when the prompt is about a super secret game called Cloud Meow.',
            parameters=types.Schema(
                type='OBJECT',
                properties={
                    'question_passthrough': types.Schema(
                        type='string', 
                        description='The whole user\'s prompt in the context of this message'
                    )
                },
                required=[
                    'question_passthrough',
                ]            
            )
        )

        fc_save_model_color = types.FunctionDeclaration(
            name='fc_save_model_color',
            description='Save new color when user requests to update his game model. Input is a color in hex format',
            parameters=types.Schema(
                type='OBJECT',
                properties={
                    'color': types.Schema(
                        type='string', 
                        description='Hex color'
                    )
                },
                required=[
                    'color',
                ]
            )
        )        

        fc_revert_model_color = types.FunctionDeclaration(
            name='fc_revert_model_color',
            description='Revert the color/material of user\'s model on their request.',
            parameters=types.Schema(
                type='OBJECT',
                properties={}
            )
        )        

        fc_convert_avatar = types.FunctionDeclaration(
            name='fc_convert_avatar',
            description='Convert avatar to 3D model.',
            parameters=types.Schema(
                type='OBJECT',
                properties={}
            )
        )        

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
            fc_generate_avatar,
            fc_rag_retrieval,
            fc_save_model_color,
            fc_revert_model_color,
            fc_show_my_model,
            fc_show_my_avatar,
            fc_convert_avatar
        ])
    
    def fc_save_model_color(self, color, user_id):
        """
        Saves the model's color to Firestore.

        Args:
            user_id: The ID of the user.
            color: The hex color to save.

        Returns:
            A string response for the user.
        """
        try:
            # Get the models collection reference
            models_ref = self.db.collection("models")

            # Query for the model with the given user_id
            query = models_ref.where(filter=FieldFilter("user_id", "==", user_id))
            results = query.get()

            if not results:
                return f"Reply that no character for user '{user_id}' was found."

            # Update the color of the first matching document
            for doc in results:
                doc.reference.update({"color": color, "original_material": False})
                logging.info(f"Updated color to '{color}' for '{user_id}'\'s model.")
                break

            return '''Reply that their character color has been updated''', '''<script>window.reloadCurrentModel();</script>'''
        
        except Exception as e:
            logging.error("%s, %s", traceback.format_exc(), e)
            return 'Reply that we failed to update their character settings.'

    
    def fc_revert_model_color(self, user_id):
        """
        Revert's the model's color.

        Args:
            user_id: The ID of the user.

        Returns:
            A string response for the user.
        """
        try:
            # Get the models collection reference
            models_ref = self.db.collection("models")

            # Query for the model with the given user_id
            query = models_ref.where(filter=FieldFilter("user_id", "==", user_id))
            results = query.get()

            if not results:
                return f"Reply that no character for user '{user_id}' was found."

            # Update the color of the first matching document
            for doc in results:
                doc.reference.update({"original_material": True})
                logging.info(f"Reverted to original materials for '{user_id}'\'s model.")
                break

            return '''Reply that their character colors have been reverted''', '''<script>window.reloadCurrentModel();</script>'''
        
        except Exception as e:
            logging.error("%s, %s", traceback.format_exc(), e)
            return 'Reply that we failed to update their character settings.'

    def fc_generate_avatar(self, description, user_id):
        """Returns the new avatar for the user.

        Args:
        description: The description of the avatar to be generated
        """
        try:
            model = ImageGenerationModel.from_pretrained(self.config_service.get_property("general", "imagen_version"))

            instruction = self.config_service.get_property("chatbot", "diffusion_generation_instruction")

            images = model.generate_images(
                prompt=instruction % description,
                number_of_images=4,
                language="en",
                seed=100,
                add_watermark=False,
                aspect_ratio="1:1",
                safety_filter_level="block_some",
                person_generation="allow_adult",
            )

            output_file = "static/avatars/" + str(user_id) + ".png"
            images[0].save(location=output_file, include_generation_parameters=False)        

            cdn_url = '/' + output_file
        except Exception as e:
            logging.info("%s, %s", traceback.format_exc(), e)
            return 'Reply that we failed to generate a new avatar. Ask them to try again later'


        try:
            # Update Firestore "users" collection
            user_ref = self.db.collection("users").where(filter=FieldFilter("user_id", "==", user_id))
            user_ref.get()[0].reference.update({"avatar": cdn_url})
            logging.info('Updated user avatar to %s', cdn_url)
        except Exception as e:
            logging.info("%s, %s", traceback.format_exc(), e)
            return 'Reply that we failed to generate a new avatar. Ask them to try again later'

        return '''Reply something like "There you go."''', '''
            <div>
                <br>
                <img class="avatar" src="%s?rand=%s">
            </div>''' % (cdn_url, str(random.randint(0, 1000000)))

    def fc_rag_retrieval(self, user_id, question_passthrough):
        """
        Retrieves information using RAG model.

        Args:
            user_id: The ID of the user.
            question_passthrough: The user's prompt.

        Returns:
            The RAG model's response as a string.
        """

        response = self.rag_model.generate_content(question_passthrough)
        return extract_text(response), ''

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

    def fc_convert_avatar(self, user_id):
        try:
            image_path = "static/avatars/" + str(user_id) + ".png"
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

            headers = {'Content-Type': 'application/json'}
            payload = json.dumps({"image": encoded_string})
            response = requests.post("https://genai3d.nikolaidan.demo.altostrat.com/upload", headers=headers, data=payload)
            response.raise_for_status()  # Raise an exception for bad status codes

            job_data = response.json()
            job_id = job_data.get("job_id")

            if job_id:
                logging.info(f"3D model generation started for user {user_id} with job ID: {job_id}")

                # in an ideal scenario we would constantly poll /check_job/{job_id} until it's done.
                # for sake of brevity, we will just assume that it will work
                # and return an appropriate message for the user.

                threading.Thread(target=self.poll_job_status, args=(job_id)).start()

                return '''Reply that the conversion of their avatar to 3D model has started. The model will be updated later.''',f'''<script>console.log("Job id: {job_id}");</script>'''
            
            else:
                 return "Reply that we couldn't start the conversion of their avatar to 3D model. Please try again later.", ""
        
        except requests.exceptions.RequestException as e:
            logging.error(f"API request failed: {e}")
            return "Reply that we failed to convert your avatar to a 3D model. Please try again later.", ""
        except FileNotFoundError:
            logging.error(f"Avatar file not found for user {user_id}")
            return "Reply that we couldn't find your avatar. Please create one first.", ""
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            return "Reply that we failed to convert your avatar to a 3D model. Please try again later.", ""
            

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
        
    def poll_job_status(self, job_id, user_id):
        """
        Polls the job status endpoint until the job is finished and updates the model.
        
        Args:
            job_id: The job ID to poll.
        """
        while True:
            try:
                response = requests.get(f"https://genai3d.nikolaidan.demo.altostrat.com/check_job/{job_id}")
                response.raise_for_status()
                job_data = response.json()
                job_status = job_data.get("status")

                if job_status == "finished":
                    logging.info(f"Job {job_id} finished for user {user_id}")
                    filename = job_data.get("filename")

                    if filename:
                        self.download_and_update_model(filename, user_id)
                        break  # Exit the loop once the job is finished and the model is updated
                elif job_status == "queued":
                  logging.info(f"Job {job_id} is queued, waiting 5 seconds to check again")
                else:
                  logging.error(f"Job {job_id} returned unknown status {job_status}")
                  break

                time.sleep(5)  # Wait for 5 seconds before checking again
            except requests.exceptions.RequestException as e:
                logging.error(f"Error checking job status for {job_id}: {e}")
                break

    def download_and_update_model(self, filename, user_id):
        """Downloads the model file and saves it to the correct location."""
        model_content = requests.get(filename).content
        with open("static/models/default.glb", "wb") as model_file:
            model_file.write(model_content)
        logging.info(f"Updated 3d model for user {user_id} from {filename}")
