# AI Agents

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-%230db7ed.svg)](https://www.docker.com/)
[![Terraform](https://img.shields.io/badge/terraform-%235835CC.svg)](https://www.terraform.io/)

**Build your own AI Agent** is an boilerplate project / workshop leveraging the power of [Google's Gemini 2.0](https://blog.google/technology/google-deepmind/google-gemini-ai-update-december-2024/) AI models. In the context of this workshop, it offers:

* **[Function Calling](https://ai.google.dev/gemini-api/docs/function-calling):** Seamless integration of custom functions.
* **[RAG (Retrieval Augmented Generation)](https://cloud.google.com/vertex-ai/generative-ai/docs/llamaindex-on-vertexai):** Enhanced responses with external data context.

Built with Python, it utilizes [Cloud Firestore](https://cloud.google.com/firestore).

## Getting Started

### Workshop easy start

Clicking this link will take you to Google Cloud Platform's console where you can start building.

[![Begin the Tutorial](.journey/journey.svg)](https://shell.cloud.google.com/cloudshell/editor?cloudshell_git_repo=https://github.com/NucleusEngineering/build-your-ai-agent.git&cloudshell_tutorial=.journey/tutorial.neos.md&show=ide&cloudshell_workspace=)

If you close this tutorial window by mistake, you can always run the following command to re-open it: 
`cloudshell launch-tutorial .journey/tutorial.neos.md`

### Prerequisites

* Python 3.x
* Some cloud computing knowledge
* Google Cloud Platform account

### Local & Cloud Shell Setup

1. Clone this repository: `git clone git@github.com:NucleusEngineering/build-your-ai-agent.git`
2. Set environment variables:
   ```bash
   export PROJECT_ID=your-gcp-project-id
   export REGION=your-gcp-region
   ```
3. Import initial data into Firestore:
   ```bash
   gcloud firestore import gs://ai-agent-data-bucket/firestore-schema --database="(default)"
   ```

4. Start the application 
   ```bash
   python3 -m flask run --host=0.0.0.0 --port=8080 --debugger --reload
   ```
5. Access the application at: http://localhost:8080 or 

### Remote Deployment

1. Navigate to `terraform` folder.
2. Customize `terraform.tfvars` based on `terraform.tfvars.template`.
3. Run: `terraform init`, `terraform plan`, `terraform apply`
4. In the root project folder, execute Cloud Build: `gcloud builds submit --config cloudbuild.yaml --substitutions _SERVICE_NAME=your-service-name,_REGION=your-gcp-region`
5. Get the URL where the demo was deployed: `gcloud run services list | grep -i build-your-ai-agent`

## Usage

You can interact with the AI Agent via writing. 

Information that AI Agent can discuss with you is:

* **Google Cloud:**  Get the information about Google Cloud's technology. E.g.: `What are Cloud Functions good for?`
* **Generate new avatar:**  Get a new avatar with help of Generative AI. E.g.: `Create me a new avatar that looks like an orange cat.`
* **Manipulate your own 3D model:**  . E.g.: `Show my character model.` or `Create me a model from my avatar.` or `Change the color of my model to glowing purple`.
* **Ask specific questions (RAG):**  Get grounded answers based on the specific documents supplied to our RAG API. E.g.: `What is a Cloud Meow game?`
* **Try multiple languages:**  Speak with the model in any language you know! E.g.: `Explica-me como function Cloud Run.` (pt-PT)

## License

Apache License 2.0. See the [LICENSE](LICENSE) file.
