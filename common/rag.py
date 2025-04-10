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

from vertexai import rag
from vertexai.generative_models import Tool

from google.genai import types

from . import config

class RAG:
    files = []

    def __init__(self, config_service: config.Config):
        corpora = rag.list_corpora()
        corpus_name = config_service.get_property('rag', 'corpus_name')
        file_paths = config_service.get_property('rag', 'paths').split('|')

        rag_corpus = None

        for c in corpora:
            if c.display_name == corpus_name:
                # rag.delete_corpus(c.name) # Comment this to stop re-creating RAG corpora every time
                rag_corpus = rag.get_corpus(c.name) # Uncomment this once you are happy with your setup

        if rag_corpus is None:
            rag_corpus = rag.create_corpus(display_name=corpus_name)
            if rag_corpus is None:
                print("No corpus!")
                raise Exception('Cannot load nor create new RagCorpus')

            # Import Files to the RagCorpus
            rag.import_files(
                rag_corpus.name,
                file_paths,
                chunk_size=512,  # Optional
                chunk_overlap=100,  # Optional
            )

        self.name = rag_corpus.name

        rag_files = list(rag.list_files(corpus_name=rag_corpus.name))
        for f in rag_files:
            self.files.append(f.name.split('/')[-1])


    def _get_rag_retrieval(self) -> Tool:
        return Tool.from_retrieval(
            retrieval=rag.Retrieval(
                source=rag.VertexRagStore(
                    rag_resources=[
                        rag.RagResource(
                            rag_corpus=self.name,  # Currently only 1 corpus is allowed.
                            # Supply IDs from `rag.list_files()`.
                            rag_file_ids=self.files,
                        )
                    ],
                    rag_retrieval_config=rag.RagRetrievalConfig(
                        filter=rag.Filter(
                            vector_distance_threshold=0.5
                        )
                    )
                ),
            )
        )
    
    def get_rag_retrieval(self) -> types.Tool:
        print(self.name)
        return types.Tool(
        retrieval=types.Retrieval(
            vertex_rag_store=types.VertexRagStore(
                rag_resources=[
                    types.VertexRagStoreRagResource(
                        rag_corpus=self.name
                    )
                ],
            similarity_top_k=3,
            vector_distance_threshold=0.5,
            )
        ),
    )

