import numpy as np
from typing import List, Dict, Tuple, Any
from transformers import AutoModel, AutoTokenizer
import torch
from qdrant_client import QdrantClient
from qdrant_client.http import models
import uuid
import time
import pandas as pd
from tqdm import tqdm
import logging
from config import LOG_FILE
from utils import setup_logger

logger = setup_logger(__name__, LOG_FILE)

class TaskAnalyzer:
    def __init__(self, qdrant_host: str, qdrant_port: int, collection_name: str):
        self.client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.model = AutoModel.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("jinaai/jina-embeddings-v3", trust_remote_code=True)
        self._create_collection_if_not_exists()
        self.embedding_cache: Dict[str, List[float]] = {}

    def _create_collection_if_not_exists(self) -> None:
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
            )
            logger.info(f"Created new collection: {self.collection_name}")

    def _get_embedding(self, text: str) -> List[float]:
        if text in self.embedding_cache:
            return self.embedding_cache[text]
        
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        self.embedding_cache[text] = embeddings.tolist()
        return embeddings.tolist()
    
    def semantic_search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        query_embedding = self._get_embedding(query)
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        results = []
        for result in search_result:
            results.append({
                'id': result.id,
                'subject': result.payload['subject'],
                'description': result.payload['description'],
                'similarity': result.score
            })
        
        logger.info(f"Semantic search for '{query}' returned {len(results)} results")
        return results
    
    def hybrid_search(self, query: str, limit: int = 10, semantic_weight: float = 0.5) -> List[Dict[str, Any]]:
        query_vector = self._get_embedding(query)
        semantic_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        keyword_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="text",
                        match=models.MatchText(text=query)
                    )
                ]
            ),
            limit=limit
        )
        
        combined_results = {}
        
        for result in semantic_results:
            combined_results[result.id] = {
                'id': result.id,
                'subject': result.payload['subject'],
                'description': result.payload['description'],
                'score': result.score * semantic_weight
            }
        
        for result in keyword_results:
            if result.id in combined_results:
                combined_results[result.id]['score'] += result.score * (1 - semantic_weight)
            else:
                combined_results[result.id] = {
                    'id': result.id,
                    'subject': result.payload['subject'],
                    'description': result.payload['description'],
                    'score': result.score * (1 - semantic_weight)
                }
        
        sorted_results = sorted(combined_results.values(), key=lambda x: x['score'], reverse=True)[:limit]
        
        logger.info(f"Hybrid search for '{query}' returned {len(sorted_results)} results")
        return sorted_results


    def add_task(self, task_id: str, subject: str, description: str) -> None:
        combined_text = f"{subject} {description}"
        task_embedding = self._get_embedding(combined_text)
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=[
                        models.PointStruct(
                            id=task_id,
                            vector=task_embedding,
                            payload={'text': combined_text, 'subject': subject, 'description': description}
                        )
                    ]
                )
                logger.info(f"Added task with ID: {task_id}")
                return
            except Exception as e:
                logger.warning(f"Error adding task {task_id} to Qdrant (attempt {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to add task {task_id} after {max_retries} attempts")
                else:
                    time.sleep(1)

    def get_task(self, task_id: str) -> Dict[str, Any] | None:
        try:
            try:
                formatted_id = int(task_id)
            except ValueError:
                formatted_id = task_id

            result = self.client.retrieve(
                collection_name=self.collection_name,
                ids=[formatted_id]
            )
            if result:
                return {
                    'id': result[0].id,
                    'subject': result[0].payload.get('subject', ''),
                    'description': result[0].payload.get('description', '')
                }
            else:
                logger.warning(f"Task not found: {task_id}")
                return None
        except Exception as e:
            logger.error(f"Error retrieving task {task_id}: {str(e)}")
            return None
    
    def list_all_tasks(self, limit: int | None = None, offset: int = 0) -> List[Dict[str, Any]]:
        try:
            all_tasks = []
            batch_size = 1000
            current_offset = offset

            while True:
                results = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=batch_size,
                    offset=current_offset,
                    with_payload=True,
                    with_vectors=False
                )[0]

                batch_tasks = [
                    {
                        'id': point.id,
                        'subject': point.payload.get('subject', ''),
                        'description': point.payload.get('description', '')
                    }
                    for point in results
                ]

                all_tasks.extend(batch_tasks)
                
                if len(batch_tasks) < batch_size or (limit is not None and len(all_tasks) >= limit):
                    break

                current_offset += batch_size

            if limit is not None:
                return all_tasks[:limit]
            else:
                return all_tasks

        except Exception as e:
            logger.error(f"Error listing tasks: {str(e)}")
            return []


    def find_similar_tasks(self, query_text: str, threshold: float = 0.7, limit: int = 10) -> List[Dict[str, Any]]:
        query_embedding = self._get_embedding(query_text)
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=limit
        )
        
        return [
            {'id': result.id, 'subject': result.payload['subject'], 'description': result.payload['description'], 'similarity': result.score}
            for result in search_result
            if result.score > threshold
        ]

    def load_csv(self, file) -> int:
        df = pd.read_csv(file, encoding='utf-8')
        df = df.fillna('')
        df['id'] = df.get('#', [str(uuid.uuid4()) for _ in range(len(df))])
        
        for _, task in tqdm(df.iterrows(), total=len(df), desc="Processing tasks"):
            self.add_task(task['id'], task.get('Subject', ''), task.get('Description', ''))
        
        logger.info(f"Loaded {len(df)} tasks from CSV")
        return len(df)

    def find_duplicates(self, threshold: float = 0.95) -> List[Tuple[str, str, float, str, str]]:
        all_points = self.client.scroll(
            collection_name=self.collection_name,
            limit=10000,
            with_payload=True,
            with_vectors=True
        )[0]
        
        duplicates = []
        for i, point in enumerate(tqdm(all_points, desc="Finding duplicates")):
            similar = self.find_similar_tasks(point.payload['text'], threshold)
            duplicates.extend(
                (point.id, task['id'], task['similarity'], point.payload['subject'], task['subject'])
                for task in similar
                if task['id'] != point.id
            )
        
        logger.info(f"Found {len(duplicates)} potential duplicates")
        return duplicates

    def get_collection_stats(self):
        try:
            info = self.client.get_collection(self.collection_name)
            vector_size = info.config.params.vectors.size if info.config.params.vectors else None
            return {
                "name": self.collection_name,
                "vector_size": vector_size,
                "distance": str(info.config.params.vectors.distance) if info.config.params.vectors else None,
                "vector_count": info.vectors_count,
                "point_count": info.points_count,
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {str(e)}")
            return {"error": str(e)}
        
    def check_qdrant_connection(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            logger.error(f"Qdrant connection error: {str(e)}")
            return False