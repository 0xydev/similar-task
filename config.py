import os

QDRANT_HOST = os.getenv('QDRANT_HOST', 'qdrant')
QDRANT_PORT = int(os.getenv('QDRANT_PORT', 6333))
COLLECTION_NAME = os.getenv('COLLECTION_NAME', 'tasks')
LOG_FILE = os.getenv('LOG_FILE', 'task_analyzer.log')