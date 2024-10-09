from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from task_analyzer import TaskAnalyzer
from utils import plot_similarity_distribution, setup_logger
from config import QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, LOG_FILE
import uuid
import io
import logging



app = Flask(__name__)
CORS(app)


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}", exc_info=True)
    return jsonify(error=str(e)), 500

analyzer = TaskAnalyzer(QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME)

@app.route('/add_task', methods=['POST'])
def add_task():
    data = request.json
    task_id = data.get('id') or str(uuid.uuid4())
    subject = data.get('subject', '')
    description = data.get('description', '')
    
    logger.info(f"Adding new task: {task_id}")
    analyzer.add_task(task_id, subject, description)
    
    return jsonify({"message": "Task added successfully", "task_id": task_id}), 201

@app.route('/get_task/<task_id>', methods=['GET'])
def get_task(task_id):
    logger.info(f"Retrieving task: {task_id}")
    task = analyzer.get_task(task_id)
    if task:
        return jsonify(task)
    else:
        return jsonify({"error": "Task not found"}), 404

@app.route('/list_tasks')
def list_tasks():
    limit = request.args.get('limit', type=int, default=None)
    offset = request.args.get('offset', type=int, default=0)
    tasks = analyzer.list_all_tasks(limit=limit, offset=offset)
    return jsonify(tasks)

@app.route('/search_similar_tasks', methods=['POST'])
def search_similar_tasks():
    data = request.json
    query = data.get('query', '')
    threshold = data.get('threshold', 0.7)
    limit = data.get('limit', 10)
    
    logger.info(f"Searching similar tasks for query: {query}")
    similar_tasks = analyzer.find_similar_tasks(query, threshold, limit)
    logger.info(f"Found {len(similar_tasks)} similar tasks")
    return jsonify(similar_tasks)

@app.route('/hybrid_search', methods=['POST'])
def hybrid_search():
    data = request.json
    query = data.get('query', '')
    limit = data.get('limit', 10)
    semantic_weight = data.get('semantic_weight', 0.5)
    
    results = analyzer.hybrid_search(query, limit=limit, semantic_weight=semantic_weight)
    return jsonify(results)


@app.route('/load_csv', methods=['POST'])
def load_csv():
    if 'file' not in request.files:
        logger.warning("No file part in the request")
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        logger.warning("No selected file")
        return jsonify({"error": "No selected file"}), 400
    if file and file.filename.endswith('.csv'):
        logger.info(f"Processing CSV file: {file.filename}")
        task_count = analyzer.load_csv(file)
        return jsonify({"message": f"CSV loaded successfully. {task_count} tasks added."}), 200
    else:
        logger.warning("Invalid file format")
        return jsonify({"error": "Invalid file format. Please upload a CSV file."}), 400

@app.route('/find_duplicates', methods=['GET'])
def find_duplicates():
    threshold = request.args.get('threshold', 0.95, type=float)
    logger.info(f"Finding duplicates with threshold: {threshold}")
    duplicates = analyzer.find_duplicates(threshold)
    return jsonify(duplicates)

@app.route('/plot_similarity_distribution', methods=['GET'])
def get_similarity_distribution():
    threshold = request.args.get('threshold', 0.95, type=float)
    logger.info(f"Plotting similarity distribution with threshold: {threshold}")
    duplicates = analyzer.find_duplicates(threshold)
    plot = plot_similarity_distribution(duplicates)
    return send_file(plot, mimetype='image/png')

@app.route('/collection_stats', methods=['GET'])
def get_collection_stats():
    logger.info("Retrieving collection statistics")
    stats = analyzer.get_collection_stats()
    return jsonify(stats)

@app.route('/semantic_search', methods=['POST'])
def semantic_search():
    data = request.json
    query = data.get('query', '')
    limit = data.get('limit', 10)
    
    logger.info(f"Performing semantic search for query: {query}")
    results = analyzer.semantic_search(query, limit)
    return jsonify(results)

@app.route('/health')
def health_check():
    qdrant_status = "OK" if analyzer.check_qdrant_connection() else "ERROR"
    return jsonify({
        "status": "OK",
        "qdrant_connection": qdrant_status
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)