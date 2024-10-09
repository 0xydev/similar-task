import logging
import matplotlib.pyplot as plt
import seaborn as sns
import io
from typing import List, Tuple

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger

def plot_similarity_distribution(duplicates: List[Tuple[str, str, float, str, str]]) -> io.BytesIO:
    similarities = [sim for _, _, sim, _, _ in duplicates]
    plt.figure(figsize=(10, 6))
    sns.histplot(similarities, kde=True)
    plt.title("Potential Duplicates Similarity Score Distribution")
    plt.xlabel("Similarity Score")
    plt.ylabel("Count")
    
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    plt.close()
    
    return img_buffer