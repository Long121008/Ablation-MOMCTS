import json
from sklearn.cluster import KMeans
import torch
import numpy as np
import glob
import os
from transformers import AutoModel, AutoTokenizer
from sklearn.metrics.pairwise import cosine_distances
import matplotlib.pyplot as plt
def load_programs_from_multiple_files(file_paths):
    all_programs = []

    for path in file_paths:
        if not os.path.exists(path):
            print(f"Cảnh báo: Không tìm thấy file {path}")
            continue
            
        with open(path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                count = 0
                for item in data:
                    prog = item.get("function") or item.get("program", "")
                    if isinstance(prog, str) and len(prog.strip()) > 0:
                        all_programs.append(prog.strip())
                        count += 1
                print(f"Đã load {count} chương trình từ: {path}")
            except Exception as e:
                print(f"Lỗi khi đọc file {path}: {e}")

    unique_programs = list(set(all_programs))
    
    if len(unique_programs) < 2:
        raise ValueError("Không đủ dữ liệu chương trình độc nhất để tính toán.")

    print(f"\n--- Tổng cộng ---")
    print(f"Tổng số mẫđộcu thu thập: {len(all_programs)}")
    print(f"Số lượng mẫu  nhất (Unique): {len(unique_programs)}")
    
    return unique_programs

def embed_programs(programs):
    model_name = "Salesforce/codet5p-110m-embedding"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"\nĐang tải model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
    
    embeddings = []
    print(f"Đang tạo embeddings trên {device} cho {len(programs)} mẫu...")
    
    for i, prog in enumerate(programs):
        inputs = tokenizer(prog, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            embedding = model(**inputs).cpu().numpy()
            embeddings.append(embedding[0])
        
        if (i + 1) % 50 == 0:
            print(f"Đã xử lý: {i+1}/{len(programs)}")
            
    return np.array(embeddings)

def compute_metrics(embeddings, n_clusters=None, random_state=42):
    dist_matrix = cosine_distances(embeddings)
    n = dist_matrix.shape[0]

    min_dists = [np.min(np.delete(dist_matrix[i], i)) for i in range(n)]
    cdi = float(np.mean(min_dists))

    if n_clusters is None:
        n_clusters = int(np.sqrt(n)) 

    n_clusters = min(n_clusters, n)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )
    labels = kmeans.fit_predict(embeddings)

    cluster_sizes = np.bincount(labels)
    M = np.sum(cluster_sizes)

    probs = cluster_sizes / M
    probs = probs[probs > 0]  

    swdi = float(-np.sum(probs * np.log(probs)))

    return cdi, swdi, n_clusters


def run_multi_file_analysis_with_plot(file_list):
    """
    Process multiple files, compute metrics, and plot the results.
    """
    results = []
    file_names = []

    for file_path in file_list:
        try:
            print(f"\nĐang xử lý file: {file_path}")
            unique_programs = load_programs_from_multiple_files([file_path])
            embeddings = embed_programs(unique_programs)
            cdi, swdi, k = compute_metrics(embeddings)

            results.append((cdi, swdi, k))
            file_names.append(os.path.basename(file_path))

            print(f"CDI: {cdi:.6f}, SWDI: {swdi:.6f}, Clusters: {k}")
        except Exception as e:
            print(f"Lỗi khi xử lý file {file_path}: {e}")

    # Plot the results
    if results:
        cdi_values = [res[0] for res in results]
        swdi_values = [res[1] for res in results]
        cluster_counts = [res[2] for res in results]

        x = range(len(file_names))

        plt.figure(figsize=(10, 6))
        plt.plot(x, cdi_values, marker="o", label="CDI (Local Diversity)")
        plt.plot(x, swdi_values, marker="s", label="SWDI (Global Diversity)")
        plt.bar(x, cluster_counts, alpha=0.5, label="Number of Clusters")

        plt.xticks(x, file_names, rotation=45, ha="right")
        plt.xlabel("Files")
        plt.ylabel("Metrics")
        plt.title("Semantic Diversity Metrics Across Files")
        plt.legend()
        plt.tight_layout()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.show()

if __name__ == "__main__":
    LIST_OF_FILES = [
        "logs/momcts/bi_cvrp/nhv_runtime/v1/samples/samples_1~300.json",
        "logs/meoh/bi_cvrp/nhv_runtime/v2/samples/samples_1~300.json",
        "logs/moead/bi_cvrp/nhv_runtime/v3/samples/samples_1~300.json",
        "logs/mpage/bi_cvrp/nhv_runtime/v4/samples/samples_1~300.json",
        "logs/nsga2/bi_cvrp/nhv_runtime/v4/samples/samples_1~300.json"
    ]

    try:
        run_multi_file_analysis_with_plot(LIST_OF_FILES)
    except Exception as e:
        print(f"Lỗi: {e}")