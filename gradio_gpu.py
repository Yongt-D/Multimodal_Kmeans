import torch
import rasterio
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from datetime import datetime
import gradio as gr
import cv2
import io
import PIL.Image
from matplotlib.figure import Figure
import plotly.graph_objects as go
import plotly.express as px
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from collections import Counter
import re
import spacy
import matplotlib
import os
import numpy as np

# 检查GPU是否可用并设置设备
if not torch.cuda.is_available():
    raise RuntimeError("This application requires a GPU to run")
device = torch.device('cuda')
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# 设置字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# 加载Spacy模型
nlp = spacy.load('en_core_web_sm')


# ===================== 图像处理相关函数 =====================

def read_image(image):
    """读取图像并转移到GPU"""
    if isinstance(image, str):
        try:
            with rasterio.open(image) as src:
                img = src.read()
                img = torch.from_numpy(img).permute(1, 2, 0)
        except:
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.from_numpy(img)
    else:
        if isinstance(image, PIL.Image.Image):
            img = torch.from_numpy(np.array(image))
        else:
            img = torch.from_numpy(image)
    return img.float().to(device)


def preprocess_image(image):
    """图像预处理 - GPU版本"""
    if len(image.shape) == 2:
        image = image.unsqueeze(-1)
    pixels = image.reshape(-1, image.shape[2])
    pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min())
    return pixels, image.shape


# ===================== 文本处理相关函数 =====================

def preprocess_text(text):
    """文本预处理"""
    doc = nlp(text.lower())
    tokens = [token.text for token in doc if not token.is_stop]
    return ' '.join(tokens)


def extract_text_features(texts):
    """提取文本特征并转移到GPU"""
    processed_texts = [preprocess_text(text) for text in texts]

    non_empty_texts = [text for text in processed_texts if len(text) > 0]
    if not non_empty_texts:
        raise ValueError("所有文本处理后为空")

    vectorizer = TfidfVectorizer(max_features=1000, min_df=1, stop_words=None)
    features = vectorizer.fit_transform(non_empty_texts)

    # 转换为GPU张量
    features_tensor = torch.from_numpy(features.toarray()).float().to(device)
    return features_tensor, vectorizer


def get_top_terms_per_cluster(cluster_labels, texts, vectorizer, n_terms=5):
    """获取每个簇的主要特征词"""
    processed_texts = [preprocess_text(text) for text in texts]
    terms = vectorizer.get_feature_names_out()

    # 将cluster_labels转移到CPU进行处理
    cluster_labels_cpu = cluster_labels.cpu().numpy()

    cluster_terms = {}
    for cluster_id in range(int(cluster_labels.max().item()) + 1):
        cluster_texts = [text for text, label in zip(processed_texts, cluster_labels_cpu) if label == cluster_id]
        cluster_words = ' '.join(cluster_texts).split()
        word_freq = Counter(cluster_words)
        top_words = [word for word, freq in word_freq.most_common(n_terms)]
        cluster_terms[f'簇 {cluster_id}'] = top_words

    return cluster_terms


def calculate_bcss(features, cluster_centers, cluster_labels):
    """计算簇间平方和 (BCSS) - GPU版本"""
    overall_center = features.mean(dim=0)
    bcss = torch.tensor(0.0, device=device)
    for i, center in enumerate(cluster_centers):
        n_samples_in_cluster = (cluster_labels == i).sum()
        bcss += n_samples_in_cluster * torch.sum((center - overall_center) ** 2)
    return bcss


# ===================== GPU K-means实现 =====================

class TorchKMeans:
    def __init__(self, n_clusters, max_iter=300, random_state=42):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.cluster_centers_ = None
        self.inertia_ = None
        torch.manual_seed(random_state)
        torch.cuda.manual_seed(random_state)

    def fit_predict(self, X):
        n_samples = X.size(0)

        # 随机初始化聚类中心
        idx = torch.randperm(n_samples, device=device)[:self.n_clusters]
        self.cluster_centers_ = X[idx].clone()

        for _ in range(self.max_iter):
            # 计算距离矩阵 - GPU上进行
            distances = torch.cdist(X, self.cluster_centers_)

            # 分配样本到最近的聚类中心
            labels = torch.argmin(distances, dim=1)

            # 更新聚类中心
            new_centers = torch.zeros_like(self.cluster_centers_, device=device)
            for k in range(self.n_clusters):
                mask = labels == k
                if mask.any():
                    new_centers[k] = X[mask].mean(dim=0)
                else:
                    new_centers[k] = self.cluster_centers_[k]

            # 检查收敛
            if torch.allclose(new_centers, self.cluster_centers_):
                break

            self.cluster_centers_ = new_centers

        # 计算惯性
        distances = torch.cdist(X, self.cluster_centers_)
        self.inertia_ = torch.sum(torch.min(distances, dim=1)[0])

        return labels


# ===================== 聚类相关函数 =====================

def perform_clustering(data, n_clusters, data_type='image'):
    """执行聚类 - GPU版本"""
    if data_type == 'image':
        pixels, original_shape = data
        kmeans = TorchKMeans(n_clusters=n_clusters)
        clustered = kmeans.fit_predict(pixels)

        # 采样计算silhouette score
        if len(pixels) > 10000:
            idx = torch.randperm(len(pixels), device=device)[:10000]
            sampled_data = pixels[idx]
            sampled_labels = clustered[idx]
        else:
            sampled_data, sampled_labels = pixels, clustered

        # 转到CPU计算silhouette score
        silhouette_avg = silhouette_score(
            sampled_data.cpu().numpy(),
            sampled_labels.cpu().numpy()
        )

        inertia = kmeans.inertia_.item()
        bcss = calculate_bcss(sampled_data, kmeans.cluster_centers_, sampled_labels).item()

        clustered_image = clustered.reshape(original_shape[0], original_shape[1])
        return clustered_image.cpu().numpy(), float(silhouette_avg), float(
            inertia), kmeans.cluster_centers_.cpu().numpy(), float(bcss)

    else:  # text
        features, vectorizer = data
        kmeans = TorchKMeans(n_clusters=n_clusters)
        clustered = kmeans.fit_predict(features)

        # 计算评估指标
        silhouette_avg = silhouette_score(
            features.cpu().numpy(),
            clustered.cpu().numpy()
        )

        inertia = kmeans.inertia_.item()
        bcss = calculate_bcss(features, kmeans.cluster_centers_, clustered).item()

        return clustered, float(silhouette_avg), float(inertia), kmeans.cluster_centers_, vectorizer, float(bcss)


# ===================== 可视化相关函数 =====================

def create_cluster_visualization(clustered_image, n_clusters):
    """创建聚类结果可视化"""
    fig = Figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    im = ax.imshow(clustered_image, cmap='tab10', interpolation='nearest')

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label('簇标签')
    cbar.set_ticks([i + 0.5 for i in range(n_clusters)])
    cbar.set_ticklabels([f'簇 {i}' for i in range(n_clusters)])

    ax.set_title('聚类结果')
    ax.axis('off')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    return PIL.Image.open(buf)


def create_metrics_plot(k_values, silhouette_scores, inertia_values):
    """创建评估指标图"""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=k_values,
        y=silhouette_scores,
        name='轮廓系数',
        mode='lines+markers',
        yaxis='y1'
    ))

    fig.add_trace(go.Scatter(
        x=k_values,
        y=inertia_values,
        name='惯性',
        mode='lines+markers',
        yaxis='y2'
    ))

    fig.update_layout(
        title='聚类指标 vs. 簇数量',
        xaxis_title='簇数量 (K)',
        yaxis_title='轮廓系数',
        yaxis2=dict(
            title='惯性',
            overlaying='y',
            side='right'
        ),
        hovermode='x'
    )

    return fig


# ===================== 主处理函数 =====================

def cluster_image(image, n_clusters, store_state=None):
    """图像聚类主函数"""
    if store_state is None:
        store_state = {"k_values": [], "silhouette_scores": [], "inertia_values": [], "bcss_values": []}

    img = read_image(image)
    pixels, original_shape = preprocess_image(img)

    clustered_image, silhouette_avg, inertia, cluster_centers, bcss = perform_clustering(
        (pixels, original_shape), n_clusters, 'image')

    result_image = create_cluster_visualization(clustered_image, n_clusters)

    if n_clusters not in store_state["k_values"]:
        store_state["k_values"].append(n_clusters)
        store_state["silhouette_scores"].append(silhouette_avg)
        store_state["inertia_values"].append(inertia)
        store_state["bcss_values"].append(bcss)

    metrics_plot = create_metrics_plot(
        store_state["k_values"],
        store_state["silhouette_scores"],
        store_state["inertia_values"]
    )

    metrics_text = f"""
    聚类评估指标 (K={n_clusters}):
    轮廓系数: {silhouette_avg:.4f}
    惯性: {inertia:.4f}
    蔟间平方和 (BCSS): {bcss:.4f}
    """

    # 清理GPU内存
    torch.cuda.empty_cache()

    return result_image, metrics_plot, metrics_text, store_state


def cluster_text(text_data, n_clusters, store_state=None):
    """文本聚类主函数"""
    if store_state is None:
        store_state = {"k_values": [], "silhouette_scores": [], "inertia_values": [], "bcss_values": []}

    texts = []
    file_path = text_data.name

    # 读取文本数据
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
        text_column_name = 'description'
        if text_column_name in df.columns:
            texts = df[text_column_name].dropna().astype(str).tolist()
        else:
            raise ValueError(f"未找到文本列 '{text_column_name}'")
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
        text_column_name = 'description'
        if text_column_name in df.columns:
            texts = df[text_column_name].dropna().astype(str).tolist()
        else:
            raise ValueError(f"未找到文本列 '{text_column_name}'")
    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            texts = [line.strip() for line in file.readlines() if line.strip()]
    else:
        raise ValueError("不支持的文件格式")

    if not texts:
        raise ValueError("没有有效的文本内容")

    if len(texts) < n_clusters:
        n_clusters = len(texts)
        print(f"文本数量 ({len(texts)}) 少于簇数量，已调整为 {n_clusters}")

    features, vectorizer = extract_text_features(texts)

    clustered, silhouette_avg, inertia, cluster_centers, vectorizer, bcss = perform_clustering(
        (features, vectorizer), n_clusters, 'text')

    cluster_terms = get_top_terms_per_cluster(clustered, texts, vectorizer)

    if n_clusters not in store_state["k_values"]:
        store_state["k_values"].append(n_clusters)
        store_state["silhouette_scores"].append(silhouette_avg)
        store_state["inertia_values"].append(inertia)
        store_state["bcss_values"].append(bcss)

    metrics_plot = create_metrics_plot(
        store_state["k_values"],
        store_state["silhouette_scores"],
        store_state["inertia_values"]
    )

    cluster_distribution = Counter(clustered.cpu().numpy())
    result_text = "聚类结果:\n\n"
    for cluster_id in range(n_clusters):
        if f'簇 {cluster_id}' in cluster_terms:
            result_text += f"簇 {cluster_id} ({cluster_distribution[cluster_id]} 篇文本):\n"

            result_text += f"主要词汇: {', '.join(cluster_terms[f'簇 {cluster_id}'])}\n\n"
        else:
            result_text += f"簇 {cluster_id} 没有生成有效的文本。\n\n"

    metrics_text = f"""
    聚类评估指标 (K={n_clusters}):
    轮廓系数: {silhouette_avg:.4f}
    惯性: {inertia:.4f}
    蔟间平方和 (BCSS): {bcss:.4f}
    """

    # 清理GPU内存
    torch.cuda.empty_cache()

    return result_text, metrics_plot, metrics_text, store_state

# ===================== Gradio界面 =====================

def create_interface():
    """创建Gradio界面"""
    # 创建默认图片
    default_image_path = "test.png"
    if not os.path.exists(default_image_path):
        default_image = torch.randint(0, 255, (256, 256, 3), dtype=torch.uint8)
        PIL.Image.fromarray(default_image.numpy()).save(default_image_path)

    # 创建默认文本文件
    default_text_file = "test.txt"
    if not os.path.exists(default_text_file):
        with open(default_text_file, 'w', encoding='utf-8') as f:
            f.write("This is a sample text file for testing clustering.\n")
            f.write("Machine learning is great.\n")
            f.write("Artificial intelligence is transforming industries.\n")
    # 读取 HTML 文件的内容
    html_path = "1肥胖水平估计.html"
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            notebook_html = f.read()
    else:
        notebook_html = "<p>无法找到指定的 HTML 文件。</p>"
    # 读取 HTML2 文件的内容
    html_path = "2糖尿病诊疗数据分析.html"
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            notebook_html2 = f.read()
    else:
        notebook_html = "<p>无法找到指定的 HTML 文件。</p>"
    # 读取 HTML3 文件的内容
    html_path = "3在线零售数据分析.html"
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            notebook_html3 = f.read()
    else:
        notebook_html = "<p>无法找到指定的 HTML 文件。</p>"
    # 读取 HTML4 文件的内容
    html_path = "4文本聚类.html"
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            notebook_html4 = f.read()
    # 读取 HTML5 文件的内容
    html_path = "5NBA画像.html"
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            notebook_html5 = f.read()
    else:
        notebook_html = "<p>无法找到指定的 HTML 文件。</p>"

    with gr.Blocks() as interface:
        gr.Markdown("# 多模态K均值聚类分析工具(GPU)——数据分析课程第16组汇报 崔灿一星、邓永涛、冯梦如")

        with gr.Tabs():
            # 图像聚类标签页
            with gr.TabItem("图像聚类"):
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(label="上传图像", type="numpy", value=default_image_path)
                        n_clusters_image = gr.Slider(
                            minimum=2,
                            maximum=20,
                            step=1,
                            value=5,
                            label="簇数量 (K)"
                        )
                        cluster_image_btn = gr.Button("执行图像聚类")

                    with gr.Column():
                        output_image = gr.Image(label="聚类结果")
                        metrics_text_image = gr.Textbox(label="评估指标", lines=3)

                metrics_plot_image = gr.Plot(label="聚类指标")
                state_image = gr.State(
                    {"k_values": [], "silhouette_scores": [], "inertia_values": [], "bcss_values": []})

            # 文本聚类标签页
            with gr.TabItem("文本聚类"):
                with gr.Row():
                    with gr.Column():
                        input_text = gr.File(label="上传文本文件 (CSV, XLSX, TXT)", value=default_text_file)
                        n_clusters_text = gr.Slider(
                            minimum=2,
                            maximum=20,
                            step=1,
                            value=5,
                            label="簇数量 (K)"
                        )
                        cluster_text_btn = gr.Button("执行文本聚类")

                    with gr.Column():
                        output_text = gr.Textbox(label="聚类结果", lines=10)
                        metrics_text_text = gr.Textbox(label="评估指标", lines=3)

                metrics_plot_text = gr.Plot(label="聚类指标")
                state_text = gr.State(
                    {"k_values": [], "silhouette_scores": [], "inertia_values": [], "bcss_values": []})
            # 新增 Notebook 展示标签页
            with gr.TabItem("例1-肥胖水平估计"):
                gr.Markdown("### 例1-肥胖水平估计")
                gr.HTML(notebook_html)
            # 新增 Notebook 展示标签页
            with gr.TabItem("例2-糖尿病诊疗数据分析"):
                gr.Markdown("### 例2-糖尿病诊疗数据分析")
                gr.HTML(notebook_html2)
            # 新增 Notebook 展示标签页
            with gr.TabItem("例3-在线零售数据分析"):
                gr.Markdown("### 例3-在线零售数据分析")
                gr.HTML(notebook_html3)
            # 新增 Notebook 展示标签页
            with gr.TabItem("例4-新闻语料聚类"):
                gr.Markdown("### 例4-新闻语料聚类")
                gr.HTML(notebook_html4)
            # 新增 Notebook 展示标签页
            with gr.TabItem("例5-NBA画像"):
                gr.Markdown("### 例5-NBA画像")
                gr.HTML(notebook_html5)
        # 设置事件处理
        cluster_image_btn.click(
            fn=cluster_image,
            inputs=[input_image, n_clusters_image, state_image],
            outputs=[output_image, metrics_plot_image, metrics_text_image, state_image]
        )

        cluster_text_btn.click(
            fn=cluster_text,
            inputs=[input_text, n_clusters_text, state_text],
            outputs=[output_text, metrics_plot_text, metrics_text_text, state_text]
        )

    return interface

# 启动应用
if __name__ == "__main__":
    interface = create_interface()
    interface.launch(share=True)