import numpy as np
import rasterio
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.utils import resample
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

# 加载Spacy的英文模型
nlp = spacy.load('en_core_web_sm')

# 设置字体为 SimHei，解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决保存图像时负号 '-' 显示为方块的问题


# ===================== 图像处理相关函数 =====================

def read_image(image):
    """读取图像数据，支持多种格式"""
    if isinstance(image, str):
        try:
            with rasterio.open(image) as src:
                img = src.read()
                img = np.transpose(img, (1, 2, 0))
        except:
            img = cv2.imread(image)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        if isinstance(image, PIL.Image.Image):
            img = np.array(image)
        else:
            img = image
    return img


def preprocess_image(image):
    """图像预处理"""
    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=-1)
    pixels = image.reshape(-1, image.shape[2])
    pixels = (pixels - pixels.min()) / (pixels.max() - pixels.min())
    return pixels, image.shape


# ===================== 文本处理相关函数 (使用Spacy) =====================

def preprocess_text(text):
    """使用Spacy对文本进行预处理"""
    doc = nlp(text.lower())  # 将文本转为小写并处理
    # 仅移除停用词，但保留非字母字符
    tokens = [token.text for token in doc if not token.is_stop]  # 仅去除停用词
    processed_text = ' '.join(tokens)

    # 打印每条预处理后的文本以进行调试
    # print(f"原始文本: {text}")
    # print(f"预处理后文本: {processed_text}")

    return processed_text

def extract_text_features(texts):
    """提取文本特征"""
    # 预处理所有文本
    processed_texts = [preprocess_text(text) for text in texts]

    # 检查是否所有文本处理后为空
    non_empty_texts = [text for text in processed_texts if len(text) > 0]
    if len(non_empty_texts) == 0:
        raise ValueError("所有文本处理后为空，无法继续进行分析。请检查输入文本或调整预处理规则。")

    # 使用TF-IDF向量化
    vectorizer = TfidfVectorizer(max_features=1000, min_df=1, stop_words=None)
    features = vectorizer.fit_transform(non_empty_texts)

    return features, vectorizer

def get_top_terms_per_cluster(cluster_labels, texts, vectorizer, n_terms=5):
    """获取每个簇的主要特征词"""
    # 预处理文本
    processed_texts = [preprocess_text(text) for text in texts]

    # 获取词频
    terms = vectorizer.get_feature_names_out()

    # 为每个簇收集主要词汇
    cluster_terms = {}
    for cluster_id in range(max(cluster_labels) + 1):
        # 获取该簇的所有文本
        cluster_texts = [text for text, label in zip(processed_texts, cluster_labels) if label == cluster_id]

        # 统计词频
        cluster_words = ' '.join(cluster_texts).split()
        word_freq = Counter(cluster_words)

        # 获取最常见的词
        top_words = [word for word, freq in word_freq.most_common(n_terms)]
        cluster_terms[f'簇 {cluster_id}'] = top_words

    return cluster_terms

def calculate_bcss(features, cluster_centers, cluster_labels):
    """计算蔟间平方和（BCSS）"""
    overall_center = features.mean(axis=0)  # 计算所有数据的中心
    bcss = 0
    for i, center in enumerate(cluster_centers):
        # 计算每个簇中心到总体中心的平方距离，并乘以该簇中样本的数量
        n_samples_in_cluster = np.sum(cluster_labels == i)
        bcss += n_samples_in_cluster * np.sum((center - overall_center) ** 2)
    return bcss
# ===================== 聚类相关函数 =====================

def perform_clustering(data, n_clusters, data_type='image'):
    """执行聚类并返回结果，包括蔟间平方和（BCSS）"""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)

    if data_type == 'image':
        pixels, original_shape = data
        clustered = kmeans.fit_predict(pixels)

        # 计算评估指标
        if len(pixels) > 10000:
            sampled_data, sampled_labels = resample(pixels, clustered, n_samples=10000, random_state=42)
        else:
            sampled_data, sampled_labels = pixels, clustered

        silhouette_avg = silhouette_score(sampled_data, sampled_labels)
        inertia = kmeans.inertia_
        bcss = calculate_bcss(sampled_data, kmeans.cluster_centers_, sampled_labels)

        # 重塑聚类结果
        clustered_image = clustered.reshape(original_shape[0], original_shape[1])
        return clustered_image, silhouette_avg, inertia, kmeans.cluster_centers_, bcss

    else:  # text
        features, vectorizer = data
        clustered = kmeans.fit_predict(features)

        silhouette_avg = silhouette_score(features.toarray(), clustered)
        inertia = kmeans.inertia_
        bcss = calculate_bcss(features.toarray(), kmeans.cluster_centers_, clustered)

        return clustered, silhouette_avg, inertia, kmeans.cluster_centers_, vectorizer, bcss


# ===================== 可视化相关函数 =====================

def create_cluster_visualization(clustered_image, n_clusters):
    """创建图像聚类结果可视化"""
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
    """创建评估指标的折线图"""
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
    """图像聚类主处理函数，增加BCSS"""
    if store_state is None:
        store_state = {"k_values": [], "silhouette_scores": [], "inertia_values": []}

    # 确保 bcss_values 在 store_state 中
    store_state.setdefault("bcss_values", [])

    img = read_image(image)
    pixels, original_shape = preprocess_image(img)

    clustered_image, silhouette_avg, inertia, cluster_centers, bcss = perform_clustering(
        (pixels, original_shape), n_clusters, 'image')

    result_image = create_cluster_visualization(clustered_image, n_clusters)

    if n_clusters not in store_state["k_values"]:
        store_state["k_values"].append(n_clusters)
        store_state["silhouette_scores"].append(silhouette_avg)
        store_state["inertia_values"].append(inertia)
        store_state["bcss_values"].append(bcss)  # 修复添加 BCSS 值

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

    return result_image, metrics_plot, metrics_text, store_state


def cluster_text(text_data, n_clusters, store_state=None):
    """文本聚类主处理函数，增加BCSS"""
    if store_state is None:
        store_state = {"k_values": [], "silhouette_scores": [], "inertia_values": [], "bcss_values": []}

    texts = []  # 存储文本内容

    # 只处理上传文件，判断文件类型并解析
    file_path = text_data.name
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)  # 读取CSV文件
        text_column_name = 'description'  # 假设文本列名为 'description'
        if text_column_name in df.columns:
            texts = df[text_column_name].dropna().astype(str).tolist()
        else:
            raise ValueError(f"数据集未找到文本列 '{text_column_name}'，请检查数据集")

    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)  # 读取Excel文件
        text_column_name = 'description'  # 假设文本列名为 'description'
        if text_column_name in df.columns:
            texts = df[text_column_name].dropna().astype(str).tolist()
        else:
            raise ValueError(f"数据集未找到文本列 '{text_column_name}'，请检查数据集")

    elif file_path.endswith('.txt'):
        with open(file_path, 'r', encoding='utf-8') as file:
            # 逐行读取TXT文件的内容，每一行作为一个独立的文本
            texts = [line.strip() for line in file.readlines() if line.strip()]

    else:
        raise ValueError("不支持的文件格式，请上传 CSV, XLSX 或 TXT 文件")

    if len(texts) == 0:
        raise ValueError("输入的文件中没有有效的文本内容")

    # 自动调整 n_clusters，以免 n_clusters 大于文本数
    if len(texts) < n_clusters:
        print(f"文本数量 ({len(texts)}) 少于簇的数量 ({n_clusters})，将簇的数量调整为 {len(texts)}。")
        n_clusters = len(texts)  # 自动调整簇的数量

    # 提取特征
    features, vectorizer = extract_text_features(texts)

    # 执行聚类
    clustered, silhouette_avg, inertia, cluster_centers, vectorizer, bcss = perform_clustering(
        (features, vectorizer), n_clusters, 'text')

    # 获取每个簇的主要词汇
    cluster_terms = get_top_terms_per_cluster(clustered, texts, vectorizer)

    # 更新状态
    if n_clusters not in store_state["k_values"]:
        store_state["k_values"].append(n_clusters)
        store_state["silhouette_scores"].append(silhouette_avg)
        store_state["inertia_values"].append(inertia)
        store_state["bcss_values"].append(bcss)  # 修复添加 BCSS 值

    # 创建指标图
    metrics_plot = create_metrics_plot(
        store_state["k_values"],
        store_state["silhouette_scores"],
        store_state["inertia_values"]
    )

    # 创建结果文本
    cluster_distribution = Counter(clustered)
    result_text = "聚类结果:\n\n"
    for cluster_id in range(n_clusters):
        if f'簇 {cluster_id}' in cluster_terms:  # 确保簇存在
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

    return result_text, metrics_plot, metrics_text, store_state



# ===================== Gradio界面 =====================

def create_interface():
    """创建Gradio界面"""
    # 默认图片路径（假设放在当前目录下）
    default_image_path = "test.png"  # 替换为你的默认图片路径
    if not os.path.exists(default_image_path):
        # 生成一个随机的默认图像以供测试
        default_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        PIL.Image.fromarray(default_image).save(default_image_path)

    # 默认文本数据文件路径
    default_text_file = "test.txt"
    if not os.path.exists(default_text_file):
        # 如果不存在默认文件，则创建一个随机文本文件
        with open(default_text_file, 'w') as f:
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
    else:
        notebook_html = "<p>无法找到指定的 HTML 文件。</p>"
    # 读取 HTML5 文件的内容
    html_path = "5NBA画像.html"
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            notebook_html5 = f.read()
    else:
        notebook_html = "<p>无法找到指定的 HTML 文件。</p>"

    with gr.Blocks() as interface:
        gr.Markdown("# 多模态K均值聚类分析工具——数据分析课程第16组汇报  崔灿一星、邓永涛、冯梦如")

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
                state_image = gr.State({"k_values": [], "silhouette_scores": [], "inertia_values": [], "bcss_values": []})

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
                state_text = gr.State({"k_values": [], "silhouette_scores": [], "inertia_values": [], "bcss_values": []})
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
