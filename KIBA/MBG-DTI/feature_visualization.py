import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import umap
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# 确保能导入项目中的文件
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(root_dir)
from architectures import MambaBiLSTMModel
from dataset import DTIDataset, collate_dti

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_features(model, dataloader, device):
    """
    运行整个测试集，通过 Pytorch Hook 提取最终分类器前的特征 (combined 变量)
    """
    model.eval()
    features_list = []
    labels_list = []
    
    # 定义全局变量保存 hook 截获的特征
    extracted_features = {}
    
    def hook_fn(module, input, output):
        # input 是一个 tuple，input[0] 就是喂给 classifier 第一层的张量，即特征 combined
        # combined 的维度在 MambaBiLSTMModel 中是 (B, hidden_dim * 3) -> 256 * 3 = 768
        extracted_features['features'] = input[0].detach().cpu().numpy()

    # 将 hook 挂载到 classifier 的第一层 (nn.Linear)
    # 根据 architectures.yaml classifier 是一个 Sequential
    handle = model.classifier[0].register_forward_pre_hook(hook_fn)
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(dataloader):
            # 这里需要根据你的具体 Dataloader 格式拆包
            # 假设 batch_data 包含 drug_input, prot_input, label
            # 这部分代码可能需要你根据实际的 train/test.py 中的写法进行微调
            drug_input, prot_input, labels = batch_data
            
            # 将数据转移到 GPU
            d_ids, d_mask, d_graph = drug_input
            d_ids, d_mask = d_ids.to(device), d_mask.to(device)
            d_graph = d_graph.to(device)
            drug_input = (d_ids, d_mask, d_graph)
            
            p_ids, p_mask, p_edge = prot_input
            p_ids, p_mask = p_ids.to(device), p_mask.to(device)
            p_edge = p_edge.to(device)
            prot_input = (p_ids, p_mask, p_edge)
            
            labels = labels.to(device)
            
            # 前向传播，hook 会自动截获特征
            _ = model(drug_input, prot_input)
            
            # 保存特征和标签
            features_list.append(extracted_features['features'])
            labels_list.append(labels.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f"提取批次 {batch_idx}...")
                
    # 移除 hook
    handle.remove()
    
    # 拼接所有批次的特征和标签
    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)
    
    return all_features, all_labels

def plot_reduction(features, labels, method='UMAP', save_path=None):
    """
    使用 t-SNE 或 UMAP 对高维特征进行降维并可视化
    """
    print(f"正在使用 {method} 进行降维处理 (特征维度: {features.shape})...")
    
    if method == 't-SNE':
        reducer = TSNE(n_components=2, random_state=42, n_iter=1000, perplexity=30)
    elif method == 'UMAP':
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    else:
        raise ValueError("Method 必须是 't-SNE' 或 'UMAP'")
        
    reduced_features = reducer.fit_transform(features)
    
    print("降维完成，开始绘制图像...")
    
    plt.figure(figsize=(10, 8))
    
    # KIBA 是连续值，但用户说这是个"二分类"任务。
    # 假设二分类标签是 0 和 1 (或者可以设置一个阈值划分)。
    # 我们这里默认 labels 是 0 和 1 的离散值。
    
    # 定义高颜值的科研配色：负样本湖蓝色，正样本绯红色
    palette = {0: '#1f77b4', 1: '#d62728'} 
    
    # 使用 seaborn 渲染散点图
    sns.scatterplot(
        x=reduced_features[:, 0], 
        y=reduced_features[:, 1],
        hue=labels,
        palette=palette,
        s=15,          # 点的大小
        alpha=0.7,     # 透明度防止重叠
        edgecolor=None # 移除边框让过渡更自然
    )
    
    plt.title(f'Representation Space Analysis via {method}', fontsize=16, fontweight='bold')
    plt.xlabel(f'{method} Dimension 1', fontsize=14)
    plt.ylabel(f'{method} Dimension 2', fontsize=14)
    
    # 美化图例
    handles, previous_labels = plt.gca().get_legend_handles_labels()
    new_labels = ['Negative (Non-interacting)', 'Positive (Interacting)']
    plt.legend(handles=handles, labels=new_labels, title="Interaction Type", title_fontsize='13', fontsize='11', loc='best')
    
    # 去除顶部和右侧的边框线
    sns.despine()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图像已保存至: {save_path}")
    
    plt.show()

if __name__ == '__main__':
    # ==========================
    # 1. 准备配置和路径
    # ==========================
    # 这里请修改为您实际的 pth 路径
    model_path = './train_result/model_fold_1.pth'
    
    print("1. 加载模型结构...")
    # 加载模型（参数需要和训练时保持一致）
    model = MambaBiLSTMModel(drug_dim=256, prot_dim=512, hidden_dim=256, fine_tune=False)
    
    print(f"2. 加载模型权重: {model_path}...")
    state_dict = torch.load(model_path, map_location=device)
    # 如果是用 DataParallel 训练的，可能键名带有 'module.' 前缀，做一下处理
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.to(device)
    
    # ==========================
    # 3. 准备测试数据 DataLoader
    # ==========================
    print("3. 准备数据 DataLoader...")
    data_path = '../../data/KIBA.txt'
    
    print("加载 Tokenizers...")
    smi_tokenizer = AutoTokenizer.from_pretrained('seyonec/ChemBERTa-zinc-base-v1')
    prot_tokenizer = AutoTokenizer.from_pretrained('facebook/esm2_t6_8M_UR50D')
    
    print(f"装载数据集: {data_path}")
    test_dataset = DTIDataset(data_path, smi_tokenizer, prot_tokenizer, max_len_drug=128, max_len_prot=350)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_dti)
    
    if test_loader is not None:
        # ==========================
        # 4. 提取特征
        # ==========================
        print("4. 开始提取高维特征...")
        features, labels = extract_features(model, test_loader, device)
        print(f"特征提取完毕！特征形状: {features.shape}, 标签形状: {labels.shape}")
        
        # 将提取的特征保存下来，以后就不用每次都跑模型了
        save_dir = os.path.dirname(model_path)
        features_path = os.path.join(save_dir, 'extracted_features.npy')
        labels_path = os.path.join(save_dir, 'extracted_labels.npy')
        np.save(features_path, features)
        np.save(labels_path, labels)
        print(f"特征和标签已保存到: {features_path} 和 {labels_path}")
        
        # ==========================
        # 5. 可视化降维
        # ==========================
        # 画 t-SNE
        tsne_save_path = os.path.join(save_dir, 'tsne_visualization.png')
        plot_reduction(features, labels, method='t-SNE', save_path=tsne_save_path)
        
        # 画 UMAP
        umap_save_path = os.path.join(save_dir, 'umap_visualization.png')
        plot_reduction(features, labels, method='UMAP', save_path=umap_save_path)
    else:
        print("请在脚本中补充 DataLoader 的加载代码后再次运行！")
