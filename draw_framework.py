import os
import webbrowser

html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>研究框架图 - MMB-DTI</title>
    <!-- 引入 Mermaid -->
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({ 
          startOnLoad: true, 
          theme: 'default',
          securityLevel: 'loose',
          flowchart: { useMaxWidth: false, htmlLabels: true, curve: 'basis' }
      });
    </script>
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background-color: #f0f2f5; 
            display: flex; 
            flex-direction: column;
            align-items: center; 
            padding: 40px; 
            margin: 0;
        }
        h1 {
            color: #333;
            margin-bottom: 30px;
        }
        .canvas-container { 
            background: white; 
            padding: 40px; 
            border-radius: 12px; 
            box-shadow: 0 8px 30px rgba(0,0,0,0.08); 
            max-width: 1200px; 
            width: 100%; 
            display: flex; 
            justify-content: center; 
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <h1>基于多模态大模型与状态空间机制的药物-靶点亲和力预测研究</h1>
    <div class="canvas-container">
        <div class="mermaid">
        flowchart TB
            %% Core Theme
            Root("<b>核心目标：深度学习驱动的多模态药物-靶点亲和力预测</b>")

            %% Subgraphs for the 3 Innovation Modules
            
            subgraph Mod1 ["创新模块一：基于 MMB-DTI 的多模态双向注意力 DTA 预测"]
                direction TB
                Input1[药物双模态输入<br/>SMILES序列 & 分子图]
                Input2[蛋白双模态输入<br/>氨基酸序列 & 接触图]
                
                Enc1[药物表征提取<br/>ChemBERTa & GAT]
                Enc2[蛋白表征提取<br/>ESM-2 & GCN]
                
                Input1 --> Enc1
                Input2 --> Enc2
                
                subgraph Features ["混合特征提取层"]
                    direction LR
                    Mamba[长程全局建模<br/>Selective State Space Model]
                    BiLSTM[局部相邻平滑<br/>BiLSTM 网络]
                end
                
                Enc1 & Enc2 --> Features
                
                Attn[跨模态交互层<br/>Bi-directional Cross-Attention]
                Features --> Attn
                
                Pred1[亲和力预测<br/>输出预测分数 pKd等]
                Attn --> Pred1
            end
            
            subgraph Mod2 ["创新模块二：基于 3D 口袋感知与大语言模型协同的亲和力对齐"]
                direction TB
                Coord[空间结构先验<br/>PDBbind 蛋白-配体 3D 坐标]
                
                Net3D[三维图神经网络学习<br/>SphereNet / PNA]
                LLM[大语言模型大本底<br/>ESM-3 序列模态]
                
                Coord --> Net3D & LLM
                
                CL[多模态对齐<br/>对比学习 Contrastive Learning]
                Net3D & LLM --> CL
                
                Distill[空间特征隐式植入<br/>3D拓扑向一维序列的知识蒸馏]
                CL --> Distill
            end
            
            subgraph Mod3 ["创新模块三：基于因果干预的 OOD 泛化与可解释活性图谱生成"]
                direction TB
                Causal[构建亲和力因果图<br/>剥离结构相似度等混杂因子]
                
                Intervene[去偏向训练框架<br/>后门调整 & 反事实数据增强]
                Causal --> Intervene
                
                Obj1[冷启动靶点预测<br/>提升零样本下的 OOD 泛化能力]
                Obj2[机制透明化<br/>生成高分辨率因果注意力热力图]
                
                Intervene --> Obj1 & Obj2
            end
            
            %% Relationships
            Root --> Mod1
            Mod1 == 升级扩展 ==> Mod2
            Mod2 == 机制校正 ==> Mod3

            %% Styling Elements
            classDef main fill:#f1f8e9,stroke:#33691e,stroke-width:2px,font-weight:bold
            classDef mod fill:#e3f2fd,stroke:#1565c0,stroke-width:2px,font-weight:bold,color:#000
            classDef input fill:#fce4ec,stroke:#880e4f,stroke-width:1px
            classDef process fill:#fff8e1,stroke:#ff8f00,stroke-width:1px
            classDef feature fill:#e0f7fa,stroke:#006064,stroke-width:1px
            classDef output fill:#e8f5e9,stroke:#1b5e20,stroke-width:1px
            
            class Root main
            class Mod1,Mod2,Mod3 mod
            class Input1,Input2,Coord,Causal input
            class Enc1,Enc2,Attn,Net3D,LLM,CL,Intervene process
            class Mamba,BiLSTM feature
            class Pred1,Distill,Obj1,Obj2 output
        </div>
    </div>
</body>
</html>
"""

# The path for our HTML canvas file
html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Research_Framework_Canvas.html")

# Write the HTML content to file
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"成功生成 Web Canvas 画布网页: {html_path}")

# Open the HTML file in the user's default web browser
url = f"file:///{html_path.replace(chr(92), '/')}"
print(f"🚀 正在自动通过默认浏览器打开展示界面...")
webbrowser.open(url)
