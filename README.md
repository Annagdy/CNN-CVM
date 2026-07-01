# Classificação da Maturação Cervical Vertebral (CVM) em Radiografias Cefalométricas Laterais

Classificação automatizada dos seis estágios de maturação das vértebras cervicais (CVM-S1 a CVM-S6) utilizando aprendizado profundo e transferência de conhecimento (*Transfer Learning*) sobre o conjunto de dados *Aariz Cephalometric Dataset*.

> Este repositório está associado a um artigo acadêmico, que investiga a aplicação de redes neurais convolucionais para auxiliar no diagnóstico ortodôntico, com ênfase na análise de robustez entre diferentes equipamentos de aquisição radiográfica.

---

## Contexto Clínico

A avaliação da Maturação das Vértebras Cervicais (*Cervical Vertebral Maturation* — CVM) é um método amplamente utilizado em ortodontia para determinar a idade esquelética e o timing ideal para intervenções ortopédicas faciais. O método classifica o desenvolvimento das vértebras C2, C3 e C4 em seis estágios (CS1 a CS6) com base em alterações morfológicas observáveis em radiografias cefalométricas laterais.

A classificação manual desses estágios é subjetiva, demorada e dependente da experiência do examinador, além de ser sensível às variações de qualidade e contraste introduzidas por diferentes equipamentos de raio-X — fenômeno conhecido como *domain shift*.

Este projeto propõe uma abordagem computacional para automatizar essa classificação, reduzindo a subjetividade e aumentando a reprodutibilidade do diagnóstico.

---

## Dataset

O **[Aariz Cephalometric Dataset](https://doi.org/10.6084/m9.figshare.27986417.v1)** (Khalid et al., 2025) é atualmente a coleção pública mais diversa para análise cefalométrica e classificação CVM, contando com:

- **~1.000 radiografias cefalométricas laterais (LCRs)**
- **7 equipamentos de raio-X distintos** com diferentes resoluções e características de contraste
- **Anotações de *ground truth*** para os 6 estágios CVM, realizadas por especialistas clínicos
- Divisão padronizada em **treino (700)**, **validação (150)** e **teste (150)**

A estrutura do dataset segue o formato original do Aariz:

```
Aariz/
├── train/
│   ├── Cephalograms/           # Imagens radiográficas
│   └── Annotations/
│       ├── Cephalometric Landmarks/
│       │   ├── Junior Orthodontists/
│       │   └── Senior Orthodontists/
│       └── CVM Stages/         # Anotações JSON dos estágios CVM
├── valid/
│   ├── Cephalograms/
│   └── Annotations/
│       └── CVM Stages/
├── test/
│   ├── Cephalograms/
│   └── Annotations/
│       └── CVM Stages/
├── cephalogram_machine_mappings.csv   # Mapeamento ceph_id -> equipamento
└── Readme.txt
```

---

## Metodologia

### Pipeline

O projeto implementa duas abordagens principais para a classificação CVM:

#### 1. Pipeline em Dois Estágios (`pipeline_cvm.py`)

- **Estágio 1 — YOLOv8n (Few-Shot):** Detecção e extração da região de interesse (ROI) contendo as vértebras C2-C4.
- **Estágio 2 — ResNet-50 com fine-tuning:** Classificação dos 6 estágios CVM a partir das regiões extraídas.

#### 2. Classificação Direta (`train_cvm_improved.py`)

- Arquiteturas suportadas: **ResNet-50**, **ViT-B/16**, **ViT-B/32**, **ViT-L/16**
- Técnicas avançadas implementadas:
  - **Focal Loss** com *label smoothing* para lidar com desbalanceamento severo entre classes
  - **WeightedRandomSampler** para *oversampling* de classes minoritárias
  - **Data augmentation** robusta (rotações de até ±15°, *RandomErasing*, *ColorJitter*, *RandomAffine*)
  - **Mixup augmentation** durante o treinamento
  - **CosineAnnealingWarmRestarts** como agendador de taxa de aprendizado
  - **Test-Time Augmentation (TTA)** na avaliação
  - *Early stopping* com base na loss de validação

### Pré-processamento

Todas as imagens passam por um fluxo de pré-processamento para padronização entre os diferentes equipamentos:

1. **Redimensionamento** para 224×224 pixels
2. **CLAHE** (*Contrast Limited Adaptive Histogram Equalization*) no canal L do espaço de cor LAB, realçando estruturas ósseas e mitigando variações de iluminação entre equipamentos
3. **Normalização** com médias e desvios padrão do ImageNet

### Estratégias de Validação

#### Divisão Padrão (*Random Split*)

Divisão clássica 70/15/15 com pesos compensatórios (*class_weight*) na função de perda para lidar com o desbalanceamento natural das classes biológicas (ex.: CVM-S5 muito mais frequente que CVM-S1).

#### Leave-One-Equipment-Out (LOEO) Cross-Validation

Para avaliar a real capacidade de generalização, implementou-se um protocolo LOEO: o modelo é treinado em 6 equipamentos e testado no 7º (totalmente desconhecido). O ciclo se repete para cada um dos 7 equipamentos, fornecendo uma medida robusta de quão bem o modelo generaliza para fontes de imagem nunca vistas.

#### K-Fold Cross-Validation (Estratificada)

Validação cruzada com K folds (padrão: 5) preservando a proporção de classes em cada fold, oferecendo uma estimativa estável do desempenho independentemente da divisão treino/teste.

---

## Arquivos do Projeto

| Arquivo | Descrição |
|---|---|
| `pipeline_cvm.py` | Pipeline principal em dois estágios (YOLO + ResNet) com suporte a LOEO |
| `train_cvm_improved.py` | Pipeline otimizado com técnicas avançadas (Focal Loss, Mixup, TTA, K-Fold, ViT) |
| `cvm_classification.ipynb` | Notebook inicial de classificação com ResNet-50 |
| `classificacaoCVM.ipynb` | Notebook robusto com análise por equipamento e LOEO |
| `best_cvm_model.pth` | Modelo salvo da classificação CVM |
| `best_resnet_cvm.pth` | Melhor modelo ResNet-50 treinado |
| `best_yolo_cvm.pt` | Melhor modelo YOLO para detecção de ROI |
| `resultados_cvm.csv` | Resultados da classificação por classe |
| `resultados/` | Diretório com gráficos de treinamento e matrizes de confusão |
| `clasificacaoPorEquipamento7/` | Modelos e notebooks para classificação por equipamento |

---

## Como Executar

### Dependências

```
torch torchvision tqdm scikit-learn matplotlib seaborn pandas Pillow opencv-python ultralytics
```

### Pipeline Completo

```bash
# Pipeline em dois estágios (YOLO + ResNet)
python pipeline_cvm.py --stage all

# Pipeline otimizado
python train_cvm_improved.py --stage all --epochs 150 --device cuda

# Apenas treinar classificador
python train_cvm_improved.py --stage train_classifier --epochs 100 --model resnet50

# Avaliar modelo treinado
python train_cvm_improved.py --stage evaluate

# K-Fold Cross-Validation
python train_cvm_improved.py --stage kfold --k-folds 5 --kfold-epochs 80

# LOEO Cross-Validation
python pipeline_cvm.py --stage loeo
```

### Uso via Notebook

Os notebooks `classificacaoCVM.ipynb` e `cvm_classification.ipynb` oferecem uma interface interativa para explorar cada etapa do pipeline, visualizar os resultados e analisar a robustez por equipamento.

---

## Análise de Robustez

Um dos principais focos do projeto é a investigação do *domain shift* causado pelos diferentes equipamentos de aquisição radiográfica. Para isso, o projeto oferece:

1. **Acurácia estratificada por equipamento:** Identifica quais máquinas apresentam maior ou menor desempenho.
2. **Desvio padrão entre equipamentos:** Quantifica a consistência do modelo entre diferentes fontes.
3. **Comparação LOEO vs. Split Padrão:** Mede o gap de desempenho quando o modelo enfrenta equipamentos nunca vistos durante o treino.

Essa análise é essencial para determinar se o modelo está aprendendo características morfológicas relevantes das vértebras ou apenas artefatos específicos de cada equipamento (efeito *Clever Hans*).

---

## Métricas de Avaliação

Devido ao desbalanceamento natural entre os estágios CVM, a acurácia isolada não é suficiente. As métricas calculadas incluem:

- **Acurácia (*Accuracy*)** global e por classe
- **F1-Score** (Macro e Weighted)
- **Cohen Kappa** (quadrático) — mede a concordância considerando a proximidade dos erros
- **AUC-ROC Macro OvR** — área sob a curva ROC
- **Matriz de Confusão** — análise da distribuição dos erros entre estágios adjacentes
- **Classification Report** completo com precisão, recall e F1 por classe

---

## Trabalhos Relacionados

Este projeto se insere em um contexto mais amplo de pesquisa em automação da avaliação CVM. Trabalhos como os de Makaremi et al. (2019), Zhou et al. (2021), Kavousinejad et al. (2024) e Khalid et al. (2025) fornecem a base sobre a qual esta abordagem se desenvolve, diferenciando-se pela combinação de classificação direta (sem marcação manual de *landmarks*) com validação rigorosa de generalização entre equipamentos. Mais detalhes podem ser encontrados no artigo associado.

---

## Licença

Este projeto utiliza o Aariz Cephalometric Dataset, que possui sua própria licença de uso. Consulte a página oficial do dataset para mais informações.

---

## Referências

- Baccetti, T., Franchi, L., & McNamara, J. A. (2002). *An improved version of the cervical vertebral maturation (CVM) method for the assessment of mandibular growth.* Angle Orthodontist.
- Baccetti, T., Franchi, L., & McNamara, J. A. (2005). *The cervical vertebral maturation (CVM) method for the assessment of optimal treatment timing in dentofacial orthopedics.* Seminars in Orthodontics.
- Khalid, M. et al. (2025). *Aariz: A Benchmark Dataset for Cephalometric Analysis and CVM Classification.* Figshare.
- Makaremi, M. et al. (2019). *Cervical vertebral maturation classification using convolutional neural networks.* International Orthodontics.
- Rana, S. S. et al. (2023). *Machine learning for cervical vertebral maturation assessment: A systematic review.* Orthodontics & Craniofacial Research.
- Zhou, J. et al. (2021). *Automatic cervical vertebral maturation assessment via deep learning.* Orthodontics & Craniofacial Research.
