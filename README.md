# Classificação da Maturação Cervical Vertebral (CVM) em Radiografias Cefalométricas Laterais

Classificação automatizada dos seis estágios de maturação das vértebras cervicais (CVM-S1 a CVM-S6) utilizando aprendizado profundo e transferência de conhecimento (*Transfer Learning*) sobre o conjunto de dados *Aariz Cephalometric Dataset*.

> Este repositório está associado a um artigo acadêmico, que investiga a aplicação de redes neurais convolucionais para auxiliar no diagnóstico ortodôntico, com ênfase na análise de robustez entre diferentes equipamentos de aquisição radiográfica.

---

## Contexto Clínico

A avaliação da Maturação das Vértebras Cervicais (*Cervical Vertebral Maturation* -- CVM) é um método amplamente utilizado em ortodontia para determinar a idade esquelética e o timing ideal para intervenções ortopédicas faciais. O método classifica o desenvolvimento das vértebras C2, C3 e C4 em seis estágios (CS1 a CS6) com base em alterações morfológicas observáveis em radiografias cefalométricas laterais.

A classificação manual desses estágios é subjetiva, demorada e dependente da experiência do examinador, além de ser sensível às variações de qualidade e contraste introduzidas por diferentes equipamentos de raio-X -- fenômeno conhecido como *domain shift*.

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
├── results/
│   ├── distribuicao_classes_random_split.png
│   ├── curvas_treinamento_random_split.png
│   ├── curvas_roc_random_split.png
│   ├── matriz_confusao_random_split.png
│   ├── acuracia_por_equipamento_random_split.png
│   └── resultados_random_split.csv
├── best_cvm_model.pth                      # Pesos do melhor modelo (maior val_acc)
└── cvm_random_split.ipynb      
└── Readme.txt
```

---

## Metodologia

---

#### Pipeline

O projeto implementa a classificação CVM por  **classificação direta** , sem etapa prévia de detecção/recorte de região de interesse.

##### Classificação Direta (`cvm_random_split.ipynb`)

* Arquitetura:  **ResNet-50** , pré-treinada na ImageNet, com **fine-tuning completo**
* Camada densa final substituída por `Dropout(0.3) + Linear(2048 → 6)`
* Otimizador **Adam** (lr=1e-4, weight_decay=1e-4)
* Scheduler **StepLR** (decaimento de 50% a cada 10 épocas)
* `CrossEntropyLoss` com pesos por classe ( *class_weight* ) para compensar o desbalanceamento entre estágios
* 30 épocas de treinamento

### Pré-processamento

Todas as imagens passam por um fluxo de pré-processamento para padronização entre os diferentes equipamentos:

1. **Redimensionamento** para 224×224 pixels
2. **CLAHE** ( *Contrast Limited Adaptive Histogram Equalization* ) no canal L do espaço de cor LAB, realçando estruturas ósseas e mitigando variações de iluminação entre equipamentos
3. **Normalização** com médias e desvios padrão do ImageNet
4. **Data augmentation** (apenas no treino): flip horizontal (p=0.3), rotação aleatória (±5°) e ColorJitter (brilho/contraste ±0.2)

#### Estratégia de Validação

Accuracy, F1-Score (macro e weighted), AUC-ROC (macro One-vs-Rest), Cohen's Kappa quadrático (apropriado para classes ordinais como o CVM) e Matriz de Confusão. Também é gerada uma quebra **informativa** de acurácia por equipamento de aquisição (usando `cephalogram_machine_mappings.csv`), para observar sensibilidade a variações de hardware dentro do próprio split oficial.

---

## Arquivos do Projeto

| Arquivo                    | Descrição                                       |
| -------------------------- | ------------------------------------------------- |
| `cvm_random_split.ipynb` | Notebook inicial de classificação com ResNet-50 |
| `best_cvm_model.pth`     | Pesos do melhor modelo (maior val_acc)            |
| `resultados_cvm.csv`     | Resultados da classificação por classe          |
| `resultados/`            | Diretório com gráficos e matrizes de confusão |

---

## Como Executar

### Criar uma venv

```Shell
python3 -m venv .venv
source .venv/bin/activate        # Linux/Mac
.venv\Scripts\activate           # Windows
```

### Dependências

```Shell
pip install torch torchvision scikit-learn seaborn matplotlib pandas opencv-python requests tqdm notebook
```

| Pacote                     | Função                                                |
| -------------------------- | ------------------------------------------------------- |
| `torch`,`torchvision`  | modelo ResNet-50, treinamento, dataloaders              |
| `scikit-learn`           | métricas (F1, AUC-ROC, Kappa, matriz de confusão)     |
| `opencv-python`          | CLAHE e leitura de imagens                              |
| `pandas`                 | manipulação de metadados/anotações e resultados     |
| `matplotlib`,`seaborn` | gráficos (curvas, matriz de confusão, distribuição) |
| `requests`               | download automático do dataset via API do Figshare     |
| `tqdm`                   | barras de progresso                                     |

### Rodar Jupyter Notebook

```Python
jupyter notebook cvm_random_split.ipynb
```

### Saídas geradas

Ao final da execução:

1. *best_cvm_model.pth -- pesos do modelo com melhor acurácia de validação*
2. *results/resultados_random_split.csv ==--== tabela-resumo das métricas de teste*
3. *results/*.png -- gráficos de distribuição de classes, curvas de treinamento,
4. matriz de confusão, curvas ROC e acurácia por equipamento

---

## Métricas de Avaliação

Devido ao desbalanceamento natural entre os estágios CVM, a acurácia isolada não é suficiente. As métricas calculadas incluem:

* **Accuracy** -- global e por classe
* **F1-Score** (macro e weighted)
* **AUC-ROC** (macro One-vs-Rest)
* **Cohen's Kappa** quadrático -- apropriado para classes ordinais como o CVM, pois pondera a proximidade entre os erros (confundir CVM-S3 com CVM-S4 pesa menos que confundir CVM-S1 com CVM-S6)
* **Matriz de Confusão** -- análise da distribuição dos erros entre estágios adjacentes
* **Classification Report** completo, com precisão, recall e F1 por classe

Também é gerada uma quebra **informativa** de acurácia por equipamento de aquisição (usando `cephalogram_machine_mappings.csv`), para observar sensibilidade a variações de hardware dentro do próprio split oficial.

---

### Resultados

| Métrica                    | Valor           |
| --------------------------- | --------------- |
| Accuracy                    | 0,5000 (50,00%) |
| F1-Score (macro)            | 0,3432          |
| F1-Score (weighted)         | 0,4918          |
| AUC-ROC (macro OvR)         | 0,7639          |
| Cohen's Kappa (quadrático) | 0,5566          |
| Nº de imagens no teste     | 150             |

### Gráficos gerados

| Arquivo                                       | Descrição                                                                                    |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| `distribuicao_classes_random_split.png`     | Distribuição dos 6 estágios CVM em treino/validação/teste                                 |
| `curvas_treinamento_random_split.png`       | Loss e acurácia por época (treino vs. validação)                                           |
| `matriz_confusao_random_split.png`          | Matriz de confusão (contagem e normalizada) no teste                                          |
| `curvas_roc_random_split.png`               | Curvas ROC One-vs-Rest por estágio CVM                                                        |
| `acuracia_por_equipamento_random_split.png` | Acurácia no teste, quebrada por equipamento de aquisição (análise informativa de robustez) |

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
