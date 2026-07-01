#!/usr/bin/env python3
"""
Pipeline Otimizado para Classificação CVM com YOLOv8 + ResNet-50
================================================================
Melhorias implementadas para aumentar a acurácia:
1. Focal Loss (handles class imbalance melhor que CrossEntropy)
2. WeightedRandomSampler (oversampling classes minoritárias)
3. Data augmentation avançada (RandAugment, mais rotação, elastic)
4. Mixup augmentation durante treino
5. CosineAnnealingWarmRestarts (melhor scheduling de LR)
6. Label Smoothing na loss
7. Classifier head mais robusto (mais dropout, camada oculta)
8. Fine-tuning completo (sem freeze)
9. Test-Time Augmentation (TTA) na avaliação
10. Early stopping com patience baseado em val_loss
"""

import os, json, random, shutil, copy, argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import Counter
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, WeightedRandomSampler
from torchvision import transforms, models

from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import StratifiedKFold

from ultralytics import YOLO

# =============================================================================
# LOGGING
# =============================================================================
import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger('cvm_improved')

# =============================================================================
# CONFIGURACAO
# =============================================================================
SEED = 42

CLASS_NAMES = ['CVM-S1', 'CVM-S2', 'CVM-S3', 'CVM-S4', 'CVM-S5', 'CVM-S6']
NUM_CLASSES = len(CLASS_NAMES)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def set_seed(seed=42):
    """Define sementes para reproducibilidade."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# FOCAL LOSS
# =============================================================================
class FocalLoss(nn.Module):
    """
    Focal Loss para lidar com desbalanceamento extremo de classes.
    gamma > 0 reduz a contribuicao relativa de exemplos bem classificados,
    forcando o modelo a focar nas classes difficais/minoritarias.
    """
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', label_smoothing=0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha  # class weights
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


# =============================================================================
# DATASET COM CLAHE E SUPORTE A MIXUP
# =============================================================================
def apply_clahe(img_np: np.ndarray, clip_limit=3.0, tile_size=(8, 8)) -> np.ndarray:
    """CLAHE no canal L do LAB. Melhora contraste local de radiografias."""
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


class CVMDataset(Dataset):
    """Dataset para classificacao CVM com suporte a crops do YOLO."""

    def __init__(self, df: pd.DataFrame, original_root: Path,
                 cropped_root: Path = None, transform=None,                 is_train=False):
        self.df = df.reset_index(drop=True)
        self.original_root = original_root
        self.cropped_root = cropped_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, row) -> str:
        orig_path = Path(row['image_path'])
        if self.cropped_root and self.cropped_root.exists():
            try:
                rel = orig_path.relative_to(self.original_root)
                crop_path = self.cropped_root / rel
                if crop_path.exists():
                    return str(crop_path)
            except ValueError:
                pass
        return str(orig_path)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = int(row['label_idx'])

        img_bgr = cv2.imread(self._resolve_path(row))
        if img_bgr is None:
            img_rgb = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            # CLAHE mais forte para radiografias
            img_rgb = apply_clahe(img_rgb, clip_limit=3.0, tile_size=(8, 8))

        from PIL import Image
        img_pil = Image.fromarray(img_rgb)
        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil, label


# =============================================================================
# DATA AUGMENTATION AVANCADA
# =============================================================================
def get_transforms(img_size=224, interpolation=transforms.InterpolationMode.BICUBIC):
    """
    Data augmentation mais forte para radiografias:
    - Rotation mais agressiva (até ±15°)
    - Affine shifts (scale, translate)
    - Interpolacao BICUBIC (recommended para ViT)
    """
    transform_train = transforms.Compose([
        # Primeiro redimensiona um pouco maior para random crop
        transforms.Resize((int(img_size * 1.15), int(img_size * 1.15)),
                          interpolation=interpolation),
        transforms.RandomCrop(img_size),
        # Augmentacoes fortes para radiografia
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.1),  # radiografias podem estar invertidas
        transforms.RandomRotation(degrees=15, fill=0),
        transforms.ColorJitter(
            brightness=0.3, contrast=0.3,
            saturation=0.1, hue=0.05
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1),
            fill=0
        ),
        # To tensor + normalize
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        # Random erase (cutout) - ajuda com overfitting
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ])

    transform_eval = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=interpolation),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])

    return transform_train, transform_eval


# =============================================================================
# TEST-TIME AUGMENTATION (TTA)
# =============================================================================
def tta_predict(model, img_tensor, device, n_augments=5):
    """
    Test-Time Augmentation: faz múltiplas predições com variações
    e calcula a média das probabilidades.
    """
    model.eval()
    all_probs = []

    # Predicao original
    with torch.no_grad():
        outputs = model(img_tensor.unsqueeze(0).to(device))
        probs = F.softmax(outputs, dim=1)
        all_probs.append(probs)

    # Augmentacoes na inferencia
    tta_transforms = [
        lambda x: x,
        lambda x: torch.flip(x, dims=[2]),   # H flip
        lambda x: torch.flip(x, dims=[3]),   # W flip
        lambda x: torch.flip(torch.flip(x, dims=[2]), dims=[3]),  # both
    ]

    for tta_fn in tta_transforms:
        with torch.no_grad():
            aug_img = tta_fn(img_tensor)
            outputs = model(aug_img.unsqueeze(0).to(device))
            probs = F.softmax(outputs, dim=1)
            # Desfaz o flip nas probabilidades se necessario
            all_probs.append(probs)

    # Media de todas as predictoes
    mean_probs = torch.mean(torch.cat(all_probs, dim=0), dim=0, keepdim=True)
    return mean_probs


# =============================================================================
# MIXUP AUGMENTATION
# =============================================================================
def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation: combina pares de imagens linearmente."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """Loss para mixup."""
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# =============================================================================
# CARGA DE DADOS
# =============================================================================
def load_split(split: str, original_root: Path, class_names: list) -> pd.DataFrame:
    """Carrega imagens e labels CVM de um split."""
    img_dir = original_root / split / 'Cephalograms'
    ann_dir = original_root / split / 'Annotations' / 'CVM Stages'

    records = []
    for json_path in sorted(ann_dir.glob('*.json')):
        with open(json_path) as f:
            data = json.load(f)

        cvm_obj = data.get('cvm_stage', {})
        stage = cvm_obj.get('title', '').strip()
        if not stage:
            continue

        img_path = None
        for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
            candidate = img_dir / (json_path.stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path and stage in class_names:
            records.append({
                'image_path': str(img_path),
                'ceph_id':    data.get('ceph_id', json_path.stem),
                'label':      stage,
                'label_idx':  class_names.index(stage)
            })

    df = pd.DataFrame(records)
    log.info(f'[{split}] {len(df)} imagens carregadas.')
    return df


def load_machine_mappings(original_root: Path) -> dict:
    """Carrega mapeamento ceph_id -> equipamento do CSV."""
    csv_path = original_root / 'cephalogram_machine_mappings.csv'
    if not csv_path.exists():
        log.warning(f'Arquivo de mapeamento nao encontrado: {csv_path}')
        return {}
    machines_df = pd.read_csv(csv_path)
    ceph_to_machine = dict(zip(machines_df['cephalogram_id'], machines_df['machine']))
    log.info(f'Mapeamento de equipamentos carregado: {len(ceph_to_machine)} registros')
    return ceph_to_machine


def compute_class_weights(df_train: pd.DataFrame, class_names: list, device: torch.device):
    """Calcula pesos para CrossEntropyLoss baseados na frequencia inversa."""
    counts = df_train['label'].value_counts().reindex(class_names, fill_value=1)
    weights = torch.tensor(
        [1.0 / max(counts[c], 1) for c in class_names], dtype=torch.float
    ).to(device)
    weights = weights / weights.sum() * len(class_names)
    return weights


# =============================================================================
# MODELO MELHORADO
# =============================================================================
def build_improved_resnet50(num_classes: int) -> nn.Module:
    """
    ResNet-50 com melhorias no classifier head:
    - Mais Dropout (0.5 em vez de 0.3)
    - Camada oculta adicional (1024)
    - Fine-tuning completo (sem freeze)
    """
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    # Full fine-tuning: descongela tudo
    for param in model.parameters():
        param.requires_grad = True

    # Classifier head melhorado
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(in_features),
        nn.Dropout(0.5),
        nn.Linear(in_features, 1024),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model


# =============================================================================
# MODEL REGISTRY & VISION TRANSFORMER
# =============================================================================

MODEL_REGISTRY = {
    'resnet50': {
        'builder': 'build_improved_resnet50',
        'default_img_size': 224,
        'default_lr': 3e-4,
        'weight_suffix': 'resnet50',
        'description': 'ResNet-50 with improved classifier head',
    },
    'vit_b_16': {
        'builder': 'build_vit_b_16',
        'default_img_size': 224,
        'default_lr': 2e-4,
        'weight_suffix': 'vit_b_16',
        'description': 'ViT-Base/16 (patch_size=16, 86M params)',
    },
    'vit_b_32': {
        'builder': 'build_vit_b_32',
        'default_img_size': 224,
        'default_lr': 2e-4,
        'weight_suffix': 'vit_b_32',
        'description': 'ViT-Base/32 (patch_size=32, faster, fewer params)',
    },
    'vit_l_16': {
        'builder': 'build_vit_l_16',
        'default_img_size': 224,
        'default_lr': 1e-4,
        'weight_suffix': 'vit_l_16',
        'description': 'ViT-Large/16 (patch_size=16, 307M params)',
    },
}


def build_model(model_name: str, num_classes: int) -> nn.Module:
    """Factory: constroi o modelo pelo nome da arquitetura."""
    builders = {
        'resnet50': build_improved_resnet50,
        'vit_b_16': build_vit_b_16,
        'vit_b_32': build_vit_b_32,
        'vit_l_16': build_vit_l_16,
    }
    if model_name not in builders:
        raise ValueError(f'Modelo desconhecido: {model_name}. '
                         f'Opcoes: {list(MODEL_REGISTRY.keys())}')
    log.info(f'Construindo modelo: {model_name} '
             f'({MODEL_REGISTRY[model_name]["description"]})')
    return builders[model_name](num_classes)


def _build_vit(vit_fn, num_classes: int):
    """Helper interno para construir e adaptar um ViT para CVM."""
    model = vit_fn(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = True
    # O classification head do ViT e um Sequential(LayerNorm, Linear)
    in_features = model.heads[-1].in_features
    model.heads = nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Dropout(0.5),
        nn.Linear(in_features, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes),
    )
    return model


def build_vit_b_16(num_classes: int) -> nn.Module:
    """ViT-Base/16: patch_size=16, hidden_dim=768. Melhor custo-beneficio."""
    return _build_vit(models.vit_b_16, num_classes)


def build_vit_b_32(num_classes: int) -> nn.Module:
    """ViT-Base/32: patch_size=32, mais rapido mas menos preciso."""
    return _build_vit(models.vit_b_32, num_classes)


def build_vit_l_16(num_classes: int) -> nn.Module:
    """ViT-Large/16: patch_size=16, hidden_dim=1024. Maior acuracia potencial."""
    model = models.vit_l_16(weights='DEFAULT')
    for param in model.parameters():
        param.requires_grad = True
    in_features = model.heads[-1].in_features
    model.heads = nn.Sequential(
        nn.LayerNorm(in_features),
        nn.Dropout(0.4),
        nn.Linear(in_features, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(1024, num_classes),
    )
    return model


# =============================================================================
# YOLO
# =============================================================================
def apply_padding(box, img_shape, pad_ratio=0.05):
    """Aplica padding percentual a bounding box."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    pad_x, pad_y = int(w * pad_ratio), int(h * pad_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(img_shape[1], x2 + pad_x)
    y2 = min(img_shape[0], y2 + pad_y)
    return int(x1), int(y1), int(x2), int(y2)


def train_yolo(roboflow_path: Path, weights_out: Path, device: torch.device):
    """Treina YOLOv8n few-shot com dataset do Roboflow."""
    data_yaml = roboflow_path / 'data.yaml'
    if not data_yaml.exists():
        log.error(f'Dataset Roboflow nao encontrado em {data_yaml}')
        return False

    log.info('Iniciando treinamento YOLOv8n few-shot...')
    model = YOLO('yolov8n.pt')
    model.train(
        data=str(data_yaml),
        epochs=200,
        patience=30,
        imgsz=640,
        batch=16,
        lr0=0.01,
        augment=True,
        degrees=5,
        translate=0.1,
        scale=0.1,
        fliplr=0.5,
        mosaic=0.5,
        device='cpu' if device.type == 'cpu' else 0,
        project='yolo_cvm',
        name='fewshot_yolov8n',
        exist_ok=True,
        verbose=True
    )

    best_src = Path('yolo_cvm/fewshot_yolov8n/weights/best.pt')
    if best_src.exists():
        shutil.copy(str(best_src), str(weights_out))
        log.info(f'Modelo YOLO salvo em: {weights_out}')
        return True
    return False


def crop_all_splits(yolo_weights: Path, original_root: Path,
                    cropped_root: Path):
    """Aplica YOLO em todas as imagens e salva crops."""
    if not yolo_weights.exists():
        log.error(f'Modelo YOLO nao encontrado: {yolo_weights}')
        return False

    model = YOLO(str(yolo_weights))
    splits = ['train', 'valid', 'test']
    stats = {'total': 0, 'detected': 0, 'fallback': 0}

    for split in splits:
        img_dir = original_root / split / 'Cephalograms'
        save_dir = cropped_root / split / 'Cephalograms'
        os.makedirs(save_dir, exist_ok=True)

        # Copia anotacoes
        ann_src = original_root / split / 'Annotations'
        ann_dst = cropped_root / split / 'Annotations'
        if not ann_dst.exists():
            shutil.copytree(ann_src, ann_dst, dirs_exist_ok=True)

        log.info(f'Processando split: {split}')
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.bmp']:
                continue

            stats['total'] += 1
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            results = model(img, verbose=False, conf=0.25)
            if len(results[0].boxes) > 0:
                stats['detected'] += 1
                box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = apply_padding(box, img.shape, 0.05)
                cropped = img[y1:y2, x1:x2]
            else:
                stats['fallback'] += 1
                cropped = img

            cv2.imwrite(str(save_dir / img_path.name), cropped)

    log.info(f'Crop concluido: {stats["detected"]}/{stats["total"]} detectadas '
             f'({100*stats["detected"]/max(stats["total"],1):.1f}%), '
             f'{stats["fallback"]} fallbacks')
    return True


# =============================================================================
# TREINAMENTO OTIMIZADO
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device, use_mixup=True):
    """Loop de treino para uma epoca com suporte opcional a Mixup."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        # Mixup augmentation
        if use_mixup:
            mixed_inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, alpha=0.2)
            outputs = model(mixed_inputs)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
        else:
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        loss.backward()
        # Gradient clipping para estabilidade
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)

        if use_mixup:
            # Para mixup, contamos acerto se predizer qualquer uma das classes
            correct += ((predicted == labels_a).sum().item() +
                       (predicted == labels_b).sum().item()) / 2.0
        else:
            correct += (predicted == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device, use_tta=True):
    """Avaliacao com suporte opcional a TTA."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc='Eval'):
            inputs, labels = inputs.to(device), labels.to(device)

            if use_tta:
                # TTA para cada imagem no batch
                batch_preds = []
                for i in range(inputs.size(0)):
                    probs = tta_predict(model, inputs[i], device)
                    _, pred = torch.max(probs, 1)
                    batch_preds.append(pred.item())
                all_preds.extend(batch_preds)
                # Loss sem TTA
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())

            running_loss += loss.item() * inputs.size(0)
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def train_classifier(args):
    """Treina o classificador otimizado."""
    log.info('=' * 60)
    log.info('TREINAMENTO OTIMIZADO DO CLASSIFICADOR CVM')
    log.info('=' * 60)

    device = torch.device(args.device)
    set_seed(args.seed)

    # Caminhos
    original_root = Path(args.original_root)
    cropped_root = Path(args.cropped_root) if args.cropped_root else None
    weights_path = args.model_weights or f'best_{args.model}_cvm.pth'
    weights_out = Path(weights_path)

    # Carrega dados
    df_train = load_split('train', original_root, CLASS_NAMES)
    df_valid = load_split('valid', original_root, CLASS_NAMES)
    df_test = load_split('test', original_root, CLASS_NAMES)

    # Pesos para Focal Loss
    class_weights = compute_class_weights(df_train, CLASS_NAMES, device)

    # Transforms avançados
    transform_train, transform_eval = get_transforms(args.img_size)

    # Datasets
    train_ds = CVMDataset(df_train, original_root, cropped_root,
                          transform_train, is_train=True)
    valid_ds = CVMDataset(df_valid, original_root, cropped_root,
                          transform_eval, is_train=False)
    test_ds = CVMDataset(df_test, original_root, cropped_root,
                         transform_eval, is_train=False)

    # WeightedRandomSampler para oversampling de classes minoritarias
    train_labels = df_train['label_idx'].values
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights_sampler = 1.0 / class_counts
    sample_weights = class_weights_sampler[train_labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=sampler, num_workers=0
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size,
        shuffle=False, num_workers=0
    )

    log.info(f'Batches: Treino={len(train_loader)}, Val={len(valid_loader)}, '
             f'Teste={len(test_loader)}')

    # Modelo (ResNet, ViT, etc.)
    model = build_model(args.model, NUM_CLASSES).to(device)

    # Focal Loss com label smoothing
    criterion = FocalLoss(
        gamma=2.0,
        alpha=class_weights,
        label_smoothing=0.1
    )

    # Otimizador com weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-4
    )

    # Cosine Annealing com Warm Restarts
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,       # Primeiro ciclo: 10 epocas
        T_mult=2,     # Proximo ciclo: dobra o tamanho
        eta_min=1e-6  # LR minimo
    )

    log.info(f'Modelo: {sum(p.numel() for p in model.parameters()):,} parametros')
    log.info(f'Treinaveis: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    # Treinamento com early stopping
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    patience = args.patience
    history = {'train_loss': [], 'train_acc': [],
               'val_loss': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        log.info(f'\nEpoca {epoch}/{args.epochs}')

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            use_mixup=(epoch <= args.epochs * 0.8)  # Mixup nas primeiras 80% epocas
        )
        val_loss, val_acc, _, _ = evaluate(
            model, valid_loader, criterion, device, use_tta=False
        )
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        current_lr = optimizer.param_groups[0]['lr']
        log.info(f'Train Loss: {train_loss:.4f} | Train Acc: {100*train_acc:.2f}%')
        log.info(f'Val Loss: {val_loss:.4f} | Val Acc: {100*val_acc:.2f}% | LR: {current_lr:.2e}')

        # Salva melhor modelo por val_acc
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), str(weights_out))
            log.info(f'  >> Novo melhor modelo (val_acc: {100*val_acc:.2f}%)')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f'  Early stopping na epoca {epoch}')
                break

    log.info(f'Treinamento concluido! Melhor val_acc: {100*best_val_acc:.2f}%')

    # Plota curvas
    _plot_training_curves(history, best_val_acc, args.output_dir)

    # Carrega melhor modelo
    model.load_state_dict(torch.load(str(weights_out), map_location=device))

    # Avaliacao final com TTA
    log.info('Avaliando no test set com TTA...')
    test_loss, test_acc, preds, labels = evaluate(
        model, test_loader, criterion, device, use_tta=True
    )
    log.info(f'Test Acc (com TTA): {100*test_acc:.2f}% | Test Loss: {test_loss:.4f}')
    log.info(f'\n{classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)}')

    # Matriz de confusao
    _plot_confusion_matrix(labels, preds, args.output_dir)

    # Resultados
    kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    log.info(f'Cohen Kappa: {kappa:.4f}')

    results_df = pd.DataFrame({
        'split': ['test'],
        'accuracy': [test_acc],
        'loss': [test_loss],
        'best_val_acc': [best_val_acc],
        'kappa': [kappa]
    })
    results_df.to_csv(str(Path(args.output_dir) / 'resultados_teste.csv'), index=False)

    return model, test_loader, test_acc, preds, labels, df_test


# =============================================================================
# AVALIACAO POR EQUIPAMENTO
# =============================================================================
def evaluate_by_machine(model, test_loader, criterion, device,
                        df_test, original_root, cropped_root,
                        ceph_to_machine, output_dir):
    """Avalia acuracia por equipamento e gera grafico."""
    log.info('\n' + '=' * 60)
    log.info('AVALIACAO POR EQUIPAMENTO')
    log.info('=' * 60)

    df_test = df_test.copy()
    df_test['machine'] = df_test['ceph_id'].map(ceph_to_machine)
    df_test_eval = df_test.dropna(subset=['machine']).copy()

    _, _, all_preds, all_labels = evaluate(
        model, test_loader, criterion, device, use_tta=True
    )

    df_test_eval['pred_idx'] = [all_preds[i] for i in df_test_eval.index]
    df_test_eval['correct'] = df_test_eval['label_idx'] == df_test_eval['pred_idx']

    acc_by_machine = df_test_eval.groupby('machine').agg(
        accuracy=('correct', 'mean'),
        count=('correct', 'count')
    ).reset_index()
    acc_by_machine['accuracy'] *= 100

    log.info('Acuracia por Equipamento:')
    for _, row in acc_by_machine.iterrows():
        log.info(f'  {row["machine"]:30s}: {row["accuracy"]:5.1f}% '
                 f'({int(row["count"])} amostras)')

    mean_acc = acc_by_machine['accuracy'].mean()
    std_acc = acc_by_machine['accuracy'].std()
    log.info(f'\nMedia: {mean_acc:.2f}% | Desvio: {std_acc:.2f}%')

    _plot_accuracy_by_machine(
        acc_by_machine, mean_acc,
        100 * (all_labels == all_preds).mean(), output_dir
    )

    return acc_by_machine


# =============================================================================
# K-FOLD CROSS-VALIDATION
# =============================================================================
def run_kfold_cv(args):
    """
    Executa validação cruzada K-Fold com estratificação.
    Garante que cada fold mantenha a proporção de classes.
    """
    log.info('=' * 70)
    log.info(f'VALIDACAO CRUZADA K-FOLD ({args.k_folds} FOLDS, ESTRATIFICADA)')
    log.info('=' * 70)

    device = torch.device(args.device)
    set_seed(args.seed)

    original_root = Path(args.original_root)
    cropped_root = Path(args.cropped_root) if args.cropped_root else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Carrega todos os splits e combina em um DataFrame único
    df_train = load_split('train', original_root, CLASS_NAMES)
    df_valid = load_split('valid', original_root, CLASS_NAMES)
    df_test = load_split('test', original_root, CLASS_NAMES)
    df_all = pd.concat([df_train, df_valid, df_test], ignore_index=True)
    log.info(f'Total de imagens para K-Fold: {len(df_all)}')

    # Distribuição de classes
    log.info(f'\nDistribuicao de classes no dataset completo:')
    class_dist = df_all['label'].value_counts().reindex(CLASS_NAMES, fill_value=0)
    for cls in CLASS_NAMES:
        log.info(f'  {cls}: {int(class_dist[cls])}')

    # Prepara estratificação
    X = df_all['image_path'].values
    y = df_all['label_idx'].values

    skf = StratifiedKFold(
        n_splits=args.k_folds,
        shuffle=True,
        random_state=args.seed
    )

    transform_train, transform_eval = get_transforms(args.img_size)

    fold_results = []
    fold_histories = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
        log.info(f'\n{"="*60}')
        log.info(f'FOLD {fold}/{args.k_folds}')
        log.info(f'{"="*60}')

        # Divide dados do fold
        df_train_fold = df_all.iloc[train_idx].reset_index(drop=True)
        df_val_fold = df_all.iloc[val_idx].reset_index(drop=True)

        log.info(f'  Treino: {len(df_train_fold)} imagens')
        log.info(f'  Validacao: {len(df_val_fold)} imagens')

        # Distribuição no treino do fold
        fold_class_dist = df_train_fold['label'].value_counts().reindex(CLASS_NAMES, fill_value=0)
        log.info(f'  Distribuicao classes treino:')
        for cls in CLASS_NAMES:
            log.info(f'    {cls}: {int(fold_class_dist[cls])}')

        # Pesos para Focal Loss
        fold_weights = compute_class_weights(df_train_fold, CLASS_NAMES, device)

        # Datasets
        train_ds_fold = CVMDataset(df_train_fold, original_root, cropped_root,
                                    transform_train, is_train=True)
        val_ds_fold = CVMDataset(df_val_fold, original_root, cropped_root,
                                  transform_eval, is_train=False)

        # WeightedRandomSampler para oversampling
        fold_labels = df_train_fold['label_idx'].values
        fold_class_counts = np.bincount(fold_labels, minlength=NUM_CLASSES)
        fold_class_weights_sampler = 1.0 / np.maximum(fold_class_counts, 1)
        fold_sample_weights = fold_class_weights_sampler[fold_labels]
        fold_sampler = WeightedRandomSampler(
            weights=fold_sample_weights,
            num_samples=len(fold_sample_weights),
            replacement=True
        )

        train_loader_fold = DataLoader(
            train_ds_fold, batch_size=args.batch_size,
            sampler=fold_sampler, num_workers=0
        )
        val_loader_fold = DataLoader(
            val_ds_fold, batch_size=args.batch_size,
            shuffle=False, num_workers=0
        )

        # Modelo (ResNet, ViT, etc.)
        model_fold = build_model(args.model, NUM_CLASSES).to(device)

        # Focal Loss
        criterion_fold = FocalLoss(
            gamma=2.0,
            alpha=fold_weights,
            label_smoothing=0.1
        )

        # Otimizador
        optimizer_fold = optim.AdamW(
            model_fold.parameters(),
            lr=args.kfold_lr,
            weight_decay=1e-4
        )

        # Scheduler Cosine Annealing
        kfold_epochs = args.kfold_epochs
        scheduler_fold = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer_fold,
            T_0=10,
            T_mult=2,
            eta_min=1e-6
        )

        # Treinamento
        best_val_acc = 0.0
        best_state_dict = None
        patience_counter = 0
        patience_fold = max(10, kfold_epochs // 5)
        fold_history = {'train_loss': [], 'train_acc': [],
                        'val_loss': [], 'val_acc': []}

        for epoch in range(1, kfold_epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model_fold, train_loader_fold, criterion_fold,
                optimizer_fold, device,
                use_mixup=(epoch <= kfold_epochs * 0.8)
            )
            val_loss, val_acc, _, _ = evaluate(
                model_fold, val_loader_fold, criterion_fold,
                device, use_tta=False
            )
            scheduler_fold.step()

            fold_history['train_loss'].append(train_loss)
            fold_history['train_acc'].append(train_acc)
            fold_history['val_loss'].append(val_loss)
            fold_history['val_acc'].append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_state_dict = {k: v.clone() for k, v in model_fold.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience_fold:
                    log.info(f'  Early stopping fold {fold} na epoca {epoch}')
                    break

        log.info(f'  Fold {fold} - Melhor Val Acc: {100*best_val_acc:.2f}%')

        # Carrega melhor modelo do fold
        model_fold.load_state_dict(best_state_dict)

        # Avaliação final com TTA
        _, val_acc_tta, preds, labels = evaluate(
            model_fold, val_loader_fold, criterion_fold,
            device, use_tta=True
        )

        # Cohen Kappa
        kappa_fold = cohen_kappa_score(labels, preds, weights='quadratic')

        log.info(f'  Fold {fold} - Val Acc (com TTA): {100*val_acc_tta:.2f}% | '
                 f'Kappa: {kappa_fold:.4f}')
        log.info(f'\n  Classification Report - Fold {fold}:')
        report = classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)
        for line in report.split('\n'):
            log.info(f'  {line}')

        fold_results.append({
            'fold': fold,
            'n_train': len(df_train_fold),
            'n_val': len(df_val_fold),
            'accuracy': val_acc_tta,
            'kappa': kappa_fold,
            'preds': preds,
            'labels': labels
        })
        fold_histories.append(fold_history)

        # Salva modelo do fold
        fold_path = output_dir / f'model_fold_{fold}.pth'
        torch.save(best_state_dict, str(fold_path))
        log.info(f'  Modelo do fold salvo em: {fold_path}')

    # ── Resultados Consolidados ──────────────────────────────────────────
    _plot_kfold_results(fold_results, output_dir)

    kfold_df = pd.DataFrame([{
        'Fold': r['fold'],
        'Train Size': r['n_train'],
        'Val Size': r['n_val'],
        'Accuracy (%)': round(100 * r['accuracy'], 2),
        'Cohen Kappa': round(r['kappa'], 4)
    } for r in fold_results])

    kfold_df.to_csv(str(output_dir / 'kfold_results.csv'), index=False)

    log.info('\n' + '=' * 70)
    log.info('RESULTADOS K-FOLD CROSS-VALIDATION')
    log.info('=' * 70)
    log.info(f'\n{kfold_df.to_string(index=False)}')

    # Estatísticas agregadas
    mean_acc = kfold_df['Accuracy (%)'].mean()
    std_acc = kfold_df['Accuracy (%)'].std()
    mean_kappa = kfold_df['Cohen Kappa'].mean()
    std_kappa = kfold_df['Cohen Kappa'].std()

    log.info(f'\n{"="*70}')
    log.info(f'SUMARIO K-FOLD ({args.k_folds} FOLDS)')
    log.info(f'{"="*70}')
    log.info(f'  Acc Media: {mean_acc:.2f}% +- {std_acc:.2f}%')
    log.info(f'  Acc Min:   {kfold_df["Accuracy (%)"].min():.2f}%')
    log.info(f'  Acc Max:   {kfold_df["Accuracy (%)"].max():.2f}%')
    log.info(f'  Kappa Medio: {mean_kappa:.4f} +- {std_kappa:.4f}')
    log.info(f'  Kappa Min:   {kfold_df["Cohen Kappa"].min():.4f}')
    log.info(f'  Kappa Max:   {kfold_df["Cohen Kappa"].max():.4f}')

    # Interpretação
    log.info(f'\n📈 INTERPRETACAO:')
    if std_acc < 3:
        log.info(f'  ✅ Modelo MUITO ESTAVEL (desvio < 3%)')
    elif std_acc < 5:
        log.info(f'  ✅ Modelo ESTAVEL (desvio < 5%)')
    elif std_acc < 8:
        log.info(f'  ⚠️  Variacao moderada (desvio < 8%)')
    else:
        log.info(f'  ❌ Modelo INSTAVEL (desvio > 8%)')

    if mean_acc > 70:
        log.info(f'  ✅ Acc media > 70%: ALTA')
    elif mean_acc > 50:
        log.info(f'  ⚠️  Acc media > 50%: MODERADA')
    else:
        log.info(f'  ❌ Acc media < 50%: BAIXA')

    return kfold_df, fold_histories, fold_results


# =============================================================================
# PLOTS
# =============================================================================
def _plot_training_curves(history, best_val_acc, output_dir):
    """Salva grafico de curvas de treinamento."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoca')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Curvas de Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].axhline(y=best_val_acc, color='green', linestyle='--', alpha=0.5,
                     label=f'Best: {100*best_val_acc:.2f}%')
    axes[1].set_xlabel('Epoca')
    axes[1].set_ylabel('Acuracia')
    axes[1].set_title('Curvas de Acuracia')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = Path(output_dir) / 'curvas_treinamento.png'
    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'Curvas salvas: {path}')


def _plot_confusion_matrix(labels, preds, output_dir):
    """Salva matriz de confusao."""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel('Predito')
    plt.ylabel('Verdadeiro')
    plt.title('Matriz de Confusao - Test Set')
    plt.tight_layout()
    path = Path(output_dir) / 'matriz_confusao.png'
    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'Matriz salva: {path}')


def _plot_kfold_results(fold_results, output_dir):
    """Salva grafico de resultados K-Fold: boxplot e barras individuais."""
    accs = [100 * r['accuracy'] for r in fold_results]
    kappas = [r['kappa'] for r in fold_results]
    fold_labels = [f'Fold {r["fold"]}' for r in fold_results]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Gráfico 1: Acuracia por fold
    colors = [plt.cm.RdYlGn(acc / 100) for acc in accs]
    bars = axes[0].bar(fold_labels, accs, color=colors, edgecolor='gray', linewidth=1.2)
    axes[0].axhline(y=np.mean(accs), color='blue', linestyle='--', linewidth=2,
                     label=f'Media: {np.mean(accs):.1f}%')
    axes[0].fill_between(range(len(accs)),
                          np.mean(accs) - np.std(accs),
                          np.mean(accs) + np.std(accs),
                          alpha=0.15, color='blue',
                          label=f'+/- 1 Std: {np.std(accs):.1f}%')
    axes[0].set_xlabel('Fold')
    axes[0].set_ylabel('Acuracia de Validacao (%)')
    axes[0].set_title(f'K-Fold Cross-Validation: Acuracia por Fold\n'
                       f'Media: {np.mean(accs):.1f}% | Desvio: {np.std(accs):.1f}%')
    axes[0].set_ylim(max(0, np.mean(accs) - 3 * np.std(accs) - 5),
                      min(100, np.mean(accs) + 3 * np.std(accs) + 5))
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, axis='y')
    for bar, acc in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                     f'{acc:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Gráfico 2: Cohen Kappa por fold
    colors_k = [plt.cm.RdYlGn(k / 1.0) for k in kappas]
    bars_k = axes[1].bar(fold_labels, kappas, color=colors_k, edgecolor='gray', linewidth=1.2)
    axes[1].axhline(y=np.mean(kappas), color='blue', linestyle='--', linewidth=2,
                     label=f'Media: {np.mean(kappas):.3f}')
    axes[1].fill_between(range(len(kappas)),
                          np.mean(kappas) - np.std(kappas),
                          np.mean(kappas) + np.std(kappas),
                          alpha=0.15, color='blue',
                          label=f'+/- 1 Std: {np.std(kappas):.3f}')
    axes[1].set_xlabel('Fold')
    axes[1].set_ylabel('Cohen Kappa (Quadratico)')
    axes[1].set_title(f'K-Fold Cross-Validation: Cohen Kappa por Fold\n'
                       f'Media: {np.mean(kappas):.3f} | Desvio: {np.std(kappas):.3f}')
    axes[1].set_ylim(max(0, np.mean(kappas) - 3 * np.std(kappas) - 0.1),
                      min(1.0, np.mean(kappas) + 3 * np.std(kappas) + 0.1))
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, kap in zip(bars_k, kappas):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                     f'{kap:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = Path(output_dir) / 'kfold_results.png'
    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'Grafico K-Fold salvo: {path}')

    # Tambem salva um boxplot combinado
    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(accs, patch_artist=True, widths=0.4)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    # Pontos individuais
    for i, acc in enumerate(accs):
        jitter = np.random.uniform(-0.1, 0.1)
        ax.plot(1 + jitter, acc, 'ro', markersize=8, alpha=0.7)
        ax.text(1.15 + jitter, acc, f'{acc:.1f}%', fontsize=9, alpha=0.8)
    ax.set_xticks([1])
    ax.set_xticklabels([f'K-Fold (n={len(accs)})'])
    ax.set_ylabel('Acuracia de Validacao (%)')
    ax.set_title(f'Distribuicao da Acuracia nos {len(accs)} Folds\n'
                  f'Media: {np.mean(accs):.1f}% | Mediana: {np.median(accs):.1f}% | '
                  f'Std: {np.std(accs):.1f}%')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    boxplot_path = Path(output_dir) / 'kfold_boxplot.png'
    plt.savefig(str(boxplot_path), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'Boxplot K-Fold salvo: {boxplot_path}')


def _plot_accuracy_by_machine(acc_df, mean_acc, global_acc, output_dir):
    """Salva grafico de acuracia por equipamento."""
    plt.figure(figsize=(12, 6))
    acc_sorted = acc_df.sort_values('accuracy', ascending=True)
    colors = [plt.cm.RdYlGn(acc / 100) for acc in acc_sorted['accuracy']]
    plt.barh(acc_sorted['machine'], acc_sorted['accuracy'], color=colors)
    plt.xlabel('Acuracia (%)')
    plt.title('Acuracia por Equipamento de Raio-X')
    plt.xlim(0, 105)
    for i, (_, row) in enumerate(acc_sorted.iterrows()):
        plt.text(row['accuracy'] + 0.5, i,
                 f'{row["accuracy"]:.1f}% (n={int(row["count"])})',
                 va='center', fontsize=10)
    plt.axvline(x=mean_acc, color='blue', linestyle='--', alpha=0.7,
                label=f'Media: {mean_acc:.1f}%')
    plt.axvline(x=global_acc, color='red', linestyle=':', alpha=0.7,
                label=f'Global: {global_acc:.1f}%')
    plt.legend()
    plt.tight_layout()
    path = Path(output_dir) / 'acuracia_por_equipamento.png'
    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'Grafico salvo: {path}')


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    """Configura argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description='Pipeline CVM Otimizado: Classificacao CVM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Treinamento completo (recomendado)
  python train_cvm_improved.py --stage all --epochs 150 --device cuda

  # Apenas treinar classificador
  python train_cvm_improved.py --stage train_classifier --epochs 100

  # Avaliar modelo ja treinado
  python train_cvm_improved.py --stage evaluate
        """
    )

    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'train_yolo', 'crop', 'train_classifier',
                                 'evaluate', 'kfold', 'loeo'],
                        help='Estagio do pipeline')

    # Caminhos
    parser.add_argument('--original-root', type=str, default='Aariz_extracted/Aariz')
    parser.add_argument('--cropped-root', type=str, default='Aariz_cropped_CVM')
    parser.add_argument('--roboflow-dataset', type=str, default='roboflow_cvm_dataset')
    parser.add_argument('--yolo-weights', type=str, default='best_yolo_cvm.pt')
    parser.add_argument('--model', type=str, default='resnet50',
                        choices=list(MODEL_REGISTRY.keys()),
                        help='Arquitetura do classificador')
    parser.add_argument('--model-weights', type=str, default=None,
                        help='Caminho para salvar/carregar pesos. '
                             '(default: best_{model}_cvm.pth)')
    parser.add_argument('--output-dir', type=str, default='resultados_melhorados')

    # Hiperparametros otimizados
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--patience', type=int, default=20)

    # K-Fold
    parser.add_argument('--k-folds', type=int, default=5,
                        help='Numero de folds para K-Fold CV')
    parser.add_argument('--kfold-epochs', type=int, default=80,
                        help='Epocas por fold no K-Fold CV')
    parser.add_argument('--kfold-lr', type=float, default=2e-4,
                        help='Learning rate para K-Fold CV')

    # LOEO
    parser.add_argument('--loeo-epochs', type=int, default=30)
    parser.add_argument('--loeo-lr', type=float, default=1e-4)

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cropped_root:
        Path(args.cropped_root).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    set_seed(args.seed)

    log.info('=' * 60)
    log.info(f'Pipeline CVM Otimizado v2.0')
    log.info(f'Estagio: {args.stage}')
    log.info(f'Modelo: {args.model} ({MODEL_REGISTRY[args.model]["description"]})')
    log.info(f'Dispositivo: {device}')
    log.info(f'Epocas: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}')
    log.info('=' * 60)

    # Estagio: Treinar YOLO
    if args.stage in ('all', 'train_yolo'):
        train_yolo(Path(args.roboflow_dataset), Path(args.yolo_weights), device)

    # Estagio: Crop
    if args.stage in ('all', 'crop'):
        crop_all_splits(Path(args.yolo_weights), Path(args.original_root),
                        Path(args.cropped_root))

    # Estagio: Treinar Classificador
    if args.stage in ('all', 'train_classifier'):
        model, test_loader, test_acc, preds, labels, df_test = train_classifier(args)
    else:
        model = test_loader = test_acc = preds = labels = df_test = None

    # Estagio: Avaliar
    if args.stage in ('all', 'evaluate'):
        if model is None:
            eval_weights = args.model_weights or f'best_{args.model}_cvm.pth'
            weights_path = Path(eval_weights)
            if not weights_path.exists():
                log.error(f'Modelo nao encontrado: {weights_path}')
                return
            log.info(f'Carregando modelo: {weights_path}')
            model = build_model(args.model, NUM_CLASSES).to(device)
            model.load_state_dict(torch.load(weights_path, map_location=device))

            transform_train, transform_eval = get_transforms(args.img_size)
            df_test = load_split('test', Path(args.original_root), CLASS_NAMES)
            test_ds = CVMDataset(
                df_test, Path(args.original_root),
                Path(args.cropped_root) if args.cropped_root else None,
                transform_eval
            )
            test_loader = DataLoader(
                test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0
            )
            criterion_dummy = nn.CrossEntropyLoss()
        else:
            criterion_dummy = nn.CrossEntropyLoss()

        ceph_to_machine = load_machine_mappings(Path(args.original_root))
        evaluate_by_machine(
            model, test_loader, criterion_dummy, device,
            df_test, Path(args.original_root),
            Path(args.cropped_root) if args.cropped_root else None,
            ceph_to_machine, args.output_dir
        )

    # Estagio: K-Fold Cross-Validation
    if args.stage in ('all', 'kfold'):
        kfold_df, fold_histories, fold_results = run_kfold_cv(args)

    # Estagio: LOEO (usando o novo modelo)
    if args.stage in ('all', 'loeo'):
        log.info("LOEO nao implementado neste script. "
                 "Veja pipeline_cvm.py para LOEO.")

    log.info('Pipeline otimizado finalizado com sucesso!')


if __name__ == '__main__':
    main()
