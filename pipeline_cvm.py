#!/usr/bin/env python3
"""
Pipeline Robusto para Classificação da Maturação Cervical Vertebral (CVM)
=====================================================================

Pipeline de dois estágios:
  Estagio 1: YOLOv8n (Few-Shot) para deteccao e extracao da ROI (C2-C4)
  Estagio 2: ResNet-50 com fine-tuning para classificacao CVM (6 estagios)

Uso:
  python pipeline_cvm.py --stage all                     # Executa pipeline completo
  python pipeline_cvm.py --stage crop                    # So extrai ROI com YOLO
  python pipeline_cvm.py --stage train_classifier        # So treina ResNet-50
  python pipeline_cvm.py --stage evaluate                # So avalia modelos
  python pipeline_cvm.py --stage loeo                    # Leave-One-Equipment-Out CV

  python pipeline_cvm.py --stage all --epochs 30 --batch-size 32  # Hiperparams customizados
  python pipeline_cvm.py --stage all --device cuda       # Forcar GPU

Autor: [Seu Nome]
Dataset: Aariz Cephalometric Dataset (~1000 imagens, 7 equipamentos)
"""

import os
import sys
import json
import random
import shutil
import copy
import logging
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import cv2
import matplotlib
matplotlib.use('Agg')  # Modo headless para servidores
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms, models

from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

from ultralytics import YOLO

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
log = logging.getLogger('pipeline_cvm')

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


# =============================================================================
# DATASET
# =============================================================================
class CVMDataset(Dataset):
    """Dataset para classificacao CVM com fallback para crops ou originais."""

    def __init__(self, df: pd.DataFrame, original_root: Path,
                 cropped_root: Path = None, transform=None):
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
            img_rgb = _apply_clahe(img_rgb)

        from PIL import Image
        img_pil = Image.fromarray(img_rgb)
        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil, label


def _apply_clahe(img_np: np.ndarray) -> np.ndarray:
    """CLAHE no canal L do LAB para melhorar contraste radiografico."""
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


# =============================================================================
# TRANSFORMS
# =============================================================================
def get_transforms(img_size=224):
    """Retorna transforms de treino e avaliacao."""
    transform_train = transforms.Compose([
        transforms.Resize((img_size + 20, img_size + 20)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    transform_eval = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    ])
    return transform_train, transform_eval


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
                'ceph_id': data.get('ceph_id', json_path.stem),
                'label': stage,
                'label_idx': class_names.index(stage)
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
# MODELO
# =============================================================================
def build_resnet50(num_classes: int, freeze_strategy='partial') -> nn.Module:
    """Constroi ResNet-50 pre-treinada."""
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

    if freeze_strategy == 'partial':
        for param in model.parameters():
            param.requires_grad = False
        for name in ['layer2', 'layer3', 'layer4']:
            for param in getattr(model, name).parameters():
                param.requires_grad = True
    else:  # full
        for param in model.parameters():
            param.requires_grad = True

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes)
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
# TREINAMENTO
# =============================================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    """Loop de treino para uma epoca."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, loader, criterion, device):
    """Avaliacao sem gradiente."""
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def train_classifier(args):
    """Treina o classificador ResNet-50 no split padrao."""
    log.info('=' * 60)
    log.info('INICIANDO TREINAMENTO DO CLASSIFICADOR')
    log.info('=' * 60)

    device = torch.device(args.device)
    set_seed(args.seed)

    # Caminhos
    original_root = Path(args.original_root)
    cropped_root = Path(args.cropped_root) if args.cropped_root else None
    weights_out = Path(args.resnet_weights)

    # Carrega dados
    df_train = load_split('train', original_root, CLASS_NAMES)
    df_valid = load_split('valid', original_root, CLASS_NAMES)
    df_test = load_split('test', original_root, CLASS_NAMES)

    # Pesos
    class_weights = compute_class_weights(df_train, CLASS_NAMES, device)

    # Transforms
    transform_train, transform_eval = get_transforms(args.img_size)

    # Datasets
    train_ds = CVMDataset(df_train, original_root, cropped_root, transform_train)
    valid_ds = CVMDataset(df_valid, original_root, cropped_root, transform_eval)
    test_ds = CVMDataset(df_test, original_root, cropped_root, transform_eval)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    log.info(f'Batches: Treino={len(train_loader)}, Val={len(valid_loader)}, Teste={len(test_loader)}')

    # Modelo
    model = build_resnet50(NUM_CLASSES, 'partial').to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr
    )
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=args.lr_gamma)

    log.info(f'Modelo: {sum(p.numel() for p in model.parameters()):,} parametros')
    log.info(f'Treinaveis: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')

    # Treinamento
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, _, _ = evaluate(model, valid_loader, criterion, device)
        scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        log.info(f'Epoch {epoch:2d}/{args.epochs} | '
                 f'Train Loss: {train_loss:.4f} Acc: {100*train_acc:.2f}% | '
                 f'Val Loss: {val_loss:.4f} Acc: {100*val_acc:.2f}%')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(weights_out))
            log.info(f'  >> Novo melhor modelo (val_acc: {100*val_acc:.2f}%)')

    log.info(f'Treinamento concluido! Melhor val_acc: {100*best_val_acc:.2f}%')

    # Plota curvas
    _plot_training_curves(history, best_val_acc, args.output_dir)

    # Avaliacao final
    log.info('Avaliando no test set...')
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    log.info(f'Test Acc: {100*test_acc:.2f}% | Test Loss: {test_loss:.4f}')
    log.info(f'\\n{classification_report(labels, preds, target_names=CLASS_NAMES, digits=4)}')

    # Matriz de confusao
    _plot_confusion_matrix(labels, preds, args.output_dir)

    # Salva resultados
    results_df = pd.DataFrame({
        'split': ['test'],
        'accuracy': [test_acc],
        'loss': [test_loss],
        'best_val_acc': [best_val_acc],
        'kappa': [cohen_kappa_score(labels, preds, weights='quadratic')]
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

    # Adiciona coluna de equipamento
    df_test = df_test.copy()
    df_test['machine'] = df_test['ceph_id'].map(ceph_to_machine)
    df_test_eval = df_test.dropna(subset=['machine']).copy()

    # Faz inferencia
    _, _, all_preds, all_labels = evaluate(model, test_loader, criterion, device)

    df_test_eval['pred_idx'] = [all_preds[i] for i in df_test_eval.index]
    df_test_eval['correct'] = df_test_eval['label_idx'] == df_test_eval['pred_idx']

    acc_by_machine = df_test_eval.groupby('machine').agg(
        accuracy=('correct', 'mean'),
        count=('correct', 'count')
    ).reset_index()
    acc_by_machine['accuracy'] *= 100

    log.info('Acuracia por Equipamento:')
    for _, row in acc_by_machine.iterrows():
        log.info(f'  {row["machine"]:30s}: {row["accuracy"]:5.1f}% ({int(row["count"])} amostras)')

    mean_acc = acc_by_machine['accuracy'].mean()
    std_acc = acc_by_machine['accuracy'].std()
    log.info(f'\nMedia: {mean_acc:.2f}% | Desvio: {std_acc:.2f}%')

    # Grafico
    _plot_accuracy_by_machine(acc_by_machine, mean_acc, 100 * (all_labels == all_preds).mean(), output_dir)

    return acc_by_machine


# =============================================================================
# LOEO CROSS-VALIDATION
# =============================================================================
def run_loeo(args):
    """Executa Leave-One-Equipment-Out Cross-Validation."""
    log.info('\n' + '=' * 70)
    log.info('LEAVE-ONE-EQUIPMENT-OUT CROSS-VALIDATION')
    log.info('=' * 70)

    device = torch.device(args.device)
    set_seed(args.seed)

    original_root = Path(args.original_root)
    cropped_root = Path(args.cropped_root) if args.cropped_root else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Carrega dados
    df_train = load_split('train', original_root, CLASS_NAMES)
    df_valid = load_split('valid', original_root, CLASS_NAMES)
    df_test = load_split('test', original_root, CLASS_NAMES)

    df_all = pd.concat([df_train, df_valid, df_test], ignore_index=True)

    ceph_to_machine = load_machine_mappings(original_root)
    df_all['machine'] = df_all['ceph_id'].map(ceph_to_machine)
    df_all = df_all.dropna(subset=['machine']).reset_index(drop=True)

    all_machines = sorted(df_all['machine'].unique())
    log.info(f'Equipamentos ({len(all_machines)}): {all_machines}')
    for m in all_machines:
        count = (df_all['machine'] == m).sum()
        log.info(f'  {m:30s}: {count:4d} imagens')

    transform_train, transform_eval = get_transforms(args.img_size)

    loeo_epochs = getattr(args, 'loeo_epochs', 20)
    loeo_lr = getattr(args, 'loeo_lr', 5e-5)

    loeo_results = []

    for i, held_out in enumerate(all_machines):
        log.info(f'\n--- Fold {i + 1}/{len(all_machines)}: Held-Out = {held_out} ---')

        df_train_fold = df_all[df_all['machine'] != held_out].copy()
        df_test_fold = df_all[df_all['machine'] == held_out].copy()

        log.info(f'  Treino: {len(df_train_fold)} imagens ({df_train_fold["machine"].nunique()} equip.)')
        log.info(f'  Teste:  {len(df_test_fold)} imagens')

        # Pesos do fold
        fold_counts = df_train_fold['label'].value_counts().reindex(CLASS_NAMES, fill_value=1)
        fold_weights = torch.tensor(
            [1.0 / max(fold_counts[c], 1) for c in CLASS_NAMES], dtype=torch.float
        ).to(device)
        fold_weights = fold_weights / fold_weights.sum() * NUM_CLASSES

        # Datasets
        train_ds_fold = CVMDataset(df_train_fold, original_root, cropped_root, transform_train)
        test_ds_fold = CVMDataset(df_test_fold, original_root, cropped_root, transform_eval)

        # Split interno 80/20
        n_train = len(train_ds_fold)
        indices = list(range(n_train))
        random.shuffle(indices)
        split = int(0.8 * n_train)
        train_sub = Subset(train_ds_fold, indices[:split])
        val_sub = Subset(train_ds_fold, indices[split:])

        train_loader = DataLoader(train_sub, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_sub, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_ds_fold, batch_size=args.batch_size, shuffle=False, num_workers=0)

        # Modelo
        model_fold = build_resnet50(NUM_CLASSES, 'full').to(device)
        criterion_fold = nn.CrossEntropyLoss(weight=fold_weights)
        optimizer_fold = optim.Adam(model_fold.parameters(), lr=loeo_lr)

        # Treino
        best_val = 0.0
        patience_counter = 0
        best_state = None

        for epoch in range(1, loeo_epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model_fold, train_loader, criterion_fold, optimizer_fold, device)
            val_loss, val_acc, _, _ = evaluate(
                model_fold, val_loader, criterion_fold, device)

            if val_acc > best_val:
                best_val = val_acc
                patience_counter = 0
                best_state = {k: v.clone() for k, v in model_fold.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= 5:
                    log.info(f'    Early stopping epoca {epoch}')
                    break

        # Avaliacao no held-out
        model_fold.load_state_dict(best_state)
        _, test_acc, preds, labels = evaluate(
            model_fold, test_loader, criterion_fold, device)
        kappa_fold = cohen_kappa_score(labels, preds, weights='quadratic')

        log.info(f'  >> Held-out: {held_out} | Acc: {100 * test_acc:.2f}% | Kappa: {kappa_fold:.4f}')

        loeo_results.append({
            'held_out': held_out,
            'n_train': len(df_train_fold),
            'n_test': len(df_test_fold),
            'accuracy': test_acc,
            'kappa': kappa_fold
        })

        # Salva modelo
        fold_path = output_dir / f'model_loeo_{held_out.replace(" ", "_")}.pth'
        torch.save(best_state, str(fold_path))

    # Resultados
    loeo_df = pd.DataFrame([{
        'Held-Out Machine': r['held_out'],
        'Train Size': r['n_train'],
        'Test Size': r['n_test'],
        'Accuracy (%)': round(100 * r['accuracy'], 2),
        'Cohen Kappa': round(r['kappa'], 4)
    } for r in loeo_results])

    log.info('\n' + '=' * 70)
    log.info('RESULTADOS LOEO')
    log.info('=' * 70)
    for _, row in loeo_df.iterrows():
        log.info(f'  {row["Held-Out Machine"]:30s}: {row["Accuracy (%)"]:.2f}% | '
                 f'Kappa: {row["Cohen Kappa"]:.4f}')

    mean_acc = loeo_df['Accuracy (%)'].mean()
    std_acc = loeo_df['Accuracy (%)'].std()
    log.info(f'\nLOEO Medio: {mean_acc:.2f}% +- {std_acc:.2f}%')
    log.info(f'Kappa Medio: {loeo_df["Cohen Kappa"].mean():.4f}')

    loeo_df.to_csv(str(output_dir / 'loeo_results.csv'), index=False)
    _plot_loeo_results(loeo_df, mean_acc, output_dir)

    return loeo_df


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
                     label=f'Best: {100 * best_val_acc:.2f}%')
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


def _plot_loeo_results(loeo_df, mean_acc, output_dir):
    """Salva graficos LOEO."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Acuracia
    loeo_sorted = loeo_df.sort_values('Accuracy (%)', ascending=True)
    colors = [plt.cm.RdYlGn(acc / 100) for acc in loeo_sorted['Accuracy (%)']]
    axes[0].barh(loeo_sorted['Held-Out Machine'], loeo_sorted['Accuracy (%)'], color=colors)
    axes[0].set_xlabel('Acuracia (%)')
    axes[0].set_title('LOEO: Acuracia por Equipamento Held-Out')
    axes[0].set_xlim(0, 105)
    for i, (_, row) in enumerate(loeo_sorted.iterrows()):
        axes[0].text(row['Accuracy (%)'] + 0.5, i,
                     f'{row["Accuracy (%)"]:.1f}% (n={int(row["Test Size"])})',
                     va='center', fontsize=9)
    axes[0].axvline(x=mean_acc, color='blue', linestyle='--', alpha=0.7,
                    label=f'Media: {mean_acc:.1f}%')
    axes[0].legend()

    # Kappa
    loeo_sorted_k = loeo_df.sort_values('Cohen Kappa', ascending=True)
    colors_k = [plt.cm.RdYlGn(k / 1.0) for k in loeo_sorted_k['Cohen Kappa']]
    axes[1].barh(loeo_sorted_k['Held-Out Machine'], loeo_sorted_k['Cohen Kappa'], color=colors_k)
    axes[1].set_xlabel('Cohen Kappa')
    axes[1].set_title('LOEO: Concordancia por Equipamento Held-Out')
    axes[1].set_xlim(0, 1.0)
    for i, (_, row) in enumerate(loeo_sorted_k.iterrows()):
        axes[1].text(row['Cohen Kappa'] + 0.01, i,
                     f'{row["Cohen Kappa"]:.3f} (n={int(row["Test Size"])})',
                     va='center', fontsize=9)
    mean_kappa = loeo_df['Cohen Kappa'].mean()
    axes[1].axvline(x=mean_kappa, color='blue', linestyle='--', alpha=0.7,
                    label=f'Media: {mean_kappa:.3f}')
    axes[1].legend()

    plt.tight_layout()
    path = Path(output_dir) / 'loeo_results.png'
    plt.savefig(str(path), dpi=150, bbox_inches='tight')
    plt.close()
    log.info(f'Grafico LOEO salvo: {path}')


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    """Configura argumentos de linha de comando."""
    parser = argparse.ArgumentParser(
        description='Pipeline CVM: Classificacao da Maturacao Cervical Vertebral',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  # Pipeline completo
  python pipeline_cvm.py --stage all

  # Usando GPU com hiperparametros customizados
  python pipeline_cvm.py --stage all --device cuda --epochs 100 --batch-size 32

  # So extrair ROI (se YOLO ja treinado)
  python pipeline_cvm.py --stage crop --yolo-weights best_yolo_cvm.pt

  # LOEO com poucas epocas para teste
  python pipeline_cvm.py --stage loeo --loeo-epochs 5
        """
    )

    # Estagio
    parser.add_argument('--stage', type=str, default='all',
                        choices=['all', 'train_yolo', 'crop', 'train_classifier',
                                 'evaluate', 'loeo'],
                        help='Estagio do pipeline a executar (default: all)')

    # Caminhos
    parser.add_argument('--original-root', type=str, default='Aariz_extracted/Aariz',
                        help='Diretorio raiz do dataset original')
    parser.add_argument('--cropped-root', type=str, default='Aariz_cropped_CVM',
                        help='Diretorio para salvar crops')
    parser.add_argument('--roboflow-dataset', type=str, default='roboflow_cvm_dataset',
                        help='Diretorio do dataset Roboflow exportado')
    parser.add_argument('--yolo-weights', type=str, default='best_yolo_cvm.pt',
                        help='Caminho para pesos YOLO')
    parser.add_argument('--resnet-weights', type=str, default='best_resnet_cvm.pth',
                        help='Caminho para salvar pesos ResNet')
    parser.add_argument('--output-dir', type=str, default='resultados',
                        help='Diretorio de saida para resultados e graficos')

    # Hiperparametros
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed para reproducibilidade')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Dispositivo (cuda / cpu)')
    parser.add_argument('--img-size', type=int, default=224,
                        help='Tamanho das imagens de entrada')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Numero de epocas para treino do classificador')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--lr-step', type=int, default=10,
                        help='Epocas entre reducoes de LR')
    parser.add_argument('--lr-gamma', type=float, default=0.5,
                        help='Fator de reducao do LR')

    # LOEO
    parser.add_argument('--loeo-epochs', type=int, default=20,
                        help='Epocas por fold no LOEO')
    parser.add_argument('--loeo-lr', type=float, default=5e-5,
                        help='Learning rate para LOEO')

    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================
def main():
    args = parse_args()

    # Cria diretorios
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.cropped_root:
        Path(args.cropped_root).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    set_seed(args.seed)

    log.info('=' * 60)
    log.info(f'Pipeline CVM v1.0')
    log.info(f'Estagio: {args.stage}')
    log.info(f'Dispositivo: {device}')
    log.info(f'Dataset: {args.original_root}')
    log.info(f'Seed: {args.seed}')
    log.info('=' * 60)

    # Estagio: Treinar YOLO
    if args.stage in ('all', 'train_yolo'):
        train_yolo(
            Path(args.roboflow_dataset),
            Path(args.yolo_weights),
            device
        )

    # Estagio: Crop
    if args.stage in ('all', 'crop'):
        crop_all_splits(
            Path(args.yolo_weights),
            Path(args.original_root),
            Path(args.cropped_root)
        )

    # Estagio: Treinar Classificador
    if args.stage in ('all', 'train_classifier'):
        model, test_loader, test_acc, preds, labels, df_test = train_classifier(args)
    else:
        model = test_loader = test_acc = preds = labels = df_test = None

    # Estagio: Avaliar
    if args.stage in ('all', 'evaluate'):
        if model is None:
            # Carrega modelo salvo
            weights_path = Path(args.resnet_weights)
            if not weights_path.exists():
                log.error(f'Modelo nao encontrado: {weights_path}. Execute train_classifier primeiro.')
                return
            log.info(f'Carregando modelo salvo: {weights_path}')
            model = build_resnet50(NUM_CLASSES, 'partial').to(device)
            model.load_state_dict(torch.load(weights_path, map_location=device))

            # Recria dataloader de teste
            transform_train, transform_eval = get_transforms(args.img_size)
            df_test = load_split('test', Path(args.original_root), CLASS_NAMES)
            test_ds = CVMDataset(df_test, Path(args.original_root),
                                 Path(args.cropped_root) if args.cropped_root else None,
                                 transform_eval)
            test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

            # Criterion dummy (apenas para inferencia)
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

    # Estagio: LOEO
    if args.stage in ('all', 'loeo'):
        run_loeo(args)

    log.info('\nPipeline finalizado com sucesso!')


if __name__ == '__main__':
    main()
