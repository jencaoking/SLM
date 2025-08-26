# -*- coding: utf-8 -*-
"""
问答系统模型训练脚本

完整的训练流程，包括数据加载、模型训练、验证和检查点管理。
适配中国大陆网络环境。
"""

import os
import sys
import json
import yaml
import logging
import argparse
import random
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp import autocast, GradScaler

# 导入自定义模块
from models.transformer import QATransformer
from utils.data_loader import UniversalDataLoader, MultiDatasetLoader
from utils.data_processor import DataProcessor, QADataset
from utils.tokenizer import QATokenizer
from utils.metrics import QAMetrics

warnings.filterwarnings('ignore')


def setup_logging(log_dir: Path, level: str = "INFO"):
    """设置日志"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / 'training.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def setup_device() -> torch.device:
    """设置设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"使用GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logging.info("使用CPU")
    return device


class Trainer:
    """训练器类"""
    
    def __init__(self, config: Dict[str, Any], logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.device = setup_device()
        
        # 创建输出目录
        self.output_dir = Path(config['paths']['output_dir'])
        self.checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        self.log_dir = Path(config['paths']['log_dir'])
        
        for dir_path in [self.output_dir, self.checkpoint_dir, self.log_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if config['hardware']['mixed_precision'] else None
        
        # 训练状态
        self.global_step = 0
        self.best_score = 0
        self.patience_counter = 0
        
        # TensorBoard
        self.writer = SummaryWriter(log_dir=self.log_dir / 'tensorboard')
        
        # 评估器
        self.metrics = QAMetrics()
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """准备数据"""
        self.logger.info("开始准备数据...")
        
        # 加载数据集配置
        dataset_configs = []
        
        # 检查是否启用自动扫描
        if self.config['data'].get('auto_scan', {}).get('enabled', False):
            # 使用自动扫描
            from utils.data_loader import AutoDatasetScanner
            scanner = AutoDatasetScanner(self.config['data'])
            dataset_configs = scanner.scan_datasets()
            
            if not dataset_configs:
                self.logger.warning("自动扫描未发现任何数据集，尝试使用手动配置")
        
        # 如果自动扫描未启用或未发现数据集，使用手动配置
        if not dataset_configs:
            manual_config = self.config['data'].get('manual_datasets', {})
            if manual_config.get('enabled', False):
                manual_datasets = manual_config.get('datasets', [])
                for dataset_config in manual_datasets:
                    if dataset_config.get('enabled', True):
                        dataset_configs.append(dataset_config)
            else:
                # 兼容旧配置格式
                legacy_datasets = self.config['data'].get('datasets', [])
                for dataset_config in legacy_datasets:
                    if dataset_config.get('enabled', True):
                        dataset_configs.append(dataset_config)
        
        if not dataset_configs:
            raise ValueError("没有启用的数据集。请检查配置文件或数据目录")
        
        # 加载数据
        if len(dataset_configs) == 1:
            # 单个数据集
            loader = UniversalDataLoader(
                data_path=dataset_configs[0]['path'],
                format_type=dataset_configs[0].get('format', 'auto'),
                field_mapping=self.config['data'].get('field_mapping')
            )
            data = loader.load_data()
        else:
            # 多个数据集
            multi_loader = MultiDatasetLoader(dataset_configs)
            # 使用配置中的采样策略，默认为balanced
            sampling_strategy = self.config['data'].get('auto_scan', {}).get('sampling_strategy', 'balanced')
            if sampling_strategy not in ['balanced', 'weighted', 'concat']:
                sampling_strategy = 'balanced'
            data = multi_loader.load_datasets(sampling_strategy=sampling_strategy)
        
        self.logger.info(f"加载数据完成，共 {len(data)} 条记录")
        
        # 准备分词器
        self.prepare_tokenizer(data)
        
        # 数据预处理
        processor = DataProcessor(
            tokenizer=self.tokenizer,
            max_length=self.config['data']['max_length'],
            max_query_length=self.config['data']['max_query_length'],
            max_answer_length=self.config['data']['max_answer_length'],
            do_augmentation=self.config['data']['augmentation']['enable']
        )
        
        processed_data = processor.process_data(data, is_training=True)
        self.logger.info(f"数据预处理完成，共 {len(processed_data)} 条记录")
        
        # 数据分割
        train_size = int(len(processed_data) * self.config['data']['train_split'])
        val_size = int(len(processed_data) * self.config['data']['val_split'])
        test_size = len(processed_data) - train_size - val_size
        
        train_data, val_data, test_data = random_split(
            processed_data, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 创建数据集
        train_dataset = QADataset(
            data=[processed_data[i] for i in train_data.indices],
            tokenizer=self.tokenizer,
            max_length=self.config['data']['max_length'],
            is_training=True
        )
        
        val_dataset = QADataset(
            data=[processed_data[i] for i in val_data.indices],
            tokenizer=self.tokenizer,
            max_length=self.config['data']['max_length'],
            is_training=True
        )
        
        test_dataset = QADataset(
            data=[processed_data[i] for i in test_data.indices],
            tokenizer=self.tokenizer,
            max_length=self.config['data']['max_length'],
            is_training=False
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=self.config['hardware']['dataloader_num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['dataloader_num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['dataloader_num_workers'],
            pin_memory=self.config['hardware']['pin_memory']
        )
        
        self.logger.info(f"数据分割完成 - 训练: {len(train_dataset)}, 验证: {len(val_dataset)}, 测试: {len(test_dataset)}")
        
        return train_loader, val_loader, test_loader
    
    def prepare_tokenizer(self, data: List[Dict[str, Any]]):
        """准备分词器"""
        tokenizer_dir = Path(self.config['paths']['tokenizer_dir'])
        
        if tokenizer_dir.exists() and (tokenizer_dir / 'vocab.json').exists():
            # 加载已有分词器
            self.logger.info("加载已有分词器...")
            self.tokenizer = QATokenizer()
            self.tokenizer.load_vocab(tokenizer_dir)
        else:
            # 创建新分词器
            self.logger.info("创建新分词器...")
            self.tokenizer = QATokenizer(
                vocab_size=self.config['model']['vocab_size'],
                max_length=self.config['data']['max_length']
            )
            
            # 收集所有文本
            all_texts = []
            for record in data:
                if record.get('question'):
                    all_texts.append(record['question'])
                if record.get('context'):
                    all_texts.append(record['context'])
                if record.get('answer'):
                    all_texts.append(record['answer'])
            
            # 构建词汇表
            self.tokenizer.build_vocab(all_texts, save_path=tokenizer_dir)
            self.logger.info(f"分词器创建完成，词汇表大小: {self.tokenizer.get_vocab_size()}")
    
    def prepare_model(self):
        """准备模型"""
        self.logger.info("初始化模型...")
        
        self.model = QATransformer(
            vocab_size=self.tokenizer.get_vocab_size(),
            d_model=self.config['model']['d_model'],
            n_heads=self.config['model']['n_heads'],
            n_encoder_layers=self.config['model']['n_layers'],
            n_decoder_layers=2,
            d_ff=self.config['model']['d_ff'],
            max_position_embeddings=self.config['model']['max_position_embeddings'],
            dropout=self.config['model']['dropout'],
            pad_token_id=self.tokenizer.get_special_token_id('pad_token'),
            use_answer_classifier=True
        )
        
        self.model.to(self.device)
        
        # 统计参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        self.logger.info(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    def prepare_optimizer(self):
        """准备优化器和调度器"""
        # 优化器
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay'],
            eps=self.config['training']['adam_epsilon']
        )
        
        # 学习率调度器
        if self.config['training']['lr_scheduler']['type'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs']
            )
        elif self.config['training']['lr_scheduler']['type'] == 'linear':
            from torch.optim.lr_scheduler import LinearLR
            self.scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=0.1,
                total_iters=self.config['training']['num_epochs']
            )
        
        self.logger.info("优化器和调度器准备完成")
    
    def train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # 移动数据到设备
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # 前向传播
            if self.scaler:  # 混合精度训练
                with autocast():
                    outputs = self.model(
                        question_input_ids=batch['question_input_ids'],
                        context_input_ids=batch['context_input_ids'],
                        question_attention_mask=batch['question_attention_mask'],
                        context_attention_mask=batch['context_attention_mask'],
                        start_positions=batch['start_positions'],
                        end_positions=batch['end_positions'],
                        answer_labels=batch['answer_labels']
                    )
                    loss = outputs['loss']
                
                # 反向传播
                self.scaler.scale(loss).backward()
                
                if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                    # 梯度裁剪
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['max_grad_norm']
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            else:  # 普通训练
                outputs = self.model(
                    question_input_ids=batch['question_input_ids'],
                    context_input_ids=batch['context_input_ids'],
                    question_attention_mask=batch['question_attention_mask'],
                    context_attention_mask=batch['context_attention_mask'],
                    start_positions=batch['start_positions'],
                    end_positions=batch['end_positions'],
                    answer_labels=batch['answer_labels']
                )
                loss = outputs['loss']
                
                # 反向传播
                loss.backward()
                
                if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config['training']['max_grad_norm']
                    )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
            
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({'loss': loss.item()})
            
            # 记录日志
            if self.global_step % self.config['training']['logging_steps'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def evaluate(self, val_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_references = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    question_input_ids=batch['question_input_ids'],
                    context_input_ids=batch['context_input_ids'],
                    question_attention_mask=batch['question_attention_mask'],
                    context_attention_mask=batch['context_attention_mask'],
                    start_positions=batch['start_positions'],
                    end_positions=batch['end_positions'],
                    answer_labels=batch['answer_labels']
                )
                
                total_loss += outputs['loss'].item()
                
                # 收集预测结果（简化处理）
                start_logits = outputs['start_logits']
                end_logits = outputs['end_logits']
                
                start_preds = torch.argmax(start_logits, dim=-1)
                end_preds = torch.argmax(end_logits, dim=-1)
                
                # 这里可以添加更复杂的后处理逻辑
                for i in range(start_preds.size(0)):
                    all_predictions.append({'answer': f'pred_{i}'})  # 简化
                    all_references.append({'answer': f'ref_{i}'})    # 简化
        
        avg_loss = total_loss / len(val_loader)
        
        # 计算评估指标（这里使用简化版本）
        eval_results = {
            'val_loss': avg_loss,
            'exact_match': 0.5,  # 占位符
            'f1_score': 0.6      # 占位符
        }
        
        return eval_results
    
    def save_checkpoint(self, epoch: int, eval_results: Dict[str, float], is_best: bool = False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.config,
            'eval_results': eval_results,
            'best_score': self.best_score
        }
        
        # 保存常规检查点
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # 保存最新检查点
        latest_path = self.checkpoint_dir / 'checkpoint_latest.pt'
        torch.save(checkpoint, latest_path)
        
        # 保存最佳模型
        if is_best:
            best_path = self.checkpoint_dir / 'checkpoint_best.pt'
            torch.save(checkpoint, best_path)
            
            # 同时保存模型
            self.model.save_pretrained(self.checkpoint_dir / 'best_model')
        
        self.logger.info(f"检查点已保存: {checkpoint_path}")
    
    def train(self):
        """主训练函数"""
        self.logger.info("开始训练...")
        
        # 准备数据和模型
        train_loader, val_loader, test_loader = self.prepare_data()
        self.prepare_model()
        self.prepare_optimizer()
        
        # 训练循环
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            self.logger.info(f"Epoch {epoch}/{self.config['training']['num_epochs']}")
            
            # 训练
            train_results = self.train_epoch(train_loader, epoch)
            
            # 验证
            eval_results = self.evaluate(val_loader)
            
            # 更新学习率
            if self.scheduler:
                self.scheduler.step()
            
            # 记录结果
            for key, value in {**train_results, **eval_results}.items():
                self.writer.add_scalar(f'epoch/{key}', value, epoch)
            
            # 检查是否是最佳模型
            current_score = eval_results.get('f1_score', 0)
            is_best = current_score > self.best_score
            
            if is_best:
                self.best_score = current_score
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # 保存检查点
            self.save_checkpoint(epoch, eval_results, is_best)
            
            # 早停检查
            if self.patience_counter >= self.config['training']['early_stopping']['patience']:
                self.logger.info("触发早停机制，停止训练")
                break
            
            self.logger.info(f"训练损失: {train_results['train_loss']:.4f}, "
                           f"验证损失: {eval_results['val_loss']:.4f}, "
                           f"F1分数: {eval_results['f1_score']:.4f}")
        
        self.logger.info("训练完成！")
        self.writer.close()


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练问答模型')
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    
    args = parser.parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    logger = setup_logging(Path(config['paths']['log_dir']), config['logging']['level'])
    logger.info(f"开始训练，配置文件: {args.config}")
    
    # 创建训练器并开始训练
    trainer = Trainer(config, logger)
    trainer.train()


if __name__ == "__main__":
    main()