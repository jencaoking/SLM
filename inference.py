# -*- coding: utf-8 -*-
"""
问答系统推理脚本

提供模型推理接口，支持单个问题和批量推理。
"""

import os
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.transformer import QATransformer
from utils.tokenizer import QATokenizer
from utils.data_processor import DataProcessor


class QAInference:
    """问答推理器"""
    
    def __init__(
        self, 
        model_path: Union[str, Path],
        tokenizer_path: Optional[Union[str, Path]] = None,
        device: Optional[str] = None
    ):
        """
        初始化推理器
        
        Args:
            model_path: 模型路径
            tokenizer_path: 分词器路径
            device: 设备类型
        """
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path) if tokenizer_path else self.model_path.parent / 'tokenizer'
        
        # 设置设备
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # 加载模型和分词器
        self.model = None
        self.tokenizer = None
        self._load_model()
        self._load_tokenizer()
        
        # 设置日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"推理器初始化完成，设备: {self.device}")
    
    def _load_model(self):
        """加载模型"""
        if self.model_path.is_file():
            # 从检查点加载
            checkpoint = torch.load(self.model_path, map_location=self.device)
            config = checkpoint['config']
            
            # 创建模型
            self.model = QATransformer(
                vocab_size=config['model']['vocab_size'],
                d_model=config['model']['d_model'],
                n_heads=config['model']['n_heads'],
                n_encoder_layers=config['model']['n_layers'],
                d_ff=config['model']['d_ff'],
                max_position_embeddings=config['model']['max_position_embeddings'],
                dropout=config['model']['dropout'],
                use_answer_classifier=True
            )
            
            # 加载权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
        elif self.model_path.is_dir():
            # 从目录加载
            self.model = QATransformer.from_pretrained(self.model_path)
        else:
            raise ValueError(f"模型路径不存在: {self.model_path}")
        
        self.model.to(self.device)
        self.model.eval()
        self.logger.info("模型加载完成")
    
    def _load_tokenizer(self):
        """加载分词器"""
        if not self.tokenizer_path.exists():
            raise ValueError(f"分词器路径不存在: {self.tokenizer_path}")
        
        self.tokenizer = QATokenizer()
        self.tokenizer.load_vocab(self.tokenizer_path)
        self.logger.info("分词器加载完成")
    
    def predict_single(
        self, 
        question: str, 
        context: str,
        n_best: int = 1,
        max_answer_length: int = 128,
        confidence_threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        单个问题推理
        
        Args:
            question: 问题文本
            context: 上下文文本
            n_best: 返回最佳答案数量
            max_answer_length: 最大答案长度
            confidence_threshold: 置信度阈值
            
        Returns:
            预测结果字典
        """
        # 编码输入
        question_encoded = self.tokenizer.encode(
            question, 
            max_length=64, 
            padding=True, 
            truncation=True
        )
        
        context_encoded = self.tokenizer.encode(
            context, 
            max_length=512, 
            padding=True, 
            truncation=True
        )
        
        # 转换为tensor
        question_ids = torch.tensor([question_encoded['input_ids']], dtype=torch.long).to(self.device)
        question_mask = torch.tensor([question_encoded['attention_mask']], dtype=torch.long).to(self.device)
        context_ids = torch.tensor([context_encoded['input_ids']], dtype=torch.long).to(self.device)
        context_mask = torch.tensor([context_encoded['attention_mask']], dtype=torch.long).to(self.device)
        
        # 模型推理
        with torch.no_grad():
            predictions = self.model.predict(
                question_input_ids=question_ids,
                context_input_ids=context_ids,
                question_attention_mask=question_mask,
                context_attention_mask=context_mask,
                n_best=n_best,
                max_answer_length=max_answer_length
            )
        
        # 处理预测结果
        result = self._process_predictions(
            predictions, 
            context, 
            context_encoded['tokens'],
            confidence_threshold
        )
        
        return result
    
    def predict_batch(
        self, 
        qa_pairs: List[Dict[str, str]],
        batch_size: int = 8,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        批量推理
        
        Args:
            qa_pairs: 问答对列表，每个元素包含'question'和'context'
            batch_size: 批处理大小
            **kwargs: 其他推理参数
            
        Returns:
            预测结果列表
        """
        results = []
        
        for i in tqdm(range(0, len(qa_pairs), batch_size), desc="批量推理"):
            batch = qa_pairs[i:i + batch_size]
            batch_results = []
            
            for qa_pair in batch:
                try:
                    result = self.predict_single(
                        question=qa_pair['question'],
                        context=qa_pair['context'],
                        **kwargs
                    )
                    batch_results.append(result)
                except Exception as e:
                    self.logger.error(f"推理失败: {e}")
                    batch_results.append({
                        'question': qa_pair['question'],
                        'predictions': [],
                        'error': str(e)
                    })
            
            results.extend(batch_results)
        
        return results
    
    def _process_predictions(
        self, 
        predictions: Dict[str, Any], 
        context: str,
        context_tokens: List[str],
        confidence_threshold: float
    ) -> Dict[str, Any]:
        """处理预测结果"""
        if not predictions['predictions'] or not predictions['predictions'][0]:
            return {
                'predictions': [],
                'best_answer': '',
                'confidence': 0.0,
                'has_answer': False
            }
        
        sample_predictions = predictions['predictions'][0]
        processed_predictions = []
        
        for pred in sample_predictions:
            start_pos = pred['start_position']
            end_pos = pred['end_position']
            score = pred['score']
            
            # 提取答案文本
            if start_pos < len(context_tokens) and end_pos < len(context_tokens):
                answer_tokens = context_tokens[start_pos:end_pos + 1]
                answer_text = self.tokenizer.decode(
                    [self.tokenizer.token2id.get(token, 0) for token in answer_tokens],
                    skip_special_tokens=True
                )
            else:
                answer_text = ""
            
            processed_predictions.append({
                'answer': answer_text,
                'confidence': score,
                'start_position': start_pos,
                'end_position': end_pos
            })
        
        # 获取最佳答案
        best_prediction = processed_predictions[0] if processed_predictions else {}
        best_answer = best_prediction.get('answer', '')
        best_confidence = best_prediction.get('confidence', 0.0)
        
        # 判断是否有答案
        has_answer = best_confidence >= confidence_threshold and len(best_answer.strip()) > 0
        
        return {
            'predictions': processed_predictions,
            'best_answer': best_answer,
            'confidence': best_confidence,
            'has_answer': has_answer
        }
    
    def interactive_mode(self):
        """交互模式"""
        print("=== 问答系统交互模式 ===")
        print("输入 'quit' 退出程序")
        print("输入格式: 问题|上下文")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\n请输入问题和上下文 (用|分隔): ").strip()
                
                if user_input.lower() == 'quit':
                    print("退出程序")
                    break
                
                if '|' not in user_input:
                    print("请按照 '问题|上下文' 的格式输入")
                    continue
                
                question, context = user_input.split('|', 1)
                question = question.strip()
                context = context.strip()
                
                if not question or not context:
                    print("问题和上下文都不能为空")
                    continue
                
                print(f"\n问题: {question}")
                print(f"上下文: {context[:100]}..." if len(context) > 100 else f"上下文: {context}")
                
                # 推理
                result = self.predict_single(question, context, n_best=3)
                
                print(f"\n推理结果:")
                if result['has_answer']:
                    print(f"最佳答案: {result['best_answer']}")
                    print(f"置信度: {result['confidence']:.4f}")
                    
                    if len(result['predictions']) > 1:
                        print(f"\n其他候选答案:")
                        for i, pred in enumerate(result['predictions'][1:], 1):
                            print(f"  {i}. {pred['answer']} (置信度: {pred['confidence']:.4f})")
                else:
                    print("未找到答案")
                
            except KeyboardInterrupt:
                print("\n\n程序被中断")
                break
            except Exception as e:
                print(f"错误: {e}")
    
    def evaluate_file(self, input_file: str, output_file: str):
        """评估文件中的问答对"""
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            if input_file.endswith('.json'):
                data = json.load(f)
            elif input_file.endswith('.jsonl'):
                data = [json.loads(line) for line in f]
            else:
                raise ValueError("不支持的文件格式，请使用JSON或JSONL")
        
        # 准备推理数据
        qa_pairs = []
        for item in data:
            qa_pairs.append({
                'question': item.get('question', ''),
                'context': item.get('context', ''),
                'reference_answer': item.get('answer', '')
            })
        
        # 批量推理
        results = self.predict_batch(qa_pairs)
        
        # 保存结果
        output_data = []
        for i, (qa_pair, result) in enumerate(zip(qa_pairs, results)):
            output_item = {
                'id': i,
                'question': qa_pair['question'],
                'context': qa_pair['context'],
                'reference_answer': qa_pair['reference_answer'],
                'predicted_answer': result.get('best_answer', ''),
                'confidence': result.get('confidence', 0.0),
                'has_answer': result.get('has_answer', False),
                'all_predictions': result.get('predictions', [])
            }
            output_data.append(output_item)
        
        # 保存到文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"评估完成，结果已保存到: {output_file}")
        
        # 简单统计
        total_questions = len(output_data)
        answered_questions = sum(1 for item in output_data if item['has_answer'])
        avg_confidence = sum(item['confidence'] for item in output_data) / total_questions
        
        print(f"\n=== 评估统计 ===")
        print(f"总问题数: {total_questions}")
        print(f"有答案问题数: {answered_questions}")
        print(f"答案率: {answered_questions / total_questions * 100:.2f}%")
        print(f"平均置信度: {avg_confidence:.4f}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='问答系统推理')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--tokenizer_path', type=str, help='分词器路径')
    parser.add_argument('--mode', type=str, choices=['interactive', 'single', 'batch', 'file'], 
                       default='interactive', help='推理模式')
    parser.add_argument('--question', type=str, help='单个问题')
    parser.add_argument('--context', type=str, help='单个上下文')
    parser.add_argument('--input_file', type=str, help='输入文件路径')
    parser.add_argument('--output_file', type=str, help='输出文件路径')
    parser.add_argument('--device', type=str, help='设备类型')
    
    args = parser.parse_args()
    
    # 创建推理器
    inference = QAInference(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        device=args.device
    )
    
    # 根据模式执行推理
    if args.mode == 'interactive':
        inference.interactive_mode()
    
    elif args.mode == 'single':
        if not args.question or not args.context:
            print("单个推理模式需要提供 --question 和 --context 参数")
            return
        
        result = inference.predict_single(args.question, args.context)
        print(f"问题: {args.question}")
        print(f"上下文: {args.context}")
        print(f"答案: {result['best_answer']}")
        print(f"置信度: {result['confidence']:.4f}")
    
    elif args.mode == 'file':
        if not args.input_file or not args.output_file:
            print("文件模式需要提供 --input_file 和 --output_file 参数")
            return
        
        inference.evaluate_file(args.input_file, args.output_file)
    
    else:
        print("请选择有效的推理模式")


if __name__ == "__main__":
    main()