# -*- coding: utf-8 -*-
"""
数据验证脚本

验证问答数据集的格式、完整性和质量。
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from tqdm import tqdm

from utils.format_detector import FormatDetector
from utils.data_loader import UniversalDataLoader


class DataValidator:
    """数据验证器"""
    
    def __init__(self, log_level: str = "INFO"):
        """初始化数据验证器"""
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        self.format_detector = FormatDetector()
        self.validation_results = {}
        
    def validate_file(self, file_path: str, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        验证单个数据文件
        
        Args:
            file_path: 数据文件路径
            config_path: 配置文件路径
            
        Returns:
            验证结果字典
        """
        file_path = Path(file_path)
        self.logger.info(f"开始验证文件: {file_path}")
        
        results = {
            'file_path': str(file_path),
            'format_detection': {},
            'data_loading': {},
            'field_validation': {},
            'quality_check': {},
            'statistics': {},
            'issues': [],
            'overall_status': 'unknown'
        }
        
        try:
            # 1. 格式检测
            self.logger.info("步骤 1/5: 格式检测")
            format_result = self._validate_format(file_path)
            results['format_detection'] = format_result
            
            if not format_result['is_valid']:
                results['overall_status'] = 'failed'
                results['issues'].append('文件格式无效')
                return results
            
            # 2. 数据加载
            self.logger.info("步骤 2/5: 数据加载")
            loading_result = self._validate_loading(file_path, format_result['format'])
            results['data_loading'] = loading_result
            
            if not loading_result['success']:
                results['overall_status'] = 'failed'
                results['issues'].append('数据加载失败')
                return results
            
            data = loading_result['data']
            
            # 3. 字段验证
            self.logger.info("步骤 3/5: 字段验证")
            field_result = self._validate_fields(data)
            results['field_validation'] = field_result
            
            # 4. 质量检查
            self.logger.info("步骤 4/5: 数据质量检查")
            quality_result = self._validate_quality(data)
            results['quality_check'] = quality_result
            
            # 5. 统计信息
            self.logger.info("步骤 5/5: 统计信息生成")
            stats_result = self._generate_statistics(data)
            results['statistics'] = stats_result
            
            # 综合评估
            results['overall_status'] = self._get_overall_status(results)
            
        except Exception as e:
            self.logger.error(f"验证过程出错: {e}")
            results['overall_status'] = 'error'
            results['issues'].append(f'验证异常: {str(e)}')
        
        self.validation_results[str(file_path)] = results
        return results
    
    def _validate_format(self, file_path: Path) -> Dict[str, Any]:
        """验证文件格式"""
        try:
            detection_result = self.format_detector.detect_format(file_path)
            
            if detection_result['is_valid']:
                self.logger.info(f"文件格式: {detection_result['format']}")
                return {
                    'is_valid': True,
                    'format': detection_result['format'],
                    'details': detection_result.get('validation_details', {}),
                    'file_size': detection_result.get('file_size', 0)
                }
            else:
                self.logger.error(f"格式检测失败: {detection_result.get('error', '未知错误')}")
                return {
                    'is_valid': False,
                    'format': 'unknown',
                    'error': detection_result.get('error', '格式检测失败')
                }
        except Exception as e:
            return {
                'is_valid': False,
                'format': 'unknown',
                'error': f'格式检测异常: {str(e)}'
            }
    
    def _validate_loading(self, file_path: Path, file_format: str) -> Dict[str, Any]:
        """验证数据加载"""
        try:
            loader = UniversalDataLoader(data_path=file_path, format_type=file_format)
            data = loader.load_data()
            
            return {
                'success': True,
                'data': data,
                'record_count': len(data),
                'loading_time': 0  # 可以添加时间统计
            }
        except Exception as e:
            self.logger.error(f"数据加载失败: {e}")
            return {
                'success': False,
                'data': [],
                'error': str(e)
            }
    
    def _validate_fields(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """验证字段完整性"""
        if not data:
            return {
                'is_valid': False,
                'error': '数据为空'
            }
        
        required_fields = ['question', 'context', 'answer']
        optional_fields = ['answer_start', 'question_id', 'title', 'is_impossible']
        
        # 检查字段存在性
        available_fields = set()
        for record in data[:100]:  # 检查前100条记录
            available_fields.update(record.keys())
        
        missing_required = [field for field in required_fields if field not in available_fields]
        available_optional = [field for field in optional_fields if field in available_fields]
        
        # 检查字段覆盖率
        field_coverage = {}
        field_types = {}
        
        for field in available_fields:
            non_empty_count = 0
            sample_values = []
            
            for record in data:
                value = record.get(field)
                if value is not None and str(value).strip():
                    non_empty_count += 1
                    if len(sample_values) < 10:
                        sample_values.append(value)
            
            field_coverage[field] = non_empty_count / len(data)
            
            # 推断数据类型
            if sample_values:
                field_types[field] = type(sample_values[0]).__name__
        
        # 验证关键字段覆盖率
        issues = []
        for field in required_fields:
            if field in field_coverage:
                coverage = field_coverage[field]
                if coverage < 0.95:  # 要求95%以上覆盖率
                    issues.append(f'字段 {field} 覆盖率不足: {coverage:.2%}')
        
        return {
            'is_valid': len(missing_required) == 0,
            'available_fields': list(available_fields),
            'missing_required': missing_required,
            'available_optional': available_optional,
            'field_coverage': field_coverage,
            'field_types': field_types,
            'issues': issues
        }
    
    def _validate_quality(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """验证数据质量"""
        issues = []
        quality_metrics = {}
        
        # 检查文本长度
        question_lengths = []
        context_lengths = []
        answer_lengths = []
        
        # 检查重复
        question_set = set()
        duplicate_questions = 0
        
        # 检查答案位置准确性
        invalid_answer_positions = 0
        
        for i, record in enumerate(tqdm(data, desc="质量检查")):
            # 文本长度检查
            question = record.get('question', '')
            context = record.get('context', '')
            answer = record.get('answer', '')
            
            if question:
                question_lengths.append(len(question.split()))
                
                # 重复检查
                if question in question_set:
                    duplicate_questions += 1
                else:
                    question_set.add(question)
            
            if context:
                context_lengths.append(len(context.split()))
            
            if answer:
                answer_lengths.append(len(answer.split()))
                
                # 答案位置检查
                answer_start = record.get('answer_start', -1)
                if answer_start >= 0 and context:
                    try:
                        extracted_answer = context[answer_start:answer_start + len(answer)]
                        if extracted_answer.strip() != answer.strip():
                            invalid_answer_positions += 1
                    except IndexError:
                        invalid_answer_positions += 1
        
        # 计算质量指标
        if question_lengths:
            quality_metrics['avg_question_length'] = np.mean(question_lengths)
            quality_metrics['question_length_std'] = np.std(question_lengths)
            
            if quality_metrics['avg_question_length'] < 3:
                issues.append('问题平均长度过短')
            elif quality_metrics['avg_question_length'] > 100:
                issues.append('问题平均长度过长')
        
        if context_lengths:
            quality_metrics['avg_context_length'] = np.mean(context_lengths)
            quality_metrics['context_length_std'] = np.std(context_lengths)
            
            if quality_metrics['avg_context_length'] < 10:
                issues.append('上下文平均长度过短')
        
        if answer_lengths:
            quality_metrics['avg_answer_length'] = np.mean(answer_lengths)
            quality_metrics['answer_length_std'] = np.std(answer_lengths)
        
        # 重复率检查
        duplicate_ratio = duplicate_questions / len(data)
        quality_metrics['duplicate_question_ratio'] = duplicate_ratio
        
        if duplicate_ratio > 0.1:  # 超过10%重复
            issues.append(f'问题重复率过高: {duplicate_ratio:.2%}')
        
        # 答案位置准确性
        if invalid_answer_positions > 0:
            invalid_ratio = invalid_answer_positions / len(data)
            quality_metrics['invalid_answer_position_ratio'] = invalid_ratio
            
            if invalid_ratio > 0.05:  # 超过5%错误
                issues.append(f'答案位置错误率过高: {invalid_ratio:.2%}')
        
        return {
            'is_valid': len(issues) == 0,
            'metrics': quality_metrics,
            'issues': issues
        }
    
    def _generate_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成统计信息"""
        stats = {
            'total_records': len(data),
            'field_distribution': {},
            'text_statistics': {},
            'answer_statistics': {}
        }
        
        if not data:
            return stats
        
        # 字段分布
        all_fields = set()
        for record in data:
            all_fields.update(record.keys())
        
        for field in all_fields:
            count = sum(1 for record in data if record.get(field))
            stats['field_distribution'][field] = {
                'count': count,
                'percentage': count / len(data) * 100
            }
        
        # 文本统计
        questions = [record.get('question', '') for record in data if record.get('question')]
        contexts = [record.get('context', '') for record in data if record.get('context')]
        answers = [record.get('answer', '') for record in data if record.get('answer')]
        
        if questions:
            question_word_counts = [len(q.split()) for q in questions]
            stats['text_statistics']['question'] = {
                'count': len(questions),
                'avg_word_count': np.mean(question_word_counts),
                'min_word_count': np.min(question_word_counts),
                'max_word_count': np.max(question_word_counts),
                'std_word_count': np.std(question_word_counts)
            }
        
        if contexts:
            context_word_counts = [len(c.split()) for c in contexts]
            stats['text_statistics']['context'] = {
                'count': len(contexts),
                'avg_word_count': np.mean(context_word_counts),
                'min_word_count': np.min(context_word_counts),
                'max_word_count': np.max(context_word_counts),
                'std_word_count': np.std(context_word_counts)
            }
        
        if answers:
            answer_word_counts = [len(a.split()) for a in answers if a]
            empty_answers = sum(1 for a in answers if not a.strip())
            
            stats['answer_statistics'] = {
                'total_answers': len(answers),
                'non_empty_answers': len(answers) - empty_answers,
                'empty_answers': empty_answers,
                'empty_ratio': empty_answers / len(answers) if answers else 0
            }
            
            if answer_word_counts:
                stats['text_statistics']['answer'] = {
                    'count': len(answer_word_counts),
                    'avg_word_count': np.mean(answer_word_counts),
                    'min_word_count': np.min(answer_word_counts),
                    'max_word_count': np.max(answer_word_counts),
                    'std_word_count': np.std(answer_word_counts)
                }
        
        return stats
    
    def _get_overall_status(self, results: Dict[str, Any]) -> str:
        """获取整体状态"""
        # 检查关键步骤是否成功
        if not results['format_detection']['is_valid']:
            return 'failed'
        
        if not results['data_loading']['success']:
            return 'failed'
        
        if not results['field_validation']['is_valid']:
            return 'failed'
        
        # 检查质量问题数量
        quality_issues = len(results['quality_check'].get('issues', []))
        field_issues = len(results['field_validation'].get('issues', []))
        
        total_issues = quality_issues + field_issues
        
        if total_issues == 0:
            return 'excellent'
        elif total_issues <= 2:
            return 'good'
        elif total_issues <= 5:
            return 'warning'
        else:
            return 'poor'
    
    def generate_report(self, results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """生成验证报告"""
        report = []
        
        # 标题
        report.append("=" * 60)
        report.append("数据验证报告")
        report.append("=" * 60)
        report.append(f"文件: {results['file_path']}")
        report.append(f"总体状态: {results['overall_status'].upper()}")
        report.append("")
        
        # 格式检测
        report.append("1. 格式检测")
        report.append("-" * 30)
        format_info = results['format_detection']
        report.append(f"文件格式: {format_info.get('format', 'unknown')}")
        report.append(f"格式有效: {'是' if format_info.get('is_valid') else '否'}")
        if 'file_size' in format_info:
            report.append(f"文件大小: {format_info['file_size']:,} 字节")
        report.append("")
        
        # 数据加载
        report.append("2. 数据加载")
        report.append("-" * 30)
        loading_info = results['data_loading']
        report.append(f"加载成功: {'是' if loading_info.get('success') else '否'}")
        if loading_info.get('success'):
            report.append(f"记录数量: {loading_info.get('record_count', 0):,}")
        report.append("")
        
        # 字段验证
        report.append("3. 字段验证")
        report.append("-" * 30)
        field_info = results['field_validation']
        if 'available_fields' in field_info:
            report.append(f"可用字段: {', '.join(field_info['available_fields'])}")
        if 'missing_required' in field_info and field_info['missing_required']:
            report.append(f"缺失必需字段: {', '.join(field_info['missing_required'])}")
        
        if 'field_coverage' in field_info:
            report.append("字段覆盖率:")
            for field, coverage in field_info['field_coverage'].items():
                report.append(f"  {field}: {coverage:.2%}")
        report.append("")
        
        # 质量检查
        report.append("4. 质量检查")
        report.append("-" * 30)
        quality_info = results['quality_check']
        if 'metrics' in quality_info:
            metrics = quality_info['metrics']
            for key, value in metrics.items():
                report.append(f"{key}: {value:.4f}")
        
        if quality_info.get('issues'):
            report.append("质量问题:")
            for issue in quality_info['issues']:
                report.append(f"  - {issue}")
        report.append("")
        
        # 统计信息
        report.append("5. 统计信息")
        report.append("-" * 30)
        stats = results['statistics']
        report.append(f"总记录数: {stats.get('total_records', 0):,}")
        
        if 'text_statistics' in stats:
            text_stats = stats['text_statistics']
            for text_type, info in text_stats.items():
                report.append(f"{text_type}平均词数: {info.get('avg_word_count', 0):.1f}")
        
        report.append("")
        
        # 问题汇总
        if results.get('issues'):
            report.append("6. 问题汇总")
            report.append("-" * 30)
            for issue in results['issues']:
                report.append(f"  - {issue}")
            report.append("")
        
        report_text = "\n".join(report)
        
        # 保存报告
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report_text)
            self.logger.info(f"验证报告已保存到: {output_file}")
        
        return report_text
    
    def save_results(self, output_file: str):
        """保存验证结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, ensure_ascii=False, indent=2)
        self.logger.info(f"验证结果已保存到: {output_file}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='数据验证工具')
    parser.add_argument('--file', type=str, required=True, help='数据文件路径')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--output', type=str, help='输出报告路径')
    parser.add_argument('--log_level', type=str, default='INFO', help='日志级别')
    
    args = parser.parse_args()
    
    # 创建验证器
    validator = DataValidator(log_level=args.log_level)
    
    # 验证文件
    results = validator.validate_file(args.file, args.config)
    
    # 生成报告
    output_file = args.output or f"{Path(args.file).stem}_validation_report.txt"
    report = validator.generate_report(results, output_file)
    
    # 打印摘要
    print("\n" + "=" * 50)
    print("验证完成!")
    print(f"文件: {args.file}")
    print(f"状态: {results['overall_status'].upper()}")
    print(f"报告: {output_file}")
    
    if results.get('issues'):
        print(f"问题数量: {len(results['issues'])}")
    
    print("=" * 50)


if __name__ == "__main__":
    main()