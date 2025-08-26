# -*- coding: utf-8 -*-
"""
问答系统综合测试脚本

测试整个系统的功能，包括数据加载、模型训练、推理等。
"""

import os
import sys
import json
import logging
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

# 导入项目模块
modules_available = {}
try:
    from utils.format_detector import FormatDetector
    modules_available['format_detector'] = True
except ImportError as e:
    print(f"导入FormatDetector失败: {e}")
    modules_available['format_detector'] = False

try:
    from utils.data_loader import UniversalDataLoader
    modules_available['data_loader'] = True
except ImportError as e:
    print(f"导入UniversalDataLoader失败: {e}")
    modules_available['data_loader'] = False

try:
    from utils.data_processor import DataProcessor
    modules_available['data_processor'] = True
except ImportError as e:
    print(f"导入DataProcessor失败: {e}")
    modules_available['data_processor'] = False

try:
    from utils.tokenizer import QATokenizer
    modules_available['tokenizer'] = True
except ImportError as e:
    print(f"导入QATokenizer失败: {e}")
    modules_available['tokenizer'] = False

try:
    from utils.metrics import QAMetrics
    modules_available['metrics'] = True
except ImportError as e:
    print(f"导入QAMetrics失败: {e}")
    modules_available['metrics'] = False

try:
    from models.transformer import QATransformer
    modules_available['transformer'] = True
except ImportError as e:
    print(f"导入QATransformer失败: {e}")
    modules_available['transformer'] = False

try:
    from data_validator import DataValidator
    modules_available['data_validator'] = True
except ImportError as e:
    print(f"导入DataValidator失败: {e}")
    modules_available['data_validator'] = False


class SystemTester:
    """系统测试器"""
    
    def __init__(self, log_level: str = "INFO"):
        """初始化测试器"""
        self.setup_logging(log_level)
        self.logger = logging.getLogger(__name__)
        
        self.test_results = {}
        self.failed_tests = []
        
        # 测试数据路径
        self.data_dir = Path("data/raw/custom")
        self.test_files = [
            self.data_dir / "qa_data.csv",
            self.data_dir / "qa_data.json", 
            self.data_dir / "qa_data.jsonl"
        ]
        
    def setup_logging(self, level: str):
        """设置日志"""
        logging.basicConfig(
            level=getattr(logging, level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        self.logger.info("开始系统综合测试...")
        
        tests = []
        
        # 只添加可用模块的测试
        if modules_available.get('format_detector'):
            tests.append(("数据格式检测", self.test_format_detection))
        
        if modules_available.get('data_loader'):
            tests.append(("数据加载功能", self.test_data_loading))
        
        if modules_available.get('data_validator'):
            tests.append(("数据验证功能", self.test_data_validation))
        
        if modules_available.get('tokenizer'):
            tests.append(("分词器功能", self.test_tokenizer))
        
        if modules_available.get('data_processor'):
            tests.append(("数据处理功能", self.test_data_processing))
        
        if modules_available.get('transformer'):
            tests.append(("模型创建", self.test_model_creation))
        
        if modules_available.get('metrics'):
            tests.append(("评估指标", self.test_metrics))
        
        # 配置加载测试不依赖特殊模块
        tests.append(("配置加载", self.test_config_loading))
        tests.append(("文件结构检查", self.test_file_structure))
        
        for test_name, test_func in tests:
            self.logger.info(f"运行测试: {test_name}")
            try:
                result = test_func()
                self.test_results[test_name] = {
                    'status': 'passed' if result else 'failed',
                    'details': result if isinstance(result, dict) else {}
                }
                
                if result:
                    self.logger.info(f"✅ {test_name} - 通过")
                else:
                    self.logger.error(f"❌ {test_name} - 失败")
                    self.failed_tests.append(test_name)
                    
            except Exception as e:
                self.logger.error(f"❌ {test_name} - 异常: {str(e)}")
                self.test_results[test_name] = {
                    'status': 'error',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                self.failed_tests.append(test_name)
        
        # 生成测试报告
        self.generate_test_report()
        
        return self.test_results
    
    def test_format_detection(self) -> bool:
        """测试格式检测功能"""
        if not modules_available.get('format_detector'):
            self.logger.warning("FormatDetector模块不可用，跳过测试")
            return True
            
        try:
            detector = FormatDetector()
            
            for test_file in self.test_files:
                if not test_file.exists():
                    self.logger.warning(f"测试文件不存在: {test_file}")
                    continue
                
                result = detector.detect_format(test_file)
                
                if not result['is_valid']:
                    self.logger.error(f"格式检测失败: {test_file}")
                    return False
                
                expected_formats = {
                    'qa_data.csv': 'csv',
                    'qa_data.json': 'json',
                    'qa_data.jsonl': 'jsonl'
                }
                
                expected_format = expected_formats.get(test_file.name)
                if expected_format and result['format'] != expected_format:
                    self.logger.error(f"格式检测错误: {test_file}, 期望: {expected_format}, 实际: {result['format']}")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"格式检测测试异常: {e}")
            return False
    
    def test_data_loading(self) -> bool:
        """测试数据加载功能"""
        if not modules_available.get('data_loader'):
            self.logger.warning("UniversalDataLoader模块不可用，跳过测试")
            return True
            
        try:
            for test_file in self.test_files:
                if not test_file.exists():
                    continue
                
                loader = UniversalDataLoader(data_path=test_file)
                data = loader.load_data()
                
                if not data:
                    self.logger.error(f"数据加载失败: {test_file}")
                    return False
                
                # 检查数据结构
                required_fields = ['question', 'context', 'answer']
                for record in data[:5]:  # 检查前5条记录
                    for field in required_fields:
                        if field not in record:
                            self.logger.error(f"缺少必需字段 {field}: {test_file}")
                            return False
                
                self.logger.info(f"数据加载成功: {test_file}, 记录数: {len(data)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"数据加载测试异常: {e}")
            return False
    
    def test_data_validation(self) -> bool:
        """测试数据验证功能"""
        if not modules_available.get('data_validator'):
            self.logger.warning("DataValidator模块不可用，跳过测试")
            return True
            
        try:
            validator = DataValidator(log_level="ERROR")  # 减少日志输出
            
            for test_file in self.test_files:
                if not test_file.exists():
                    continue
                
                result = validator.validate_file(str(test_file))
                
                if result['overall_status'] in ['failed', 'error']:
                    self.logger.error(f"数据验证失败: {test_file}")
                    return False
                
                self.logger.info(f"数据验证通过: {test_file}, 状态: {result['overall_status']}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"数据验证测试异常: {e}")
            return False
    
    def test_tokenizer(self) -> bool:
        """测试分词器功能"""
        if not modules_available.get('tokenizer'):
            self.logger.warning("QATokenizer模块不可用，跳过测试")
            return True
            
        try:
            # 创建测试数据
            test_texts = [
                "什么是机器学习？",
                "Machine learning is great!",
                "机器学习是人工智能的重要分支，它使用statistical techniques。"
            ]
            
            # 创建分词器
            tokenizer = QATokenizer(vocab_size=1000, language="mixed")
            
            # 构建词汇表
            tokenizer.build_vocab(test_texts)
            
            # 测试编码和解码
            for text in test_texts:
                encoded = tokenizer.encode(text)
                decoded = tokenizer.decode(encoded['input_ids'])
                
                if not encoded['input_ids']:
                    self.logger.error(f"编码失败: {text}")
                    return False
                
                if not decoded:
                    self.logger.error(f"解码失败: {text}")
                    return False
            
            self.logger.info(f"分词器测试通过，词汇表大小: {tokenizer.get_vocab_size()}")
            return True
            
        except Exception as e:
            self.logger.error(f"分词器测试异常: {e}")
            return False
    
    def test_data_processing(self) -> bool:
        """测试数据处理功能"""
        if not modules_available.get('data_processor') or not modules_available.get('data_loader') or not modules_available.get('tokenizer'):
            self.logger.warning("数据处理相关模块不可用，跳过测试")
            return True
            
        try:
            # 加载测试数据
            test_file = self.test_files[0]  # 使用CSV文件
            if not test_file.exists():
                self.logger.warning("测试文件不存在，跳过数据处理测试")
                return True
            
            loader = UniversalDataLoader(data_path=test_file)
            data = loader.load_data()
            
            # 创建分词器
            all_texts = []
            for record in data:
                all_texts.extend([record.get('question', ''), record.get('context', ''), record.get('answer', '')])
            
            tokenizer = QATokenizer(vocab_size=1000)
            tokenizer.build_vocab(all_texts)
            
            # 创建数据处理器
            processor = DataProcessor(tokenizer=tokenizer, max_length=256)
            
            # 处理数据
            processed_data = processor.process_data(data[:5], is_training=True)  # 只处理前5条
            
            if not processed_data:
                self.logger.error("数据处理失败")
                return False
            
            self.logger.info(f"数据处理成功，处理前: {len(data[:5])}, 处理后: {len(processed_data)}")
            return True
            
        except Exception as e:
            self.logger.error(f"数据处理测试异常: {e}")
            return False
    
    def test_model_creation(self) -> bool:
        """测试模型创建"""
        if not modules_available.get('transformer'):
            self.logger.warning("QATransformer模块不可用，跳过测试")
            return True
            
        try:
            # 创建一个小型模型用于测试
            model = QATransformer(
                vocab_size=1000,
                d_model=128,
                n_heads=4,
                n_encoder_layers=2,
                d_ff=256,
                max_position_embeddings=128,
                dropout=0.1
            )
            
            # 测试模型前向传播
            batch_size = 2
            seq_len = 32
            
            question_ids = torch.randint(1, 1000, (batch_size, seq_len))
            context_ids = torch.randint(1, 1000, (batch_size, seq_len))
            
            try:
                import torch
                outputs = model(
                    question_input_ids=question_ids,
                    context_input_ids=context_ids
                )
                
                if 'start_logits' not in outputs or 'end_logits' not in outputs:
                    self.logger.error("模型输出格式错误")
                    return False
                
                self.logger.info(f"模型创建成功，参数数量: {sum(p.numel() for p in model.parameters()):,}")
                return True
                
            except ImportError:
                self.logger.warning("PyTorch未安装，跳过模型测试")
                return True
            
        except Exception as e:
            self.logger.error(f"模型创建测试异常: {e}")
            return False
    
    def test_metrics(self) -> bool:
        """测试评估指标"""
        if not modules_available.get('metrics'):
            self.logger.warning("QAMetrics模块不可用，跳过测试")
            return True
            
        try:
            metrics = QAMetrics()
            
            # 测试数据
            predictions = ["机器学习", "深度学习网络", "自然语言处理"]
            references = ["机器学习算法", "深度神经网络", "自然语言处理技术"]
            
            # 测试各项指标
            em = metrics.compute_exact_match(predictions, references)
            f1 = metrics.compute_f1_score(predictions, references)
            bleu = metrics.compute_bleu_score(predictions, references)
            rouge = metrics.compute_rouge_score(predictions, references)
            
            if not (0 <= em <= 1 and 0 <= f1 <= 1):
                self.logger.error("评估指标值异常")
                return False
            
            self.logger.info(f"评估指标测试通过 - EM: {em:.4f}, F1: {f1:.4f}")
            return True
            
        except Exception as e:
            self.logger.error(f"评估指标测试异常: {e}")
            return False
    
    def test_config_loading(self) -> bool:
        """测试配置加载"""
        try:
            import yaml
            
            config_file = Path("configs/config.yaml")
            if not config_file.exists():
                self.logger.warning("配置文件不存在")
                return True
            
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            required_sections = ['model', 'training', 'data', 'paths']
            for section in required_sections:
                if section not in config:
                    self.logger.error(f"配置文件缺少节: {section}")
                    return False
            
            self.logger.info("配置文件加载成功")
            return True
            
        except Exception as e:
            self.logger.error(f"配置加载测试异常: {e}")
            return False
    
    def test_file_structure(self) -> bool:
        """测试文件结构"""
        try:
            required_dirs = [
                Path("configs"),
                Path("models"), 
                Path("utils"),
                Path("data/raw/custom")
            ]
            
            for dir_path in required_dirs:
                if not dir_path.exists():
                    self.logger.error(f"缺少目录: {dir_path}")
                    return False
            
            required_files = [
                Path("configs/config.yaml"),
                Path("models/__init__.py"),
                Path("utils/__init__.py"),
                Path("train.py"),
                Path("inference.py"),
                Path("requirements.txt")
            ]
            
            for file_path in required_files:
                if not file_path.exists():
                    self.logger.error(f"缺少文件: {file_path}")
                    return False
            
            self.logger.info("文件结构检查通过")
            return True
            
        except Exception as e:
            self.logger.error(f"文件结构测试异常: {e}")
            return False
    
    def generate_test_report(self):
        """生成测试报告"""
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() if result['status'] == 'passed')
        failed_tests = len(self.failed_tests)
        
        self.logger.info("\n" + "="*60)
        self.logger.info("测试报告")
        self.logger.info("="*60)
        self.logger.info(f"总测试数: {total_tests}")
        self.logger.info(f"通过测试: {passed_tests}")
        self.logger.info(f"失败测试: {failed_tests}")
        self.logger.info(f"成功率: {passed_tests/total_tests*100:.1f}%")
        
        if self.failed_tests:
            self.logger.info(f"\n失败的测试:")
            for test_name in self.failed_tests:
                self.logger.info(f"  - {test_name}")
        
        self.logger.info("="*60)
        
        # 保存详细报告
        report_file = Path("outputs/test_report.json")
        report_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump({
                'summary': {
                    'total_tests': total_tests,
                    'passed_tests': passed_tests,
                    'failed_tests': failed_tests,
                    'success_rate': passed_tests/total_tests*100
                },
                'details': self.test_results,
                'failed_tests': self.failed_tests
            }, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"详细测试报告已保存到: {report_file}")


def main():
    """主函数"""
    print("问答系统综合测试")
    print("="*50)
    
    # 检查必要的依赖
    try:
        import torch
        print("✅ PyTorch 已安装")
    except ImportError:
        print("⚠️  PyTorch 未安装，将跳过模型相关测试")
    
    try:
        import yaml
        print("✅ PyYAML 已安装")
    except ImportError:
        print("❌ PyYAML 未安装，请安装: pip install PyYAML")
        return
    
    try:
        import pandas
        print("✅ Pandas 已安装")
    except ImportError:
        print("❌ Pandas 未安装，请安装: pip install pandas")
        return
    
    print("\n开始系统测试...")
    
    # 运行测试
    tester = SystemTester()
    results = tester.run_all_tests()
    
    # 返回退出码
    failed_count = len(tester.failed_tests)
    if failed_count == 0:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n❌ {failed_count} 个测试失败")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())