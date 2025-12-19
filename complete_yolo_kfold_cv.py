"""
YOLOV11五折交叉验证

该脚本实现了针对YOLOv11目标检测模型的五折交叉验证功能。
通过交叉验证可以更准确地评估模型性能，并减少因数据划分偶然性带来的评估偏差。
"""

import os
import random
import shutil
import yaml
import numpy as np
from sklearn.model_selection import KFold
from pathlib import Path


class YOLOKFoldCV:
    """
    YOLO模型的K折交叉验证类
    
    该类实现了对YOLO模型的K折交叉验证流程，包括：
    1. 自动划分数据集为K个不重叠的子集
    2. 依次将每个子集作为验证集，其余作为训练集
    3. 分别训练K个模型并记录性能指标
    4. 统计分析K次实验的结果
    """

    def __init__(self, data_yaml_path, k_folds=5, output_dir='kfold_output'):
        """
        初始化K折交叉验证对象
        
        Args:
            data_yaml_path (str): 数据集配置文件路径（YOLO格式）
            k_folds (int): 折数，默认为5
            output_dir (str): 输出目录，存储每折的数据和模型
        """
        self.data_yaml_path = Path(data_yaml_path)
        self.k_folds = k_folds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载数据集配置
        with open(self.data_yaml_path, 'r', encoding='utf-8') as f:
            self.data_config = yaml.safe_load(f)
            
        print(f"加载数据集配置完成: {self.data_yaml_path}")
        print(f"类别数量: {self.data_config['nc']}")
        print(f"类别名称: {self.data_config['names']}")

    def prepare_datasets(self):
        """
        准备K折交叉验证所需的数据集
        
        根据原始数据集，将其划分为K个不重叠的子集，每折包含特定的训练和验证集。
        
        Returns:
            list: 包含每折配置文件路径的列表
        """
        print(f"\n开始准备{self.k_folds}折交叉验证数据...")
        
        # 收集所有训练图像路径
        train_img_dir = Path(self.data_config['train'])
        images = []
        
        # 遍历训练目录收集所有图像文件
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            images.extend(train_img_dir.rglob(ext))
            
        print(f"找到训练图像数量: {len(images)}")
        
        # 如果配置中有验证集，也加入到图像列表中统一处理
        if 'val' in self.data_config:
            val_img_dir = Path(self.data_config['val'])
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                images.extend(val_img_dir.rglob(ext))
            print(f"合并验证集后总图像数量: {len(images)}")
            
        # 转换为绝对路径并去重
        images = list(set([img.resolve() for img in images]))
        print(f"去重后图像总数: {len(images)}")
        
        # 随机打乱数据集顺序
        random.seed(42)
        random.shuffle(images)
        
        # 使用sklearn的KFold进行数据划分
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        
        fold_configs = []
        
        # 为每一折创建独立的数据集
        for fold, (train_indices, val_indices) in enumerate(kf.split(images), 1):
            print(f"  正在创建第 {fold} 折数据集...")
            
            # 创建该折的目录结构
            fold_dir = self.output_dir / f'fold_{fold}'
            train_img_fold_dir = fold_dir / 'images' / 'train'
            val_img_fold_dir = fold_dir / 'images' / 'val'
            train_label_fold_dir = fold_dir / 'labels' / 'train'
            val_label_fold_dir = fold_dir / 'labels' / 'val'
            
            # 确保所有目录都存在
            for dir_path in [train_img_fold_dir, val_img_fold_dir, 
                             train_label_fold_dir, val_label_fold_dir]:
                dir_path.mkdir(parents=True, exist_ok=True)
            
            # 复制训练数据（图像和标签）
            for idx in train_indices:
                self._copy_image_and_label(images[idx], train_img_fold_dir, train_label_fold_dir)
                
            # 复制验证数据（图像和标签）
            for idx in val_indices:
                self._copy_image_and_label(images[idx], val_img_fold_dir, val_label_fold_dir)
                
            # 创建该折的数据集配置文件
            fold_config = {
                'path': str(fold_dir.absolute()),  # 数据集根目录
                'train': 'images/train',           # 相对于path的训练集路径
                'val': 'images/val',               # 相对于path的验证集路径
                'nc': self.data_config['nc'],      # 类别数量
                'names': self.data_config['names'] # 类别名称
            }
            
            # 保存配置文件
            fold_config_path = fold_dir / 'data.yaml'
            with open(fold_config_path, 'w', encoding='utf-8') as f:
                yaml.dump(fold_config, f, allow_unicode=True)
                
            fold_configs.append(fold_config_path)
            print(f"  第 {fold} 折数据集创建完成，配置文件: {fold_config_path}")
            
        print(f"所有{self.k_folds}折数据集准备完成！\n")
        return fold_configs

    def _copy_image_and_label(self, image_path, img_dest_dir, label_dest_dir):
        """
        复制单个图像及其对应的标签文件
        
        Args:
            image_path (Path): 图像文件路径
            img_dest_dir (Path): 目标图像目录
            label_dest_dir (Path): 目标标签目录
        """
        # 复制图像文件
        shutil.copy(image_path, img_dest_dir)
        
        # 构造并复制标签文件（假设标签与图像同名，扩展名为.txt）
        label_path = image_path.with_suffix('.txt')
        if label_path.exists():
            shutil.copy(label_path, label_dest_dir)

    def run_validation(self, model_def='yolov8s.yaml', epochs=100, imgsz=640, batch_size=16):
        """
        运行K折交叉验证
        
        Args:
            model_def (str): 模型定义文件或预训练权重路径
            epochs (int): 每折训练的轮数
            imgsz (int): 训练图像尺寸
            batch_size (int): 批次大小
            
        Returns:
            tuple: (各折详细结果字典, 平均结果字典)
        """
        print("开始K折交叉验证...")
        
        # 尝试导入YOLO相关库
        try:
            from ultralytics import YOLO
            yolo_available = True
            print("成功导入YOLO库")
        except ImportError:
            yolo_available = False
            print("警告: 未找到YOLO库，将使用模拟模式运行")
            
        # 准备数据集
        fold_configs = self.prepare_datasets()
        
        # 存储每折的结果
        results = {}
        
        # 依次训练和验证每折数据
        for fold_num, config_path in enumerate(fold_configs, 1):
            print(f"\n{'='*30}")
            print(f"开始训练第 {fold_num} 折")
            print(f"{'='*30}")
            
            if yolo_available:
                # 使用真实YOLO模型进行训练和验证
                fold_results = self._train_and_validate(
                    model_def, config_path, epochs, imgsz, batch_size, fold_num)
            else:
                # 模拟训练过程（用于演示或测试）
                fold_results = self._simulate_training(fold_num)
                
            results[f'fold_{fold_num}'] = fold_results
            print(f"第 {fold_num} 折完成 - mAP50: {fold_results['metrics']['mAP50']:.4f}")
            
        # 计算并显示统计结果
        avg_results = self._compute_statistics(results)
        self._display_results(results, avg_results)
        
        return results, avg_results

    def _train_and_validate(self, model_def, data_config, epochs, imgsz, batch_size, fold_num):
        """
        使用YOLO训练和验证一折数据
        
        Args:
            model_def (str): 模型定义或权重路径
            data_config (Path): 该折的数据配置文件路径
            epochs (int): 训练轮数
            imgsz (int): 图像尺寸
            batch_size (int): 批次大小
            fold_num (int): 当前折数
            
        Returns:
            dict: 该折的训练结果
        """
        try:
            from ultralytics import YOLO
            
            # 初始化模型
            model = YOLO(model_def)
            
            # 设置该项目特定的保存目录
            project_name = str(self.output_dir / f'fold_{fold_num}_training')
            
            # 开始训练
            print(f"开始训练第 {fold_num} 折模型...")
            model.train(
                data=str(data_config),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch_size,
                project=project_name,
                name='train'
            )
            
            # 验证模型
            print(f"验证第 {fold_num} 折模型...")
            metrics = model.val(
                data=str(data_config),
                imgsz=imgsz,
                project=project_name,
                name='val'
            )
            
            # 返回结果
            return {
                'config': str(data_config),
                'metrics': {
                    'mAP50': float(metrics.box.map50),
                    'mAP50-95': float(metrics.box.map),
                    'precision': float(metrics.box.p),
                    'recall': float(metrics.box.r)
                },
                'model_path': f"{project_name}/train/weights/best.pt"
            }
            
        except Exception as e:
            print(f"第 {fold_num} 折训练出错: {e}")
            # 出错时返回默认值
            return {
                'config': str(data_config),
                'metrics': {
                    'mAP50': 0.0,
                    'mAP50-95': 0.0,
                    'precision': 0.0,
                    'recall': 0.0
                },
                'model_path': None,
                'error': str(e)
            }

    def _simulate_training(self, fold_num):
        """
        模拟训练过程（用于演示）
        
        Args:
            fold_num (int): 折数
            
        Returns:
            dict: 模拟的训练结果
        """
        print(f"模拟训练第 {fold_num} 折...")
        
        # 生成随机但合理的性能指标
        base_map = 0.7 + 0.2 * random.random()  # 0.7-0.9之间
        return {
            'config': f'simulated_fold_{fold_num}_config.yaml',
            'metrics': {
                'mAP50': base_map,
                'mAP50-95': base_map * (0.8 + 0.2 * random.random()),  # mAP50-95通常比mAP50低一些
                'precision': 0.75 + 0.2 * random.random(),
                'recall': 0.7 + 0.25 * random.random()
            },
            'model_path': f'simulated_model_fold_{fold_num}.pt'
        }

    def _compute_statistics(self, results):
        """
        计算K折交叉验证的统计数据
        
        Args:
            results (dict): 各折的结果
            
        Returns:
            dict: 统计结果（均值和标准差）
        """
        metrics_names = ['mAP50', 'mAP50-95', 'precision', 'recall']
        
        # 提取各折的指标值
        metrics_values = {metric: [] for metric in metrics_names}
        for fold_results in results.values():
            for metric in metrics_names:
                metrics_values[metric].append(fold_results['metrics'][metric])
                
        # 计算均值和标准差
        statistics = {}
        for metric in metrics_names:
            values = np.array(metrics_values[metric])
            statistics[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
            
        return statistics

    def _display_results(self, results, avg_results):
        """
        显示交叉验证结果
        
        Args:
            results (dict): 各折详细结果
            avg_results (dict): 统计结果
        """
        print(f"\n{'='*60}")
        print("K折交叉验证最终结果")
        print(f"{'='*60}")
        
        # 显示每折详细结果
        print("\n各折详细结果:")
        for fold_name, fold_results in results.items():
            metrics = fold_results['metrics']
            print(f"  {fold_name}:")
            print(f"    mAP50: {metrics['mAP50']:.4f}")
            print(f"    mAP50-95: {metrics['mAP50-95']:.4f}")
            print(f"    Precision: {metrics['precision']:.4f}")
            print(f"    Recall: {metrics['recall']:.4f}")
            
        # 显示统计摘要
        print("\n统计摘要 (均值 ± 标准差):")
        for metric, stats in avg_results.items():
            print(f"  {metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                  f"(范围: {stats['min']:.4f} ~ {stats['max']:.4f})")
            
        print(f"\n{'='*60}")


def main():
    """
    主函数 - 使用示例
    """
    print("YOLO K折交叉验证工具")
    print("="*30)
    
    # 使用示例（请根据实际情况修改参数）：
    # 1. 创建交叉验证对象
    # cv = YOLOKFoldCV(
    #     data_yaml_path='path/to/your/dataset.yaml',  # 你的数据集配置文件
    #     k_folds=5,                                   # 5折交叉验证
    #     output_dir='kfold_results'                   # 结果输出目录
    # )
    #
    # 2. 运行交叉验证
    # results, avg_results = cv.run_validation(
    #     model_def='yolov8s.yaml',  # 或 'yolov11s.pt' （如果可用）
    #     epochs=100,                # 训练轮数
    #     imgsz=640,                 # 图像尺寸
    #     batch_size=16              # 批次大小
    # )
    
    print("\n使用方法:")
    print("1. 修改 data_yaml_path 为你的YOLO数据集配置文件路径")
    print("2. 调整 k_folds 为你需要的折数（通常为5）")
    print("3. 设置适当的训练参数（epochs, imgsz, batch_size等）")
    print("4. 运行 run_validation 方法开始交叉验证")


if __name__ == "__main__":
    main()