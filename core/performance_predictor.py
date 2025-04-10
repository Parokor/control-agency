import os
import json
import time
import numpy as np
import pickle
from typing import Dict, Any, List, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class PerformancePredictor:
    """Machine learning model to predict execution times for tasks across different GPU platforms"""
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'task_type_id', 'input_size_mb', 'output_size_mb', 'memory_required_gb',
            'cpu_cores_required', 'gpu_memory_required_gb', 'is_batch_processing',
            'platform_id', 'gpu_type_id', 'cpu_type_id', 'network_speed_mbps'
        ]
        self.model_path = model_path or "models/performance_predictor.pkl"
        self.scaler_path = os.path.join(os.path.dirname(self.model_path), "performance_predictor_scaler.pkl")
        self.history_path = os.path.join(os.path.dirname(self.model_path), "performance_history.json")
        self.performance_history = []
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        # Load existing model if available
        self._load_model()
        self._load_history()
    
    def _load_model(self):
        """Load the trained model if it exists"""
        try:
            if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"Loaded performance prediction model from {self.model_path}")
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
        
        # Initialize new model if loading fails
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        self.scaler = StandardScaler()
        return False
    
    def _load_history(self):
        """Load performance history if it exists"""
        try:
            if os.path.exists(self.history_path):
                with open(self.history_path, 'r') as f:
                    self.performance_history = json.load(f)
                print(f"Loaded {len(self.performance_history)} performance history records")
        except Exception as e:
            print(f"Error loading history: {e}")
            self.performance_history = []
    
    def _save_model(self):
        """Save the trained model"""
        try:
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            with open(self.scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            print(f"Saved performance prediction model to {self.model_path}")
            return True
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
    
    def _save_history(self):
        """Save performance history"""
        try:
            with open(self.history_path, 'w') as f:
                json.dump(self.performance_history, f)
            return True
        except Exception as e:
            print(f"Error saving history: {e}")
            return False
    
    def _encode_task_type(self, task_type: str) -> int:
        """Encode task type to numeric value"""
        task_types = {
            "inference": 0,
            "training": 1,
            "data_processing": 2,
            "fine_tuning": 3,
            "embedding_generation": 4,
            "image_generation": 5,
            "video_processing": 6,
            "audio_processing": 7,
            "text_processing": 8,
            "optimization": 9
        }
        return task_types.get(task_type.lower(), -1)
    
    def _encode_platform(self, platform: str) -> int:
        """Encode platform to numeric value"""
        platforms = {
            "colab": 0,
            "kaggle": 1,
            "paperspace": 2,
            "huggingface": 3,
            "gitpod": 4,
            "intel_devcloud": 5,
            "github_codespaces": 6
        }
        return platforms.get(platform.lower(), -1)
    
    def _encode_gpu_type(self, gpu_type: str) -> int:
        """Encode GPU type to numeric value"""
        gpu_types = {
            "none": 0,
            "t4": 1,
            "p100": 2,
            "v100": 3,
            "a100": 4,
            "k80": 5,
            "p4": 6,
            "intel_xe": 7,
            "intel_iris_xe": 8,
            "intel_arc": 9,
            "nvidia_rtx": 10
        }
        return gpu_types.get(gpu_type.lower(), -1)
    
    def _encode_cpu_type(self, cpu_type: str) -> int:
        """Encode CPU type to numeric value"""
        cpu_types = {
            "standard": 0,
            "intel_xeon": 1,
            "amd_epyc": 2,
            "intel_core": 3,
            "amd_ryzen": 4,
            "arm": 5,
            "intel_platinum": 6
        }
        return cpu_types.get(cpu_type.lower(), -1)
    
    def _prepare_features(self, task_info: Dict[str, Any], platform_info: Dict[str, Any]) -> np.ndarray:
        """Prepare features for prediction"""
        # Extract and encode features
        features = {
            'task_type_id': self._encode_task_type(task_info.get('task_type', 'inference')),
            'input_size_mb': task_info.get('input_size_mb', 0),
            'output_size_mb': task_info.get('output_size_mb', 0),
            'memory_required_gb': task_info.get('memory_required_gb', 1),
            'cpu_cores_required': task_info.get('cpu_cores_required', 1),
            'gpu_memory_required_gb': task_info.get('gpu_memory_required_gb', 0),
            'is_batch_processing': 1 if task_info.get('is_batch_processing', False) else 0,
            'platform_id': self._encode_platform(platform_info.get('platform', 'colab')),
            'gpu_type_id': self._encode_gpu_type(platform_info.get('gpu_type', 'none')),
            'cpu_type_id': self._encode_cpu_type(platform_info.get('cpu_type', 'standard')),
            'network_speed_mbps': platform_info.get('network_speed_mbps', 100)
        }
        
        # Convert to numpy array
        X = np.array([[features[col] for col in self.feature_columns]])
        
        # Scale features if scaler is trained
        if self.scaler and hasattr(self.scaler, 'mean_'):
            X = self.scaler.transform(X)
        
        return X
    
    def predict_execution_time(self, task_info: Dict[str, Any], platform_info: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Predict execution time for a task on a specific platform"""
        # Check if model is trained
        if self.model is None or not hasattr(self.model, 'feature_importances_'):
            # Not enough data to make a prediction
            # Return a heuristic estimate
            return self._heuristic_estimate(task_info, platform_info)
        
        # Prepare features
        X = self._prepare_features(task_info, platform_info)
        
        # Make prediction
        predicted_time = self.model.predict(X)[0]
        
        # Add confidence interval
        prediction_info = {
            'predicted_time_seconds': max(0.1, predicted_time),  # Ensure positive prediction
            'confidence_interval': self._get_confidence_interval(X),
            'model_version': getattr(self.model, 'version', '1.0'),
            'features_used': self.feature_columns
        }
        
        return prediction_info['predicted_time_seconds'], prediction_info
    
    def _heuristic_estimate(self, task_info: Dict[str, Any], platform_info: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """Provide a heuristic estimate when model is not trained"""
        # Base execution time
        base_time = 10.0  # seconds
        
        # Adjust for task type
        task_type = task_info.get('task_type', 'inference').lower()
        task_multipliers = {
            'inference': 1.0,
            'training': 5.0,
            'data_processing': 2.0,
            'fine_tuning': 8.0,
            'embedding_generation': 1.5,
            'image_generation': 3.0,
            'video_processing': 7.0,
            'audio_processing': 2.5,
            'text_processing': 1.2,
            'optimization': 4.0
        }
        task_multiplier = task_multipliers.get(task_type, 1.0)
        
        # Adjust for input size
        input_size_mb = task_info.get('input_size_mb', 0)
        input_multiplier = 1.0 + (input_size_mb / 100.0)
        
        # Adjust for GPU availability
        gpu_type = platform_info.get('gpu_type', 'none').lower()
        gpu_multiplier = 0.5 if gpu_type != 'none' else 2.0
        
        # Calculate estimated time
        estimated_time = base_time * task_multiplier * input_multiplier * gpu_multiplier
        
        # Return estimate with confidence interval
        prediction_info = {
            'predicted_time_seconds': estimated_time,
            'confidence_interval': (estimated_time * 0.5, estimated_time * 2.0),
            'model_version': 'heuristic',
            'features_used': []
        }
        
        return estimated_time, prediction_info
    
    def _get_confidence_interval(self, X: np.ndarray) -> Tuple[float, float]:
        """Calculate confidence interval for prediction"""
        # For random forest, we can use the predictions of individual trees
        if hasattr(self.model, 'estimators_'):
            predictions = [tree.predict(X)[0] for tree in self.model.estimators_]
            lower_bound = np.percentile(predictions, 10)
            upper_bound = np.percentile(predictions, 90)
            return (max(0.1, lower_bound), upper_bound)
        else:
            # Fallback for other models
            predicted_time = self.model.predict(X)[0]
            return (max(0.1, predicted_time * 0.7), predicted_time * 1.3)
    
    def record_actual_performance(self, task_info: Dict[str, Any], platform_info: Dict[str, Any], 
                                execution_time: float, success: bool = True) -> bool:
        """Record actual performance for model improvement"""
        # Create performance record
        record = {
            'timestamp': time.time(),
            'task_info': task_info,
            'platform_info': platform_info,
            'execution_time': execution_time,
            'success': success
        }
        
        # Add to history
        self.performance_history.append(record)
        
        # Save history
        self._save_history()
        
        # Retrain model if we have enough data
        if len(self.performance_history) % 10 == 0:  # Retrain every 10 new records
            self.train_model()
        
        return True
    
    def train_model(self) -> Dict[str, Any]:
        """Train the performance prediction model"""
        if len(self.performance_history) < 10:
            return {'status': 'error', 'message': 'Not enough data to train model'}
        
        # Prepare training data
        X_data = []
        y_data = []
        
        for record in self.performance_history:
            # Skip failed executions
            if not record.get('success', True):
                continue
            
            task_info = record['task_info']
            platform_info = record['platform_info']
            execution_time = record['execution_time']
            
            # Extract features
            features = []
            for col in self.feature_columns:
                if col == 'task_type_id':
                    features.append(self._encode_task_type(task_info.get('task_type', 'inference')))
                elif col == 'platform_id':
                    features.append(self._encode_platform(platform_info.get('platform', 'colab')))
                elif col == 'gpu_type_id':
                    features.append(self._encode_gpu_type(platform_info.get('gpu_type', 'none')))
                elif col == 'cpu_type_id':
                    features.append(self._encode_cpu_type(platform_info.get('cpu_type', 'standard')))
                elif col == 'is_batch_processing':
                    features.append(1 if task_info.get('is_batch_processing', False) else 0)
                elif col in task_info:
                    features.append(task_info[col])
                elif col in platform_info:
                    features.append(platform_info[col])
                else:
                    features.append(0)  # Default value
            
            X_data.append(features)
            y_data.append(execution_time)
        
        # Convert to numpy arrays
        X = np.array(X_data)
        y = np.array(y_data)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate model
        y_pred = self.model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Save model
        self._save_model()
        
        # Return evaluation metrics
        return {
            'status': 'success',
            'metrics': {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            },
            'data_points': len(X_data),
            'feature_importance': dict(zip(self.feature_columns, self.model.feature_importances_))
        }
    
    def get_optimal_platform(self, task_info: Dict[str, Any], available_platforms: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Determine the optimal platform for a given task"""
        if not available_platforms:
            return {'status': 'error', 'message': 'No platforms available'}
        
        platform_predictions = []
        
        for platform in available_platforms:
            predicted_time, prediction_info = self.predict_execution_time(task_info, platform)
            
            platform_predictions.append({
                'platform': platform,
                'predicted_time': predicted_time,
                'prediction_info': prediction_info
            })
        
        # Sort by predicted time (ascending)
        platform_predictions.sort(key=lambda x: x['predicted_time'])
        
        # Return the best platform
        best_platform = platform_predictions[0]
        
        return {
            'status': 'success',
            'optimal_platform': best_platform['platform'],
            'predicted_time': best_platform['predicted_time'],
            'prediction_info': best_platform['prediction_info'],
            'all_predictions': platform_predictions
        }
    
    def analyze_platform_performance(self) -> Dict[str, Any]:
        """Analyze performance across different platforms"""
        if len(self.performance_history) < 10:
            return {'status': 'error', 'message': 'Not enough data for analysis'}
        
        # Group by platform
        platform_stats = {}
        
        for record in self.performance_history:
            platform = record['platform_info'].get('platform', 'unknown')
            if platform not in platform_stats:
                platform_stats[platform] = {
                    'count': 0,
                    'total_time': 0,
                    'success_count': 0,
                    'failure_count': 0,
                    'times': []
                }
            
            stats = platform_stats[platform]
            stats['count'] += 1
            stats['total_time'] += record['execution_time']
            stats['times'].append(record['execution_time'])
            
            if record.get('success', True):
                stats['success_count'] += 1
            else:
                stats['failure_count'] += 1
        
        # Calculate statistics
        for platform, stats in platform_stats.items():
            if stats['count'] > 0:
                stats['avg_time'] = stats['total_time'] / stats['count']
                stats['min_time'] = min(stats['times'])
                stats['max_time'] = max(stats['times'])
                stats['median_time'] = np.median(stats['times'])
                stats['success_rate'] = stats['success_count'] / stats['count']
                
                # Remove raw times to reduce size
                del stats['times']
        
        return {
            'status': 'success',
            'platform_stats': platform_stats,
            'total_records': len(self.performance_history)
        }
    
    def get_performance_trends(self, days: int = 30) -> Dict[str, Any]:
        """Get performance trends over time"""
        if len(self.performance_history) < 10:
            return {'status': 'error', 'message': 'Not enough data for trend analysis'}
        
        # Calculate cutoff time
        cutoff_time = time.time() - (days * 24 * 60 * 60)
        
        # Filter records
        recent_records = [r for r in self.performance_history if r['timestamp'] >= cutoff_time]
        
        if len(recent_records) < 5:
            return {'status': 'error', 'message': f'Not enough data in the last {days} days'}
        
        # Group by day
        daily_stats = {}
        
        for record in recent_records:
            # Convert timestamp to day
            day = time.strftime('%Y-%m-%d', time.localtime(record['timestamp']))
            
            if day not in daily_stats:
                daily_stats[day] = {
                    'count': 0,
                    'total_time': 0,
                    'success_count': 0
                }
            
            stats = daily_stats[day]
            stats['count'] += 1
            stats['total_time'] += record['execution_time']
            
            if record.get('success', True):
                stats['success_count'] += 1
        
        # Calculate daily averages
        trend_data = []
        
        for day, stats in sorted(daily_stats.items()):
            if stats['count'] > 0:
                trend_data.append({
                    'date': day,
                    'avg_execution_time': stats['total_time'] / stats['count'],
                    'task_count': stats['count'],
                    'success_rate': stats['success_count'] / stats['count']
                })
        
        return {
            'status': 'success',
            'trend_data': trend_data,
            'days_analyzed': days,
            'total_records': len(recent_records)
        }
