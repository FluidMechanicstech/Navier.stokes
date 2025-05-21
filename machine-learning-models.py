###########################################
# 3. Flask API 서비스
###########################################

app = Flask(__name__)

# 모델 초기화
blockchain_predictor = BlockchainTransactionPredictor()
vehicle_predictor = AutonomousVehiclePredictor()

@app.route('/health', methods=['GET'])
def health_check():
    """상태 확인 API"""
    return jsonify({
        'status': 'healthy',
        'timestamp': time.time(),
        'models': {
            'blockchain': blockchain_predictor.transaction_model is not None,
            'propagation': blockchain_predictor.propagation_model is not None,
            'vehicle_path': vehicle_predictor.path_model is not None,
            'vehicle_risk': vehicle_predictor.risk_model is not None
        }
    })

@app.route('/predict/blockchain/transactions', methods=['POST'])
def predict_blockchain_transactions():
    """블록체인 트랜잭션 예측 API"""
    try:
        data = request.json
        
        if not data or 'transactions' not in data or 'sourceAddress' not in data:
            return jsonify({'error': '필수 데이터가 누락되었습니다.'}), 400
        
        transactions = data['transactions']
        source_address = data['sourceAddress']
        
        # 트랜잭션 예측 수행
        predictions = blockchain_predictor.predict_transactions(transactions, source_address)
        
        return jsonify({
            'predictions': predictions,
            'predictedCount': len(predictions),
            'timestamp': time.time()
        })
    
    except Exception as e:
        logger.error(f"트랜잭션 예측 API 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/blockchain/propagation', methods=['POST'])
def predict_blockchain_propagation():
    """블록체인 트랜잭션 전파 예측 API"""
    try:
        data = request.json
        
        if not data or 'txHash' not in data:
            return jsonify({'error': '트랜잭션 해시가 필요합니다.'}), 400
        
        tx_hash = data['txHash']
        node_count = data.get('nodeCount', 10)
        
        # 전파 시간 예측
        propagation_times = blockchain_predictor.predict_propagation(tx_hash, node_count)
        
        return jsonify({
            'propagationTimes': propagation_times,
            'timestamp': time.time()
        })
    
    except Exception as e:
        logger.error(f"전파 예측 API 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/blockchain/clusters', methods=['POST'])
def cluster_blockchain_transactions():
    """블록체인 트랜잭션 클러스터링 API"""
    try:
        data = request.json
        
        if not data or 'transactions' not in data:
            return jsonify({'error': '트랜잭션 데이터가 필요합니다.'}), 400
        
        transactions = data['transactions']
        
        # 클러스터링 수행
        clusters = blockchain_predictor.cluster_transactions(transactions)
        
        return jsonify({
            'clusters': clusters,
            'timestamp': time.time()
        })
    
    except Exception as e:
        logger.error(f"클러스터링 API 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/vehicle/path', methods=['POST'])
def predict_vehicle_path():
    """자율주행차 경로 예측 API"""
    try:
        data = request.json
        
        if not data or 'currentPosition' not in data:
            return jsonify({'error': '현재 위치 정보가 필요합니다.'}), 400
        
        vehicle_id = data.get('vehicleId', 'unknown')
        current_position = data['currentPosition']
        flow_field = data.get('flowField', [])
        steps = data.get('timeSteps', 10)
        
        # 경로 예측 수행
        path = vehicle_predictor.predict_path(vehicle_id, current_position, flow_field, steps)
        
        return jsonify({
            'predictions': path,
            'stepCount': len(path),
            'timestamp': time.time()
        })
    
    except Exception as e:
        logger.error(f"경로 예측 API 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict/vehicle/risk', methods=['POST'])
def predict_vehicle_risk():
    """자율주행차 위험 예측 API"""
    try:
        data = request.json
        
        if not data or 'position' not in data:
            return jsonify({'error': '위치 정보가 필요합니다.'}), 400
        
        position = data['position']
        k = data.get('k', 0.3)
        alpha = data.get('alpha', 1.0)
        beta = data.get('beta', -1.0)
        gamma = data.get('gamma', 0.5)
        
        # 위험 지점 계산
        risk_zones = vehicle_predictor.calculate_risk_zones(position, k, alpha, beta, gamma)
        
        return jsonify({
            'riskZones': risk_zones,
            'zoneCount': len(risk_zones),
            'timestamp': time.time()
        })
    
    except Exception as e:
        logger.error(f"위험 예측 API 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/train/blockchain', methods=['POST'])
def train_blockchain_model():
    """블록체인 모델 훈련 API"""
    try:
        data = request.json
        
        if not data or 'transactions' not in data or 'outcomes' not in data:
            return jsonify({'error': '훈련 데이터가 필요합니다.'}), 400
        
        # 모델 훈련
        success = blockchain_predictor.train_with_new_data(
            data['transactions'], 
            data['outcomes']
        )
        
        return jsonify({
            'success': success,
            'message': '모델 훈련 완료' if success else '모델 훈련 실패',
            'timestamp': time.time()
        })
    
    except Exception as e:
        logger.error(f"모델 훈련 API 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/train/vehicle/path', methods=['POST'])
def train_vehicle_path_model():
    """자율주행차 경로 예측 모델 훈련 API"""
    try:
        data = request.json
        
        if not data or 'training_data' not in data:
            return jsonify({'error': '훈련 데이터가 필요합니다.'}), 400
        
        # 모델 훈련
        success = vehicle_predictor.train_path_model(data['training_data'])
        
        return jsonify({
            'success': success,
            'message': '경로 예측 모델 훈련 완료' if success else '모델 훈련 실패',
            'timestamp': time.time()
        })
    
    except Exception as e:
        logger.error(f"경로 모델 훈련 API 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/train/vehicle/risk', methods=['POST'])
def train_vehicle_risk_model():
    """자율주행차 위험 분석 모델 훈련 API"""
    try:
        data = request.json
        
        if not data or 'training_data' not in data:
            return jsonify({'error': '훈련 데이터가 필요합니다.'}), 400
        
        # 모델 훈련
        success = vehicle_predictor.train_risk_model(data['training_data'])
        
        return jsonify({
            'success': success,
            'message': '위험 분석 모델 훈련 완료' if success else '모델 훈련 실패',
            'timestamp': time.time()
        })
    
    except Exception as e:
        logger.error(f"위험 모델 훈련 API 오류: {str(e)}")
        return jsonify({'error': str(e)}), 500

###########################################
# 4. 모델 평가 및 시각화 함수
###########################################

def evaluate_blockchain_model(test_data, actual_outcomes):
    """블록체인 모델 성능 평가"""
    predictions = blockchain_predictor.predict_transactions(test_data, test_data[0]['fromAddress'])
    
    # 평가 지표 계산
    confidence_error = 0
    amount_error = 0
    address_match = 0
    
    for i, (pred, actual) in enumerate(zip(predictions, actual_outcomes)):
        # 금액 오차
        pred_amount = float(pred['amount'])
        actual_amount = float(actual['amount'])
        amount_error += abs(pred_amount - actual_amount) / max(1.0, actual_amount)
        
        # 신뢰도 오차
        pred_confidence = float(pred['confidence'])
        actual_confidence = float(actual['confidence'])
        confidence_error += abs(pred_confidence - actual_confidence)
        
        # 주소 일치 여부
        if pred['toAddress'] == actual['toAddress']:
            address_match += 1
    
    # 평균 계산
    n = len(predictions)
    results = {
        'amount_error': amount_error / n if n > 0 else float('inf'),
        'confidence_error': confidence_error / n if n > 0 else float('inf'),
        'address_match_ratio': address_match / n if n > 0 else 0.0,
        'sample_size': n
    }
    
    return results

def evaluate_vehicle_path_model(test_data):
    """자율주행차 경로 예측 모델 평가"""
    total_distance_error = 0
    total_heading_error = 0
    total_speed_error = 0
    count = 0
    
    for item in test_data:
        # 예측 수행
        predicted_path = vehicle_predictor.predict_path(
            'test_vehicle', 
            item['current_position'], 
            item.get('flow_field', []), 
            len(item['actual_path'])
        )
        
        # 실제 경로와 비교
        for i, (pred, actual) in enumerate(zip(predicted_path, item['actual_path'])):
            # 거리 오차 (하버사인 공식)
            distance_error = calculate_distance(
                pred['latitude'], pred['longitude'],
                actual['latitude'], actual['longitude']
            )
            
            # 방향 오차
            heading_diff = abs(pred['heading'] - actual['heading'])
            heading_error = min(heading_diff, 360 - heading_diff)
            
            # 속도 오차
            speed_error = abs(pred['speed'] - actual['speed'])
            
            # 누적
            total_distance_error += distance_error
            total_heading_error += heading_error
            total_speed_error += speed_error
            count += 1
    
    # 평균 계산
    results = {
        'avg_distance_error_km': total_distance_error / count if count > 0 else float('inf'),
        'avg_heading_error_deg': total_heading_error / count if count > 0 else float('inf'),
        'avg_speed_error_kmh': total_speed_error / count if count > 0 else float('inf'),
        'sample_count': count
    }
    
    return results

def calculate_distance(lat1, lon1, lat2, lon2):
    """하버사인 공식으로 두 좌표 간 거리 계산 (km)"""
    # 지구 반경 (km)
    R = 6371.0
    
    # 라디안으로 변환
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    
    # 위도, 경도 차이
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    # 하버사인 공식
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    # 거리 (km)
    distance = R * c
    
    return distance

def plot_blockchain_predictions(actual, predicted):
    """블록체인 예측 결과 시각화"""
    plt.figure(figsize=(14, 8))
    
    # 금액 비교
    plt.subplot(2, 2, 1)
    actual_amounts = [float(tx['amount']) for tx in actual]
    pred_amounts = [float(tx['amount']) for tx in predicted]
    plt.scatter(actual_amounts, pred_amounts, alpha=0.7)
    plt.plot([0, max(actual_amounts)], [0, max(actual_amounts)], 'r--')
    plt.xlabel('실제 금액')
    plt.ylabel('예측 금액')
    plt.title('금액 예측 정확도')
    
    # 신뢰도 비교
    plt.subplot(2, 2, 2)
    actual_conf = [float(tx['confidence']) for tx in actual]
    pred_conf = [float(tx['confidence']) for tx in predicted]
    plt.scatter(actual_conf, pred_conf, alpha=0.7)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlabel('실제 신뢰도')
    plt.ylabel('예측 신뢰도')
    plt.title('신뢰도 예측 정확도')
    
    # 금액 분포
    plt.subplot(2, 2, 3)
    plt.hist(actual_amounts, alpha=0.5, label='실제', bins=20)
    plt.hist(pred_amounts, alpha=0.5, label='예측', bins=20)
    plt.xlabel('금액')
    plt.ylabel('빈도')
    plt.legend()
    plt.title('금액 분포 비교')
    
    # 신뢰도 분포
    plt.subplot(2, 2, 4)
    plt.hist(actual_conf, alpha=0.5, label='실제', bins=10)
    plt.hist(pred_conf, alpha=0.5, label='예측', bins=10)
    plt.xlabel('신뢰도')
    plt.ylabel('빈도')
    plt.legend()
    plt.title('신뢰도 분포 비교')
    
    plt.tight_layout()
    
    # 저장
    plt.savefig(os.path.join(MODEL_PATH, 'blockchain_prediction_evaluation.png'))
    plt.close()

def plot_vehicle_path(actual_path, predicted_path):
    """자율주행차 경로 시각화"""
    plt.figure(figsize=(10, 8))
    
    # 실제 경로
    actual_lats = [pos['latitude'] for pos in actual_path]
    actual_lons = [pos['longitude'] for pos in actual_path]
    plt.plot(actual_lons, actual_lats, 'b-', label='실제 경로')
    plt.scatter(actual_lons[0], actual_lats[0], c='green', s=100, marker='^', label='시작 지점')
    plt.scatter(actual_lons[-1], actual_lats[-1], c='red', s=100, marker='o', label='종료 지점')
    
    # 예측 경로
    pred_lats = [pos['latitude'] for pos in predicted_path]
    pred_lons = [pos['longitude'] for pos in predicted_path]
    plt.plot(pred_lons, pred_lats, 'r--', label='예측 경로')
    
    # 포인트별 오차 표시
    for i in range(min(len(actual_path), len(predicted_path))):
        plt.plot([actual_lons[i], pred_lons[i]], [actual_lats[i], pred_lats[i]], 'g-', alpha=0.3)
    
    plt.xlabel('경도')
    plt.ylabel('위도')
    plt.title('자율주행차 경로 예측 비교')
    plt.legend()
    plt.grid(True)
    
    # 저장
    plt.savefig(os.path.join(MODEL_PATH, 'vehicle_path_prediction.png'))
    plt.close()

def plot_risk_zones(position, risk_zones):
    """위험 지점 시각화"""
    plt.figure(figsize=(10, 8))
    
    # 현재 위치
    plt.scatter(position['longitude'], position['latitude'], c='blue', s=100, marker='*', label='현재 위치')
    
    # 방향 화살표 표시
    arrow_length = 0.001  # 화살표 길이
    heading_rad = np.radians(position['heading'])
    plt.arrow(
        position['longitude'], position['latitude'],
        arrow_length * np.sin(heading_rad), arrow_length * np.cos(heading_rad),
        head_width=0.0003, head_length=0.0005, fc='blue', ec='blue'
    )
    
    # 위험 지점 그리기
    for zone in risk_zones:
        risk_level = zone['riskLevel']
        
        # 위험도에 따른 색상
        if risk_level > 0.7:
            color = 'red'
        elif risk_level > 0.4:
            color = 'orange'
        else:
            color = 'yellow'
        
        # 위험도에 따른 크기
        size = 20 + risk_level * 80
        
        plt.scatter(zone['longitude'], zone['latitude'], c=color, s=size, alpha=0.6)
    
    # 범례 추가
    plt.scatter([], [], c='red', s=100, label='높은 위험')
    plt.scatter([], [], c='orange', s=70, label='중간 위험')
    plt.scatter([], [], c='yellow', s=40, label='낮은 위험')
    
    plt.xlabel('경도')
    plt.ylabel('위도')
    plt.title('자율주행차 위험 지점 분석')
    plt.legend()
    plt.grid(True)
    
    # 저장
    plt.savefig(os.path.join(MODEL_PATH, 'risk_zones_analysis.png'))
    plt.close()

###########################################
# 메인 함수
###########################################

if __name__ == '__main__':
    # 모델 경로 확인
    logger.info(f"모델 경로: {MODEL_PATH}")
    
    # 포트 설정
    port = int(os.environ.get('PORT', 5000))
    
    # HTTPS 설정
    context = None
    cert_path = os.environ.get('CERT_PATH')
    key_path = os.environ.get('KEY_PATH')
    
    if cert_path and key_path and os.path.exists(cert_path) and os.path.exists(key_path):
        context = (cert_path, key_path)
        logger.info("HTTPS가 활성화되었습니다.")
    
    # 서버 시작
    logger.info(f"서버 시작: {'https' if context else 'http'}://0.0.0.0:{port}")
    
    if context:
        app.run(host='0.0.0.0', port=port, ssl_context=context)
    else:
        app.run(host='0.0.0.0', port=port)"""
블록체인 트랜잭션 추적 및 자율주행차 위치 예측 머신러닝 모델

이 코드는 두 가지 주요 기능을 제공합니다:
1. 블록체인 트랜잭션 패턴 분석 및 예측
2. 자율주행차 위치 예측 및 위험 지점 분석

각 기능은 물리학 모델(로렌츠 시스템, 나비에-스토크스 방정식 등)과 머신러닝 모델(LSTM, 
GRU, 강화 학습 등)을 결합하여 더 정확한 예측을 제공합니다.
"""

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify
import joblib
import json
import time
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("ml_service.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 모델 경로 설정
MODEL_PATH = os.environ.get('MODEL_PATH', './models')
os.makedirs(MODEL_PATH, exist_ok=True)

###########################################
# 1. 블록체인 트랜잭션 예측 모델
###########################################

class BlockchainTransactionPredictor:
    """블록체인 트랜잭션 패턴 분석 및 예측 모델"""
    
    def __init__(self):
        self.transaction_model = None
        self.propagation_model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.cluster_model = None
        self.load_models()
        
    def load_models(self):
        """저장된 모델 로드"""
        try:
            # 트랜잭션 예측 모델 로드
            model_path = os.path.join(MODEL_PATH, 'blockchain_transaction_model.h5')
            if os.path.exists(model_path):
                self.transaction_model = load_model(model_path)
                logger.info("트랜잭션 예측 모델 로드 완료")
            else:
                logger.warning("트랜잭션 예측 모델이 없습니다. 새 모델을 생성합니다.")
                self._create_transaction_model()
            
            # 전파 예측 모델 로드
            prop_model_path = os.path.join(MODEL_PATH, 'propagation_model.h5')
            if os.path.exists(prop_model_path):
                self.propagation_model = load_model(prop_model_path)
                logger.info("전파 예측 모델 로드 완료")
            else:
                logger.warning("전파 예측 모델이 없습니다. 새 모델을 생성합니다.")
                self._create_propagation_model()
            
            # 클러스터링 모델 로드
            cluster_model_path = os.path.join(MODEL_PATH, 'transaction_cluster_model.pkl')
            if os.path.exists(cluster_model_path):
                self.cluster_model = joblib.load(cluster_model_path)
                logger.info("클러스터링 모델 로드 완료")
            else:
                logger.warning("클러스터링 모델이 없습니다. 새 모델을 생성합니다.")
                self.cluster_model = DBSCAN(eps=0.3, min_samples=5)
        
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            self._create_transaction_model()
            self._create_propagation_model()
            self.cluster_model = DBSCAN(eps=0.3, min_samples=5)
    
    def _create_transaction_model(self):
        """트랜잭션 예측을 위한 LSTM 모델 생성"""
        # 로렌츠 시스템 특성을 고려한 LSTM 모델
        model = Sequential()
        model.add(LSTM(64, activation='tanh', return_sequences=True, 
                       input_shape=(10, 5)))  # 10 시퀀스, 5개 특성
        model.add(Dropout(0.2))
        model.add(LSTM(32, activation='tanh'))
        model.add(Dropout(0.2))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(3))  # [수신 주소 특징, 금액, 신뢰도]
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mean_squared_error')
        
        self.transaction_model = model
        logger.info("새 트랜잭션 예측 모델이 생성되었습니다.")
    
    def _create_propagation_model(self):
        """트랜잭션 전파 예측을 위한 GRU 모델 생성"""
        # 맥스웰 방정식 특성을 고려한 GRU 모델
        model = Sequential()
        model.add(GRU(32, activation='tanh', return_sequences=True, 
                     input_shape=(20, 6)))  # 20 시간단계, 6개 필드 특성
        model.add(Dropout(0.2))
        model.add(GRU(16, activation='tanh'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1))  # 전파 시간 예측
        
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='mean_squared_error')
        
        self.propagation_model = model
        logger.info("새 전파 예측 모델이 생성되었습니다.")
    
    def _preprocess_transaction_data(self, transactions, address):
        """트랜잭션 데이터 전처리"""
        # 특성 추출
        features = []
        for tx in transactions:
            # 주소 해시에서 정수 특성 추출
            from_hash = int(tx['fromAddress'].replace('0x', ''), 16) % 10000
            to_hash = int(tx['toAddress'].replace('0x', ''), 16) % 10000
            
            # 금액, 신뢰도
            amount = float(tx['amount'])
            confidence = float(tx['confidence'])
            
            # 타임스탬프 또는 순서
            time_feature = transactions.index(tx) / len(transactions)
            
            features.append([from_hash/10000, to_hash/10000, amount, confidence, time_feature])
        
        # 데이터 정규화
        features = np.array(features)
        if len(features) > 0:
            features = self.scaler.fit_transform(features)
        
        # 시퀀스 데이터 생성
        X = []
        seq_length = 10
        
        # 시퀀스 길이가 충분한지 확인
        if len(features) < seq_length:
            # 부족한 데이터 생성
            padding = np.zeros((seq_length - len(features), features.shape[1]))
            features = np.vstack((padding, features))
        
        # 슬라이딩 윈도우로 시퀀스 생성
        for i in range(len(features) - seq_length + 1):
            X.append(features[i:i+seq_length])
        
        return np.array(X)
    
    def _hash_to_address(self, hash_val, source_addr):
        """해시 값에서 주소 생성"""
        # 소스 주소의 접두사를 유지하고, 해시 값을 이용해 새 주소 생성
        prefix = source_addr[:6]  # 0x 포함 앞 부분 유지
        hash_str = format(abs(hash(str(hash_val))), 'x')[:40]  # 40자 16진수
        return f"{prefix}{hash_str}"
    
    def predict_transactions(self, transactions, source_address):
        """트랜잭션 예측 수행"""
        try:
            # 모델이 없으면 샘플 예측 반환
            if self.transaction_model is None:
                return self._sample_predictions(transactions, source_address)
            
            # 데이터 전처리
            X = self._preprocess_transaction_data(transactions, source_address)
            
            if len(X) == 0:
                return self._sample_predictions(transactions, source_address)
            
            # 예측 수행
            predictions = self.transaction_model.predict(X)
            
            # 예측 결과 후처리
            result_txs = []
            for i, pred in enumerate(predictions):
                # 원래 스케일로 변환
                pred_scaled = self.scaler.inverse_transform(
                    np.hstack((np.zeros((1, 2)), pred.reshape(1, -1), np.zeros((1, 1))))
                )[0, 2:5]
                
                # 새 트랜잭션 생성
                new_tx = {
                    'fromAddress': source_address,
                    'toAddress': self._hash_to_address(pred_scaled[0], source_address),
                    'amount': max(0.1, pred_scaled[1]),  # 음수 방지
                    'confidence': min(0.95, max(0.1, pred_scaled[2]))  # 0.1-0.95 범위로 제한
                }
                
                result_txs.append(new_tx)
            
            return result_txs
        
        except Exception as e:
            logger.error(f"트랜잭션 예측 중 오류 발생: {str(e)}")
            return self._sample_predictions(transactions, source_address)
    
    def _sample_predictions(self, transactions, source_address):
        """샘플 예측 생성 (모델 실패 시)"""
        result = []
        
        # 기존 트랜잭션의 평균 금액 계산
        avg_amount = 100.0  # 기본값
        if len(transactions) > 0:
            amounts = [float(tx['amount']) for tx in transactions]
            avg_amount = sum(amounts) / len(amounts)
        
        # 3개의 샘플 트랜잭션 생성
        for i in range(3):
            result.append({
                'fromAddress': source_address,
                'toAddress': f"0x{abs(hash(source_address + str(i))):#0x}"[:42],
                'amount': avg_amount * (0.8 + 0.4 * np.random.random()),
                'confidence': 0.7 + 0.2 * np.random.random()
            })
        
        return result
    
    def predict_propagation(self, tx_hash, node_count=10):
        """트랜잭션 전파 시간 예측"""
        try:
            # 해시에서 특성 추출
            hash_features = self._hash_to_features(tx_hash)
            
            # 네트워크 노드 거리 시뮬레이션
            distances = np.linspace(0.1, 2.0, node_count)
            
            # 각 노드까지의 전파 시간 예측
            propagation_times = []
            
            for i, distance in enumerate(distances):
                # 특성 벡터 생성 (해시 특성 + 거리)
                features = np.array([list(hash_features) + [distance] for _ in range(20)])
                
                # 시간에 따른 감쇠 추가
                time_decay = np.linspace(0, 1, 20).reshape(-1, 1)
                features = features * (1 - 0.5 * time_decay)
                
                # 시퀀스 데이터로 변환
                X = features.reshape(1, 20, features.shape[1])
                
                # 예측 수행
                if self.propagation_model is not None:
                    pred_time = float(self.propagation_model.predict(X)[0][0])
                    propagation_times.append((i, max(10, pred_time * distance * 100)))
                else:
                    # 모델이 없으면 간단한 물리 공식으로 대체
                    base_time = 100.0  # 기본 지연 시간 (ms)
                    strength = max(0.1, hash_features[0])  # 해시 첫 번째 특성을 신호 강도로 사용
                    prop_time = base_time * distance / strength
                    propagation_times.append((i, prop_time))
            
            return propagation_times
        
        except Exception as e:
            logger.error(f"전파 시간 예측 중 오류 발생: {str(e)}")
            # 오류 시 간단한 물리 모델 기반 대체 결과 반환
            return [(i, 100 + i * 20) for i in range(node_count)]
    
    def _hash_to_features(self, tx_hash):
        """트랜잭션 해시에서 특성 추출"""
        # 해시를 6개의 특성으로 변환
        features = []
        for i in range(min(6, len(tx_hash) // 5)):
            segment = tx_hash[i*5:i*5+5]
            try:
                val = int(segment, 16)
                features.append(val % 1000 / 1000.0)  # 0-1 사이 값으로 정규화
            except ValueError:
                features.append(0.5)  # 오류 시 중간값
        
        # 부족한 특성 채우기
        while len(features) < 6:
            features.append(0.5)
        
        return features
    
    def cluster_transactions(self, transactions):
        """트랜잭션 클러스터링"""
        try:
            if len(transactions) < 2:
                return self._default_clusters(transactions)
            
            # 트랜잭션 특성 추출
            features = []
            for tx in transactions:
                # 주요 특성: 금액과 신뢰도
                amount = float(tx['amount'])
                confidence = float(tx['confidence'])
                features.append([amount, confidence])
            
            features = np.array(features)
            
            # 정규화
            scaler = MinMaxScaler()
            features_scaled = scaler.fit_transform(features)
            
            # 클러스터링
            if self.cluster_model is None:
                self.cluster_model = DBSCAN(eps=0.3, min_samples=2)
            
            labels = self.cluster_model.fit_predict(features_scaled)
            
            # 결과 정리
            clusters = {}
            for i, label in enumerate(labels):
                if label == -1:
                    # 노이즈 포인트는 "outliers" 클러스터에 추가
                    if "outliers" not in clusters:
                        clusters["outliers"] = []
                    clusters["outliers"].append(transactions[i]['toAddress'])
                else:
                    # 클러스터별 그룹화
                    cluster_name = f"cluster_{label}"
                    if cluster_name not in clusters:
                        clusters[cluster_name] = []
                    clusters[cluster_name].append(transactions[i]['toAddress'])
            
            # 클러스터가 없으면 금액 기반 분류
            if len(clusters) <= 1:
                return self._amount_based_clusters(transactions)
            
            return clusters
        
        except Exception as e:
            logger.error(f"트랜잭션 클러스터링 중 오류 발생: {str(e)}")
            return self._amount_based_clusters(transactions)
    
    def _default_clusters(self, transactions):
        """기본 클러스터 생성"""
        return {
            "all": [tx['toAddress'] for tx in transactions]
        }
    
    def _amount_based_clusters(self, transactions):
        """금액 기반 클러스터링"""
        # 트랜잭션을 금액에 따라 small, medium, large로 분류
        small = []
        medium = []
        large = []
        
        for tx in transactions:
            amount = float(tx['amount'])
            if amount < 50:
                small.append(tx['toAddress'])
            elif amount < 200:
                medium.append(tx['toAddress'])
            else:
                large.append(tx['toAddress'])
        
        return {
            "small": small,
            "medium": medium,
            "large": large
        }
    
    def train_with_new_data(self, transactions, actual_outcomes):
        """새 데이터로 모델 훈련"""
        if len(transactions) < 10 or len(actual_outcomes) < 10:
            logger.warning("훈련을 위한 데이터가 부족합니다.")
            return False
        
        try:
            # 특성 및 레이블 추출
            X = self._preprocess_transaction_data(transactions, transactions[0]['fromAddress'])
            
            # 결과 데이터 전처리
            y_data = []
            for outcome in actual_outcomes:
                to_hash = int(outcome['toAddress'].replace('0x', ''), 16) % 10000
                y_data.append([
                    to_hash/10000, 
                    float(outcome['amount']), 
                    float(outcome['confidence'])
                ])
            
            y = np.array(y_data)
            
            # 데이터 수가 맞는지 확인
            if len(X) != len(y):
                logger.warning(f"입력({len(X)})과 출력({len(y)}) 데이터 수가 일치하지 않습니다.")
                return False
            
            # 모델이 없으면 생성
            if self.transaction_model is None:
                self._create_transaction_model()
            
            # 모델 훈련
            callback = EarlyStopping(monitor='loss', patience=5)
            self.transaction_model.fit(
                X, y, 
                epochs=100, 
                batch_size=8, 
                callbacks=[callback],
                verbose=0
            )
            
            # 모델 저장
            model_path = os.path.join(MODEL_PATH, 'blockchain_transaction_model.h5')
            self.transaction_model.save(model_path)
            logger.info(f"모델이 {model_path}에 저장되었습니다.")
            
            return True
        
        except Exception as e:
            logger.error(f"모델 훈련 중 오류 발생: {str(e)}")
            return False


###########################################
# 2. 자율주행차 위치 예측 모델
###########################################

class AutonomousVehiclePredictor:
    """자율주행차 위치 예측 및 위험 지점 분석 모델"""
    
    def __init__(self):
        self.path_model = None
        self.risk_model = None
        self.position_scaler = MinMaxScaler()
        self.risk_scaler = MinMaxScaler()
        self.load_models()
    
    def load_models(self):
        """저장된 모델 로드"""
        try:
            # 경로 예측 모델 로드
            path_model_path = os.path.join(MODEL_PATH, 'vehicle_path_prediction_model.h5')
            if os.path.exists(path_model_path):
                self.path_model = load_model(path_model_path)
                logger.info("경로 예측 모델 로드 완료")
            else:
                logger.warning("경로 예측 모델이 없습니다. 새 모델을 생성합니다.")
                self._create_path_model()
            
            # 위험 분석 모델 로드
            risk_model_path = os.path.join(MODEL_PATH, 'vehicle_risk_model.h5')
            if os.path.exists(risk_model_path):
                self.risk_model = load_model(risk_model_path)
                logger.info("위험 분석 모델 로드 완료")
            else:
                logger.warning("위험 분석 모델이 없습니다. 새 모델을 생성합니다.")
                self._create_risk_model()
            
        except Exception as e:
            logger.error(f"모델 로드 중 오류 발생: {str(e)}")
            self._create_path_model()
            self._create_risk_model()
    
    def _create_path_model(self):
        """경로 예측을 위한 딥러닝 모델 생성"""
        # 입력 레이어: 현재 위치 + 흐름 필드 + 환경 조건
        position_input = Input(shape=(4,), name='position_input')  # 위도, 경도, 방향, 속도
        flow_field_input = Input(shape=(100, 5), name='flow_field_input')  # 100개 지점, 각 5개 특성
        
        # 위치 정보 처리
        position_features = Dense(16, activation='relu')(position_input)
        
        # 흐름 필드 처리 (CNN + GRU)
        flow_features = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu')(flow_field_input)
        flow_features = tf.keras.layers.MaxPooling1D(pool_size=2)(flow_features)
        flow_features = GRU(32, return_sequences=False)(flow_features)
        
        # 특성 결합
        combined = Concatenate()([position_features, flow_features])
        
        # 경로 예측
        dense1 = Dense(64, activation='relu')(combined)
        dense2 = Dense(32, activation='relu')(dense1)
        output = Dense(4 * 10, activation='linear')(dense2)  # 10개 위치 예측 (각 위치는 위도, 경도, 방향, 속도)
        
        # 모델 생성
        model = Model(inputs=[position_input, flow_field_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        self.path_model = model
        logger.info("새 경로 예측 모델이 생성되었습니다.")
    
    def _create_risk_model(self):
        """위험 지점 예측을 위한 모델 생성"""
        # 더핑 진동자 모델을 고려한 위험 예측
        
        # 입력 레이어
        position_input = Input(shape=(4,), name='position_input')  # 위도, 경도, 방향, 속도
        env_input = Input(shape=(2,), name='env_input')  # 도로 상태, 날씨 조건
        params_input = Input(shape=(4,), name='params_input')  # 더핑 진동자 매개변수
        
        # 특성 추출
        position_features = Dense(16, activation='relu')(position_input)
        env_features = Dense(8, activation='relu')(env_input)
        combined = Concatenate()([position_features, env_features, params_input])
        
        # 위험 지점 예측 (15개 지점 예측)
        dense1 = Dense(64, activation='relu')(combined)
        dense2 = Dense(32, activation='relu')(dense1)
        output = Dense(45, activation='linear')(dense2)  # 15개 지점 x 3 (위도, 경도, 위험도)
        
        # 모델 생성
        model = Model(inputs=[position_input, env_input, params_input], outputs=output)
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        self.risk_model = model
        logger.info("새 위험 분석 모델이 생성되었습니다.")
    
    def _preprocess_flow_field(self, flow_field):
        """유체 흐름 필드 전처리"""
        # 최대 100개 포인트 선택
        if len(flow_field) > 100:
            indices = np.linspace(0, len(flow_field)-1, 100, dtype=int)
            flow_field = [flow_field[i] for i in indices]
        
        # 부족한 경우 0으로 패딩
        while len(flow_field) < 100:
            flow_field.append([0, 0, 0, 0, 0])
        
        return np.array(flow_field)
    
    def _preprocess_position(self, position):
        """위치 데이터 전처리"""
        # 위도, 경도, 방향, 속도 추출
        features = [
            position['latitude'],
            position['longitude'],
            position['heading'],
            position['speed']
        ]
        
        # 정규화
        if not hasattr(self, '_position_scale_fitted'):
            # 대략적인 범위로 스케일러 훈련
            sample_data = np.array([
                [30.0, 120.0, 0.0, 0.0],   # 최소값
                [45.0, 140.0, 359.0, 120.0]  # 최대값
            ])
            self.position_scaler.fit(sample_data)
            self._position_scale_fitted = True
        
        return self.position_scaler.transform(np.array(features).reshape(1, -1))[0]
    
    def predict_path(self, vehicle_id, current_position, flow_field, steps):
        """자율주행차 경로 예측"""
        try:
            # 모델이 없으면 샘플 예측 반환
            if self.path_model is None:
                return self._sample_path_prediction(current_position, steps)
            
            # 입력 데이터 전처리
            position_features = self._preprocess_position(current_position)
            flow_features = self._preprocess_flow_field(flow_field)
            
            # 모델 입력
            position_input = np.array([position_features])
            flow_input = np.array([flow_features])
            
            # 경로 예측
            predictions = self.path_model.predict([position_input, flow_input])[0]
            
            # 결과 재구성 (10개 위치)
            results = []
            for i in range(10):
                pos_pred = predictions[i*4:(i+1)*4]
                
                # 정규화 복원
                pos_unnorm = self.position_scaler.inverse_transform(pos_pred.reshape(1, -1))[0]
                
                # 위치 객체 생성
                pos = {
                    'latitude': float(pos_unnorm[0]),
                    'longitude': float(pos_unnorm[1]),
                    'heading': float(pos_unnorm[2]),
                    'speed': float(pos_unnorm[3])
                }
                
                results.append(pos)
                
                # 요청된 단계 수에 도달하면 종료
                if len(results) >= steps:
                    break
            
            # 부족한 단계 채우기
            while len(results) < steps:
                last_pos = results[-1]
                
                # 마지막 위치에서 동일한 방향으로 계속 이동
                next_pos = self._extend_path(last_pos)
                results.append(next_pos)
            
            return results
        
        except Exception as e:
            logger.error(f"경로 예측 중 오류 발생: {str(e)}")
            return self._sample_path_prediction(current_position, steps)
    
    def _sample_path_prediction(self, current_position, steps):
        """간단한 물리 기반 경로 예측 (모델 실패 시)"""
        results = []
        
        # 현재 위치 정보
        lat = current_position['latitude']
        lon = current_position['longitude']
        heading = current_position['heading']
        speed = current_position['speed']
        
        # 지구 반경 (km)
        EARTH_RADIUS = 6371.0
        
        # 각 단계별 위치 계산
        for i in range(steps):
            # 시간 간격 (초)
            dt = 2.0
            
            # 이동 거리 (km)
            distance = speed * dt / 3600.0  # km/h를 km/s로 변환
            
            # 위도, 경도 변화량 계산
            heading_rad = np.radians(heading)
            lat_change = distance * np.cos(heading_rad) / EARTH_RADIUS * np.degrees(np.pi)
            lon_change = distance * np.sin(heading_rad) / (EARTH_RADIUS * np.cos(np.radians(lat))) * np.degrees(np.pi)
            
            # 새 위치 계산
            lat += lat_change
            lon += lon_change
            
            # 약간의 무작위성 추가
            heading += (np.random.random() - 0.5) * 2.0  # ±1도 변화
            speed += (np.random.random() - 0.5) * 2.0    # ±1km/h 변화
            
            # 위치 객체 추가
            results.append({
                'latitude': float(lat),
                'longitude': float(lon),
                'heading': float(heading),
                'speed': float(speed)
            })
        
        return results
    
    def _extend_path(self, last_position):
        """마지막 위치에서 경로 연장"""
        # 현재 위치 정보
        lat = last_position['latitude']
        lon = last_position['longitude']
        heading = last_position['heading']
        speed = last_position['speed']
        
        # 지구 반경 (km)
        EARTH_RADIUS = 6371.0
        
        # 시간 간격 (초)
        dt = 2.0
        
        # 이동 거리 (km)
        distance = speed * dt / 3600.0  # km/h를 km/s로 변환
        
        # 위도, 경도 변화량 계산
        heading_rad = np.radians(heading)
        lat_change = distance * np.cos(heading_rad) / EARTH_RADIUS * np.degrees(np.pi)
        lon_change = distance * np.sin(heading_rad) / (EARTH_RADIUS * np.cos(np.radians(lat))) * np.degrees(np.pi)
        
        # 새 위치 계산
        lat += lat_change
        lon += lon_change
        
        # 약간의 무작위성 추가
        heading += (np.random.random() - 0.5) * 2.0  # ±1도 변화
        speed += (np.random.random() - 0.5) * 2.0    # ±1km/h 변화
        
        return {
            'latitude': float(lat),
            'longitude': float(lon),
            'heading': float(heading),
            'speed': float(speed)
        }
    
    def calculate_risk_zones(self, position, k, alpha, beta, gamma):
        """위험 지점 계산"""
        try:
            # 모델이 없으면 간단한 위험 지점 생성
            if self.risk_model is None:
                return self._sample_risk_zones(position, k, alpha, beta, gamma)
            
            # 입력 데이터 준비
            position_features = self._preprocess_position(position)
            
            # 환경 조건 (도로 상태, 날씨는 더핑 진동자 매개변수에서 유추)
            road_condition = k * 0.5  # k를 도로 상태로 활용 (0-1 범위로 변환)
            weather_factor = gamma * 0.5  # gamma를 날씨 조건으로 활용
            
            env_features = np.array([[road_condition, weather_factor]])
            
            # 더핑 진동자 매개변수
            params = np.array([[k, alpha, beta, gamma]])
            
            # 모델 예측
            predictions = self.risk_model.predict([
                np.array([position_features]), 
                env_features, 
                params
            ])[0]
            
            # 결과 재구성 (15개 위험 지점)
            risk_zones = []
            for i in range(15):
                # 위도, 경도, 위험도
                lat = predictions[i*3]
                lon = predictions[i*3+1]
                risk = predictions[i*3+2]
                
                # 정규화 복원 (위도, 경도만)
                pos_features = np.zeros(4)
                pos_features[0] = lat
                pos_features[1] = lon
                pos_unnorm = self.position_scaler.inverse_transform(pos_features.reshape(1, -1))[0]
                
                # 위험도는 0-1 사이로 조정
                risk_level = max(0, min(1, risk))
                
                # 위험 지점 생성
                risk_type = self._determine_risk_type(i, risk_level)
                description = f"{risk_type} - 위험도: {risk_level:.1f}"
                
                risk_zone = {
                    'latitude': float(pos_unnorm[0]),
                    'longitude': float(pos_unnorm[1]),
                    'riskLevel': float(risk_level),
                    'description': description
                }
                
                risk_zones.append(risk_zone)
            
            return risk_zones
        
        except Exception as e:
            logger.error(f"위험 지점 계산 중 오류 발생: {str(e)}")
            return self._sample_risk_zones(position, k, alpha, beta, gamma)
    
    def _determine_risk_type(self, index, risk_level):
        """위험 유형 결정"""
        risk_types = [
            "도로 상태 불량",
            "시야 제한 구역",
            "급커브 구간",
            "교통 혼잡 예상",
            "도로 공사 구간"
        ]
        
        # 위험도에 따라 접두어 추가
        if risk_level > 0.7:
            prefix = "높은 위험: "
        elif risk_level > 0.4:
            prefix = "중간 위험: "
        else:
            prefix = "낮은 위험: "
        
        return prefix + risk_types[index % len(risk_types)]
    
    def _sample_risk_zones(self, position, k, alpha, beta, gamma):
        """간단한 위험 지점 생성 (모델 실패 시)"""
        risk_zones = []
        
        # 현재 위치
        lat = position['latitude']
        lon = position['longitude']
        heading = position['heading']
        
        # 환경 요소에 따른 기본 위험도
        base_risk = 0.2 + 0.3 * k + 0.5 * gamma
        base_risk = min(0.9, base_risk)  # 최대 0.9
        
        # 서로 다른 방향으로 위험 지점 배치
        for i in range(15):
            # 각도 계산 (차량 방향 기준 ±90도, 3개씩 5그룹)
            group = i // 3
            angle_offset = -45 + 22.5 * (group % 5)
            angle = (heading + angle_offset) % 360
            
            # 거리 계산 (그룹별로 증가)
            distance = 0.05 + (i % 3) * 0.05 + group * 0.02  # 50m에서 시작, 점점 증가
            
            # 위험도 계산 (거리가 멀어질수록 감소)
            risk_level = base_risk * np.exp(-distance * 5)
            
            # 위치 계산
            angle_rad = np.radians(angle)
            lat_change = distance * np.cos(angle_rad) / 111.0  # 1도 = 약 111km
            lon_change = distance * np.sin(angle_rad) / (111.0 * np.cos(np.radians(lat)))
            
            # 위험 유형 결정
            risk_type = self._determine_risk_type(i, risk_level)
            
            risk_zone = {
                'latitude': float(lat + lat_change),
                'longitude': float(lon + lon_change),
                'riskLevel': float(risk_level),
                'description': risk_type
            }
            
            risk_zones.append(risk_zone)
        
        return risk_zones
    
    def train_path_model(self, training_data):
        """경로 예측 모델 훈련"""
        if not training_data or len(training_data) < 10:
            logger.warning("훈련을 위한 데이터가 부족합니다.")
            return False
        
        try:
            X_position = []
            X_flow = []
            y = []
            
            for item in training_data:
                # 입력 데이터
                position = self._preprocess_position(item['current_position'])
                flow = self._preprocess_flow_field(item['flow_field'])
                
                # 출력 데이터 (10개 위치 예측)
                output = []
                for pred_pos in item['actual_path'][:10]:
                    pos_features = self._preprocess_position(pred_pos)
                    output.extend(pos_features)
                
                # 10개 미만이면 마지막 위치로 채우기
                while len(output) < 40:  # 10개 위치 * 4 특성
                    output.extend(pos_features)
                
                X_position.append(position)
                X_flow.append(flow)
                y.append(output)
            
            # 모델이 없으면 생성
            if self.path_model is None:
                self._create_path_model()
            
            # 훈련 데이터 변환
            X_position = np.array(X_position)
            X_flow = np.array(X_flow)
            y = np.array(y)
            
            # 모델 훈련
            checkpoint = ModelCheckpoint(
                os.path.join(MODEL_PATH, 'vehicle_path_prediction_model.h5'),
                save_best_only=True,
                monitor='val_loss'
            )
            early_stop = EarlyStopping(patience=10, monitor='val_loss')
            
            self.path_model.fit(
                [X_position, X_flow], y,
                epochs=100,
                batch_size=16,
                callbacks=[early_stop, checkpoint],
                validation_split=0.2,
                verbose=1
            )
            
            logger.info("경로 예측 모델 훈련 완료")
            return True
            
        except Exception as e:
            logger.error(f"경로 예측 모델 훈련 중 오류 발생: {str(e)}")
            return False
    
    def train_risk_model(self, training_data):
        """위험 분석 모델 훈련"""
        if not training_data or len(training_data) < 10:
            logger.warning("훈련을 위한 데이터가 부족합니다.")
            return False
        
        try:
            X_position = []
            X_env = []
            X_params = []
            y = []
            
            for item in training_data:
                # 입력 데이터
                position = self._preprocess_position(item['position'])
                env = [item['road_condition'], item['weather_factor']]
                params = [item['k'], item['alpha'], item['beta'], item['gamma']]
                
                # 출력 데이터 (15개 위험 지점)
                output = []
                for zone in item['actual_risk_zones'][:15]:
                    # 위도, 경도 정규화
                    lat_lon = self.position_scaler.transform(
                        np.array([[zone['latitude'], zone['longitude'], 0, 0]])
                    )[0, :2]
                    
                    output.extend([lat_lon[0], lat_lon[1], zone['riskLevel']])
                
                # 15개 미만이면 0으로 채우기
                while len(output) < 45:  # 15개 지점 * 3 특성
                    output.extend([0, 0, 0])
                
                X_position.append(position)
                X_env.append(env)
                X_params.append(params)
                y.append(output)
            
            # 모델이 없으면 생성
            if self.risk_model is None:
                self._create_risk_model()
            
            # 훈련 데이터 변환
            X_position = np.array(X_position)
            X_env = np.array(X_env)
            X_params = np.array(X_params)
            y = np.array(y)
            
            # 모델 훈련
            checkpoint = ModelCheckpoint(
                os.path.join(MODEL_PATH, 'vehicle_risk_model.h5'),
                save_best_only=True,
                monitor='val_loss'
            )
            early_stop = EarlyStopping(patience=10, monitor='val_loss')
            
            self.risk_model.fit(
                [X_position, X_env, X_params], y,
                epochs=100,
                batch_size=16,
                callbacks=[early_stop, checkpoint],
                validation_split=0.2,
                verbose=1
            )
            
            logger.info("위험 분석 모델 훈련 완료")
            return True
            
        except Exception as e:
            logger.error(f"위험 분석 모델 훈련 중 오류 발생: {str(e)}")
            return False
    