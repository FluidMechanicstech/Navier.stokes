package com.blocktracker;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * 블록체인 트랜잭션 추적 시스템
 * 
 * 이 시스템은 로렌츠 시스템(카오스 이론)과 맥스웰 방정식(전자기학)을 활용하여
 * 블록체인 트랜잭션을 추적하고 예측하는 기능을 제공합니다.
 * 
 * 주요 구성 요소:
 * 1. 물리 모델 (LorenzSystem, MaxwellPropagationModel)
 * 2. 서비스 (BlockchainTrackerService)
 * 3. 데이터 모델 (BlockchainTransaction)
 * 4. 머신러닝 통합 (MachineLearningService)
 */

/**
 * 로렌츠 시스템 (카오스 방정식) 구현
 * 
 * 카오스 이론을 기반으로 비선형 패턴을 분석하고 예측하기 위한 모델
 */
class LorenzSystem {
    // 로렌츠 시스템 매개변수
    private double sigma = 10.0; // 시그마
    private double rho = 28.0; // 로
    private double beta = 8.0/3.0; // 베타
    
    /**
     * 로렌츠 방정식 시스템을 계산
     * dx/dt = sigma * (y - x)
     * dy/dt = x * (rho - z) - y
     * dz/dt = x * y - beta * z
     */
    public double[][] solve(double[] initialState, int steps) {
        double x = initialState[0];
        double y = initialState[1];
        double z = initialState[2];
        double[][] result = new double[steps][3];
        double dt = 0.01; // 시간 단계
        
        for (int i = 0; i < steps; i++) {
            // 룽게-쿠타 4차 적분법으로 더 정확한 해 계산
            double[] k1 = derivatives(x, y, z);
            double[] k2 = derivatives(x + k1[0]*dt/2, y + k1[1]*dt/2, z + k1[2]*dt/2);
            double[] k3 = derivatives(x + k2[0]*dt/2, y + k2[1]*dt/2, z + k2[2]*dt/2);
            double[] k4 = derivatives(x + k3[0]*dt, y + k3[1]*dt, z + k3[2]*dt);
            
            x += (k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) * dt / 6;
            y += (k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) * dt / 6;
            z += (k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) * dt / 6;
            
            result[i][0] = x;
            result[i][1] = y;
            result[i][2] = z;
        }
        
        return result;
    }
    
    /**
     * 로렌츠 방정식의 미분 계산
     */
    private double[] derivatives(double x, double y, double z) {
        double dx = sigma * (y - x);
        double dy = x * (rho - z) - y;
        double dz = x * y - beta * z;
        return new double[] {dx, dy, dz};
    }
    
    /**
     * 리아푸노프 지수 계산을 통한 트랜잭션 패턴의 카오스적 특성 분석
     * 양의 리아푸노프 지수는 시스템의 카오스적 특성을 나타냄
     */
    public double calculateLyapunovExponent(double[] initialState, int steps) {
        double epsilon = 1e-10; // 초기 편차
        
        // 원래 궤적 계산
        double[][] traj1 = solve(initialState, steps);
        
        // 약간 다른 초기 조건으로 궤적 계산
        double[] perturbedState = new double[3];
        perturbedState[0] = initialState[0] + epsilon;
        perturbedState[1] = initialState[1];
        perturbedState[2] = initialState[2];
        double[][] traj2 = solve(perturbedState, steps);
        
        // 시간에 따른 분리율 계산
        double[] distances = new double[steps];
        for (int i = 0; i < steps; i++) {
            double dx = traj1[i][0] - traj2[i][0];
            double dy = traj1[i][1] - traj2[i][1];
            double dz = traj1[i][2] - traj2[i][2];
            distances[i] = Math.sqrt(dx*dx + dy*dy + dz*dz);
        }
        
        // 리아푸노프 지수 계산
        double sum = 0;
        for (int i = 1; i < steps; i++) {
            if (distances[i] > 0 && distances[i-1] > 0) {
                sum += Math.log(distances[i] / distances[i-1]);
            }
        }
        
        return sum / (steps * 0.01); // 0.01은 시간 간격 dt
    }
    
    /**
     * 매개변수 설정 메서드
     */
    public void setParameters(double sigma, double rho, double beta) {
        this.sigma = sigma;
        this.rho = rho;
        this.beta = beta;
    }
}

/**
 * 맥스웰 전파 모델 구현
 * 
 * 전자기학 기반으로 블록체인 네트워크에서 트랜잭션이 전파되는 방식을 모델링
 */
class MaxwellPropagationModel {
    // 전자기장 특성
    private double permittivity = 8.85e-12; // 유전율 (ε₀)
    private double permeability = 1.257e-6; // 투자율 (μ₀)
    private double conductivity = 0.01; // 전도도 (σ)
    
    // 시뮬레이션 매개변수
    private int gridSize = 50; // 그리드 크기
    private double timeStep = 1e-10; // 시간 단계
    private double cellSize = 0.01; // 셀 크기 (m)
    
    /**
     * 블록체인 트랜잭션 전파 시뮬레이션
     * 맥스웰 방정식을 사용하여 네트워크 내에서 트랜잭션이 전파되는 방식을 모델링
     */
    public double[][] simulatePropagation(double[] initialField, int steps) {
        // 전기장 및 자기장 초기화
        double[][][] electricField = new double[gridSize][gridSize][3]; // 3차원 전기장 (Ex, Ey, Ez)
        double[][][] magneticField = new double[gridSize][gridSize][3]; // 3차원 자기장 (Bx, By, Bz)
        double[][][] current = new double[gridSize][gridSize][3]; // 전류 밀도 (Jx, Jy, Jz)
        
        // 중앙에 초기 필드 설정
        int center = gridSize / 2;
        electricField[center][center][0] = initialField[0]; // Ex
        electricField[center][center][1] = initialField[1]; // Ey
        electricField[center][center][2] = initialField[2]; // Ez
        magneticField[center][center][0] = initialField[3]; // Bx
        magneticField[center][center][1] = initialField[4]; // By
        magneticField[center][center][2] = initialField[5]; // Bz
        
        // 결과 저장 배열
        double[][] propagationResult = new double[steps][6]; // 중앙점의 전기장과 자기장 값 저장
        
        // 시간에 따른 전자기장 계산
        for (int step = 0; step < steps; step++) {
            // 전기장 업데이트 (∇×B = μ₀J + μ₀ε₀∂E/∂t)
            updateElectricField(electricField, magneticField, current);
            
            // 자기장 업데이트 (∇×E = -∂B/∂t)
            updateMagneticField(electricField, magneticField);
            
            // 전류 밀도 업데이트 (J = σE)
            updateCurrent(electricField, current);
            
            // 중앙점의 필드 값 저장
            propagationResult[step][0] = electricField[center][center][0];
            propagationResult[step][1] = electricField[center][center][1];
            propagationResult[step][2] = electricField[center][center][2];
            propagationResult[step][3] = magneticField[center][center][0];
            propagationResult[step][4] = magneticField[center][center][1];
            propagationResult[step][5] = magneticField[center][center][2];
        }
        
        return propagationResult;
    }
    
    /**
     * FDTD(유한 차분 시간 영역) 방법으로 전기장 업데이트
     */
    private void updateElectricField(double[][][] electricField, double[][][] magneticField, double[][][] current) {
        double factor = timeStep / (permittivity * cellSize);
        for (int i = 1; i < gridSize - 1; i++) {
            for (int j = 1; j < gridSize - 1; j++) {
                // x 방향 전기장 업데이트
                electricField[i][j][0] += factor * (
                    (magneticField[i][j+1][2] - magneticField[i][j][2]) / cellSize -
                    (magneticField[i+1][j][1] - magneticField[i][j][1]) / cellSize -
                    current[i][j][0]
                );
                
                // y 방향 전기장 업데이트
                electricField[i][j][1] += factor * (
                    (magneticField[i+1][j][0] - magneticField[i][j][0]) / cellSize -
                    (magneticField[i][j+1][2] - magneticField[i][j][2]) / cellSize -
                    current[i][j][1]
                );
                
                // z 방향 전기장 업데이트
                electricField[i][j][2] += factor * (
                    (magneticField[i][j+1][1] - magneticField[i][j][1]) / cellSize -
                    (magneticField[i+1][j][0] - magneticField[i][j][0]) / cellSize -
                    current[i][j][2]
                );
            }
        }
    }
    
    /**
     * FDTD 방법으로 자기장 업데이트
     */
    private void updateMagneticField(double[][][] electricField, double[][][] magneticField) {
        double factor = timeStep / (permeability * cellSize);
        for (int i = 1; i < gridSize - 1; i++) {
            for (int j = 1; j < gridSize - 1; j++) {
                // x 방향 자기장 업데이트
                magneticField[i][j][0] -= factor * (
                    (electricField[i][j+1][2] - electricField[i][j][2]) / cellSize -
                    (electricField[i+1][j][1] - electricField[i][j][1]) / cellSize
                );
                
                // y 방향 자기장 업데이트
                magneticField[i][j][1] -= factor * (
                    (electricField[i+1][j][0] - electricField[i][j][0]) / cellSize -
                    (electricField[i][j+1][2] - electricField[i][j][2]) / cellSize
                );
                
                // z 방향 자기장 업데이트
                magneticField[i][j][2] -= factor * (
                    (electricField[i][j+1][1] - electricField[i][j][1]) / cellSize -
                    (electricField[i+1][j][0] - electricField[i][j][0]) / cellSize
                );
            }
        }
    }
    
    /**
     * 전도도에 따른 전류 밀도 업데이트
     */
    private void updateCurrent(double[][][] electricField, double[][][] current) {
        for (int i = 0; i < gridSize; i++) {
            for (int j = 0; j < gridSize; j++) {
                // J = σE
                current[i][j][0] = conductivity * electricField[i][j][0];
                current[i][j][1] = conductivity * electricField[i][j][1];
                current[i][j][2] = conductivity * electricField[i][j][2];
            }
        }
    }
    
    /**
     * 블록체인 네트워크에서 트랜잭션 전파 시간 예측
     * 맥스웰 방정식을 기반으로 트랜잭션이 네트워크 내에서 전파되는 시간 예측
     */
    public double[][] predictPropagationTimes(String txHash, int nodeCount) {
        // 트랜잭션 해시에서 초기 필드 값 생성
        double[] initialField = getFieldsFromTxHash(txHash);
        
        // 전파 시뮬레이션 실행
        double[][] fieldEvolution = simulatePropagation(initialField, 100);
        
        // 필드 강도에 따른 전파 시간 계산
        double[][] propagationTimes = new double[nodeCount][2]; // [노드 ID, 전파 시간]
        for (int i = 0; i < nodeCount; i++) {
            // 노드의 네트워크 상 거리 계산 (중앙에서의 거리)
            double distance = getNodeDistance(i, nodeCount);
            
            // 거리와 필드 강도에 따른 전파 시간 추정
            double fieldStrength = getFieldStrengthAtDistance(fieldEvolution, distance);
            double propagationTime = calculatePropagationTime(distance, fieldStrength);
            
            propagationTimes[i][0] = i; // 노드 ID
            propagationTimes[i][1] = propagationTime; // 전파 시간 (ms)
        }
        
        return propagationTimes;
    }
    
    /**
     * 트랜잭션 해시에서 전자기장 초기값 추출
     */
    private double[] getFieldsFromTxHash(String txHash) {
        // 트랜잭션 해시값을 전기장과 자기장 성분으로 변환
        double[] fields = new double[6];
        for (int i = 0; i < 6 && i < txHash.length() - 5; i++) {
            String hex = txHash.substring(i * 5, Math.min(i * 5 + 5, txHash.length()));
            try {
                long value = Long.parseLong(hex, 16);
                fields[i] = (value % 1000) / 1000.0; // 0-1 범위로 정규화
            } catch (NumberFormatException e) {
                fields[i] = 0.5; // 기본값
            }
        }
        return fields;
    }
    
    /**
     * 노드의 네트워크 상 거리 계산 (단순화된 모델)
     */
    private double getNodeDistance(int nodeId, int nodeCount) {
        // 노드 ID에 따라 중앙에서의 거리 계산
        // 실제 구현에서는 네트워크 토폴로지에 따라 다르게 계산해야 함
        return (nodeId % 10) * 0.5 + 0.5; // 0.5-5.0 범위의 거리
    }
    
    /**
     * 거리에 따른 필드 강도 계산
     */
    private double getFieldStrengthAtDistance(double[][] fieldEvolution, double distance) {
        // 필드 강도는 거리에 따라 감소 (역제곱 법칙)
        int timeStep = (int) (distance * 10); // 거리에 따른 시간 지연
        timeStep = Math.min(timeStep, fieldEvolution.length - 1);
        
        // 전기장과 자기장의 총 강도 계산
        double eField = Math.sqrt(
            Math.pow(fieldEvolution[timeStep][0], 2) +
            Math.pow(fieldEvolution[timeStep][1], 2) +
            Math.pow(fieldEvolution[timeStep][2], 2)
        );
        
        double bField = Math.sqrt(
            Math.pow(fieldEvolution[timeStep][3], 2) +
            Math.pow(fieldEvolution[timeStep][4], 2) +
            Math.pow(fieldEvolution[timeStep][5], 2)
        );
        
        // 포인팅 벡터에 비례하는 강도
        return eField * bField;
    }
    
    /**
     * 거리와 필드 강도에 따른 전파 시간 계산
     */
    private double calculatePropagationTime(double distance, double fieldStrength) {
        // 통신 지연은 거리에 비례하고 신호 강도에 반비례
        double baseDelay = 100.0; // 기본 지연 시간 (ms)
        double strengthFactor = Math.max(0.1, fieldStrength); // 너무 작은 값 방지
        return baseDelay * distance / strengthFactor;
    }
    
    /**
     * 매개변수 설정 메서드
     */
    public void setParameters(double permittivity, double permeability, double conductivity) {
        this.permittivity = permittivity;
        this.permeability = permeability;
        this.conductivity = conductivity;
    }
}

/**
 * 블록체인 트랜잭션 데이터 모델
 */
class BlockchainTransaction {
    private long id;
    private long trackingId;
    private String fromAddress;
    private String toAddress;
    private double amount;
    private double confidence;
    private long timestamp;
    
    // Getter와 Setter 메서드
    public long getId() { return id; }
    public void setId(long id) { this.id = id; }
    
    public long getTrackingId() { return trackingId; }
    public void setTrackingId(long trackingId) { this.trackingId = trackingId; }
    
    public String getFromAddress() { return fromAddress; }
    public void setFromAddress(String fromAddress) { this.fromAddress = fromAddress; }
    
    public String getToAddress() { return toAddress; }
    public void setToAddress(String toAddress) { this.toAddress = toAddress; }
    
    public double getAmount() { return amount; }
    public void setAmount(double amount) { this.amount = amount; }
    
    public double getConfidence() { return confidence; }
    public void setConfidence(double confidence) { this.confidence = confidence; }
    
    public long getTimestamp() { return timestamp; }
    public void setTimestamp(long timestamp) { this.timestamp = timestamp; }
    
    @Override
    public String toString() {
        return "Transaction{" +
               "from='" + fromAddress + '\'' +
               ", to='" + toAddress + '\'' +
               ", amount=" + amount +
               ", confidence=" + confidence +
               '}';
    }
}

/**
 * 머신러닝 서비스 인터페이스
 */
interface MachineLearningService {
    /**
     * 트랜잭션 예측 수행
     */
    List<BlockchainTransaction> predictTransactions(List<BlockchainTransaction> baseTransactions, String sourceAddress);
    
    /**
     * 트랜잭션 클러스터링 및 패턴 인식
     */
    Map<String, List<String>> clusterTransactions(List<BlockchainTransaction> transactions);
}

/**
 * 간단한 머신러닝 서비스 구현 (실제로는 외부 Python/TensorFlow 서비스와 통합)
 */
class SimpleMachineLearningService implements MachineLearningService {

    @Override
    public List<BlockchainTransaction> predictTransactions(List<BlockchainTransaction> baseTransactions, String sourceAddress) {
        // 실제 구현에서는 REST API를 통해 ML 서비스 호출
        // 여기서는 예시로 간단한 패턴 기반 예측 수행
        List<BlockchainTransaction> predictions = new ArrayList<>();
        
        // 기존 트랜잭션에서 패턴 인식 및 새 트랜잭션 예측
        double totalAmount = baseTransactions.stream()
                .mapToDouble(BlockchainTransaction::getAmount)
                .sum();
        
        double avgAmount = totalAmount / baseTransactions.size();
        
        // 새로운 트랜잭션 생성
        for (int i = 0; i < 3; i++) {
            BlockchainTransaction newTx = new BlockchainTransaction();
            newTx.setFromAddress(sourceAddress);
            newTx.setToAddress("0x" + Integer.toHexString((sourceAddress.hashCode() + i * 10000) & 0xffffff));
            newTx.setAmount(avgAmount * (0.8 + Math.random() * 0.4)); // 평균의 80-120%
            newTx.setConfidence(0.7 + Math.random() * 0.25); // 70-95% 신뢰도
            newTx.setTimestamp(System.currentTimeMillis());
            
            predictions.add(newTx);
        }
        
        return predictions;
    }

    @Override
    public Map<String, List<String>> clusterTransactions(List<BlockchainTransaction> transactions) {
        // 간단한 금액 기반 클러스터링
        Map<String, List<String>> clusters = new HashMap<>();
        
        // 트랜잭션을 금액 크기에 따라 대략적으로 분류
        clusters.put("small", transactions.stream()
                .filter(tx -> tx.getAmount() < 50)
                .map(BlockchainTransaction::getToAddress)
                .collect(Collectors.toList()));
        
        clusters.put("medium", transactions.stream()
                .filter(tx -> tx.getAmount() >= 50 && tx.getAmount() < 200)
                .map(BlockchainTransaction::getToAddress)
                .collect(Collectors.toList()));
        
        clusters.put("large", transactions.stream()
                .filter(tx -> tx.getAmount() >= 200)
                .map(BlockchainTransaction::getToAddress)
                .collect(Collectors.toList()));
        
        return clusters;
    }
}

/**
 * 블록체인 저장소 인터페이스
 */
interface BlockchainRepository {
    /**
     * 추적 정보 저장
     */
    long saveTracking(Long userId, String address);
    
    /**
     * 트랜잭션 결과 저장
     */
    void saveResults(List<BlockchainTransaction> transactions);
}

/**
 * 메모리 기반 블록체인 저장소 구현 (테스트용)
 */
class InMemoryBlockchainRepository implements BlockchainRepository {
    private long nextTrackingId = 1;
    private long nextResultId = 1;
    private final List<Map<String, Object>> trackings = new ArrayList<>();
    private final List<BlockchainTransaction> results = new ArrayList<>();

    @Override
    public long saveTracking(Long userId, String address) {
        long trackingId = nextTrackingId++;
        Map<String, Object> tracking = new HashMap<>();
        tracking.put("trackingId", trackingId);
        tracking.put("userId", userId);
        tracking.put("address", address);
        tracking.put("trackingDate", System.currentTimeMillis());
        trackings.add(tracking);
        return trackingId;
    }

    @Override
    public void saveResults(List<BlockchainTransaction> transactions) {
        for (BlockchainTransaction tx : transactions) {
            tx.setId(nextResultId++);
            tx.setTimestamp(System.currentTimeMillis());
            results.add(tx);
        }
    }
    
    // 테스트용 메서드
    public List<BlockchainTransaction> getAllResults() {
        return new ArrayList<>(results);
    }
}

/**
 * 블록체인 트랜잭션 추적 서비스
 */
class BlockchainTrackerService {
    
    private final BlockchainRepository blockchainRepository;
    private final LorenzSystem lorenzSystem;
    private final MaxwellPropagationModel maxwellModel;
    private final MachineLearningService mlService;
    
    /**
     * 생성자
     */
    public BlockchainTrackerService(
            BlockchainRepository blockchainRepository,
            LorenzSystem lorenzSystem,
            MaxwellPropagationModel maxwellModel,
            MachineLearningService mlService) {
        this.blockchainRepository = blockchainRepository;
        this.lorenzSystem = lorenzSystem;
        this.maxwellModel = maxwellModel;
        this.mlService = mlService;
    }
    
    /**
     * 블록체인 주소 트랜잭션 추적
     * @param address 블록체인 주소
     * @param depth 추적 깊이
     * @param userId 사용자 ID
     * @return 예측된 트랜잭션 목록
     */
    public List<BlockchainTransaction> trackTransactions(String address, int depth, Long userId) {
        // 추적 정보 저장
        long trackingId = blockchainRepository.saveTracking(userId, address);
        
        // 트랜잭션 데이터를 로렌츠 시스템의 초기 상태로 변환
        double[] initialState = getInitialStateFromAddress(address);
        
        // 로렌츠 시스템 시뮬레이션
        double[][] lorenzTrajectory = lorenzSystem.solve(initialState, depth * 100);
        
        // 리아푸노프 지수 계산으로 카오스 정도 평가
        double lyapunovExponent = lorenzSystem.calculateLyapunovExponent(initialState, depth * 100);
        
        // 결과 데이터 생성
        List<BlockchainTransaction> results = new ArrayList<>();
        
        // 특이점 탐지 및 트랜잭션 생성
        for (int i = 0; i < depth; i++) {
            int idx = i * 100 + 50; // 각 깊이 단계에서 중간 지점
            
            // 로렌츠 시스템의 상태를 트랜잭션 데이터로 변환
            BlockchainTransaction tx = new BlockchainTransaction();
            tx.setTrackingId(trackingId);
            tx.setFromAddress(address);
            
            // 목적지 주소 생성 (예시용 임의 생성)
            tx.setToAddress("0x" + Integer.toHexString(Math.abs((int)(lorenzTrajectory[idx][0] * 1000000))) + 
                         Integer.toHexString(Math.abs((int)(lorenzTrajectory[idx][1] * 1000000))));
            
            // 금액 및 신뢰도 계산
            double zValue = lorenzTrajectory[idx][2];
            tx.setAmount(zValue * 100); // 예시: z 값 기반 금액
            
            // 신뢰도는 리아푸노프 지수와 역상관 관계
            // 카오스가 높을수록 신뢰도 낮음
            double confidence = Math.max(0.1, Math.min(0.95, 1.0 / (1.0 + Math.abs(lyapunovExponent) * 0.1)));
            tx.setConfidence(confidence);
            
            results.add(tx);
        }
        
        // 머신러닝 서비스를 통한 예측 보완
        if (depth > 5) {
            List<BlockchainTransaction> mlPredictions = mlService.predictTransactions(results, address);
            
            // 물리 모델과 ML 모델 결과 통합
            results = integrateResults(results, mlPredictions);
        }
        
        // 결과 저장
        blockchainRepository.saveResults(results);
        
        return results;
    }
    
    /**
     * 트랜잭션 전파 추적
     * @param txHash 트랜잭션 해시
     * @return 전파 시간 예측 맵
     */
    public Map<String, Double> trackTransactionPropagation(String txHash) {
        // 맥스웰 모델을 사용하여 트랜잭션 전파 예측
        double[][] propagationTimes = maxwellModel.predictPropagationTimes(txHash, 10);
        
        Map<String, Double> result = new HashMap<>();
        for (int i = 0; i < propagationTimes.length; i++) {
            String nodeId = "node-" + (int)propagationTimes[i][0];
            double time = propagationTimes[i][1];
            result.put(nodeId, time);
        }
        
        return result;
    }
    
    /**
     * 종합적인 트랜잭션 분석
     * 여러 물리 모델을 조합하여 더 정확한 예측 제공
     */
    public Map<String, Object> trackTransactionsComprehensive(String address, int depth, Long userId) {
        // 기본 추적 수행
        List<BlockchainTransaction> basicResults = trackTransactions(address, depth, userId);
        
        // 맥스웰 모델을 사용한 전파 특성 분석
        Map<String, Double> propagationMap = new HashMap<>();
        for (BlockchainTransaction tx : basicResults) {
            String txHash = generateTxHash(tx);
            Map<String, Double> propTimes = trackTransactionPropagation(txHash);
            double avgPropTime = propTimes.values().stream().mapToDouble(Double::doubleValue).average().orElse(0);
            propagationMap.put(txHash, avgPropTime);
        }
        
        // 머신러닝을 통한 클러스터링 및 패턴 인식
        Map<String, List<String>> clusters = mlService.clusterTransactions(basicResults);
        
        // 결과 통합
        Map<String, Object> result = new HashMap<>();
        result.put("transactions", basicResults);
        result.put("propagationTimes", propagationMap);
        result.put("clusters", clusters);
        result.put("lyapunovExponent", lorenzSystem.calculateLyapunovExponent(
                                            getInitialStateFromAddress(address), depth * 100));
        
        return result;
    }
    
    /**
     * 블록체인 주소를 로렌츠 시스템 초기 상태로 변환
     */
    private double[] getInitialStateFromAddress(String address) {
        double[] initialState = new double[3];
        
        // 주소 해시 문자를 숫자로 변환하여 초기값 생성
        int hashCode = address.hashCode();
        initialState[0] = 1.0 + (hashCode % 1000) / 1000.0;
        initialState[1] = 1.0 + ((hashCode / 1000) % 1000) / 1000.0;
        initialState[2] = 1.0 + ((hashCode / 1000000) % 1000) / 1000.0;
        
        return initialState;
    }
    
    /**
     * 물리 모델과 머신러닝 모델 결과 통합
     */
    private List<BlockchainTransaction> integrateResults(List<BlockchainTransaction> physicsResults, 
                                                        List<BlockchainTransaction> mlResults) {
        List<BlockchainTransaction> integrated = new ArrayList<>(physicsResults);
        
        // ML 결과에서 높은 신뢰도를 가진 결과만 추가
        for (BlockchainTransaction mlTx : mlResults) {
            if (mlTx.getConfidence() > 0.7) {
                boolean exists = false;
                
                // 이미 존재하는 트랜잭션인지 확인
                for (BlockchainTransaction tx : physicsResults) {
                    if (tx.getToAddress().equals(mlTx.getToAddress())) {
                        exists = true;
                        break;
                    }
                }
                
                // 새로운 트랜잭션이면 추가
                if (!exists) {
                    integrated.add(mlTx);
                }
            }
        }
        
        return integrated;
    }
    
    /**
     * 트랜잭션 해시 생성
     */
    private String generateTxHash(BlockchainTransaction tx) {
        String data = tx.getFromAddress() + tx.getToAddress() + tx.getAmount();
        return sha256(data);
    }
    
    /**
     * SHA-256 해시 함수
     */
    private String sha256(String data) {
        try {
            java.security.MessageDigest digest = java.security.MessageDigest.getInstance("SHA-256");
            byte[] hash = digest.digest(data.getBytes("UTF-8"));
            StringBuilder hexString = new StringBuilder();
            
            for (byte b : hash) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1) hexString.append('0');
                hexString.append(hex);
            }
            
            return hexString.toString();
        } catch (Exception e) {
            throw new RuntimeException("SHA-256 해싱 오류", e);
        }
    }
}

/**
 * 메인 데모 클래스
 */
public class BlockchainTrackingSystem {
    public static void main(String[] args) {
        // 서비스 구성 요소 초기화
        LorenzSystem lorenzSystem = new LorenzSystem();
        MaxwellPropagationModel maxwellModel = new MaxwellPropagationModel();
        MachineLearningService mlService = new SimpleMachineLearningService();
        BlockchainRepository repository = new InMemoryBlockchainRepository();
        
        // 트랜잭션 추적 서비스 생성
        BlockchainTrackerService trackerService = new BlockchainTrackerService(
            repository, lorenzSystem, maxwellModel, mlService
        );
        
        // 샘플 블록체인 주소
        String blockchainAddress = "0x742d35Cc6634C0532925a3b844Bc454e4438f44e";
        
        // 트랜잭션 추적 실행
        System.out.println("=== 블록체인 트랜잭션 추적 시작 ===");
        System.out.println("주소: " + blockchainAddress);
        System.out.println();
        
        List<BlockchainTransaction> transactions = trackerService.trackTransactions(blockchainAddress, 10, 1L);
        
        System.out.println("=== 예측된 트랜잭션 ===");
        for (BlockchainTransaction tx : transactions) {
            System.out.printf("From: %s, To: %s, Amount: %.2f, Confidence: %.2f%%\n",
                tx.getFromAddress(),
                tx.getToAddress(),
                tx.getAmount(),
                tx.getConfidence() * 100
            );
        }
        
        System.out.println();
        System.out.println("=== 전파 시간 분석 ===");
        String txHash = trackerService.generateTxHash(transactions.get(0));
        Map<String, Double> propagationTimes = trackerService.trackTransactionPropagation(txHash);
        
        for (Map.Entry<String, Double> entry : propagationTimes.entrySet()) {
            System.out.printf("%s: %.2f ms\n", entry.getKey(), entry.getValue());
        }
        
        System.out.println();
        System.out.println("=== 종합 분석 ===");
        Map<String, Object> comprehensiveResult = trackerService.trackTransactionsComprehensive(blockchainAddress, 10, 1L);
        System.out.println("리아푸노프 지수: " + comprehensiveResult.get("lyapunovExponent"));
        
        @SuppressWarnings("unchecked")
        Map<String, List<String>> clusters = (Map<String, List<String>>) comprehensiveResult.get("clusters");
        
        for (Map.Entry<String, List<String>> entry : clusters.entrySet()) {
            System.out.println("클러스터 [" + entry.getKey() + "]: " + entry.getValue().size() + "개 주소");
        }
    }
}