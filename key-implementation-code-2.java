// 1. 나비에-스토크스 시뮬레이터
package com.blocktracker.service.physics;

import org.springframework.stereotype.Component;

@Component
public class NavierStokesSimulator {
    // 시뮬레이션 매개변수
    private double viscosity = 0.01; // 점성 계수 (μ)
    private double density = 1.0; // 밀도 (ρ)
    private int resolution = 128; // 격자 해상도
    private double gridSize = 0.01; // km 단위

    /**
     * 지정된 위치 주변의 유체 흐름 필드를 생성
     * 현실 세계 좌표를 시뮬레이션 격자로 변환하여 계산
     *
     * ρ(∂v/∂t + v·∇v) = -∇p + μ∇²v + F
     * ∇·v = 0 (비압축성 조건)
     */
    public double[][] generateFlowField(double longitude, double latitude, double heading) {
        // 격자 초기화
        double[][] velocityX = new double[resolution][resolution];
        double[][] velocityY = new double[resolution][resolution];
        double[][] pressure = new double[resolution][resolution];
        
        // 중심점을 현재 위치로 설정
        int centerI = resolution / 2;
        int centerJ = resolution / 2;
        
        // 초기 속도 설정 (차량 진행 방향 기준)
        initializeVelocities(velocityX, velocityY, heading);
        
        // 나비에-스토크스 방정식 시뮬레이션 (여러 시간 단계에 걸쳐 해결)
        for (int step = 0; step < 20; step++) {
            // 1. 확산 단계 (점성 효과)
            diffuse(velocityX, velocityY);
            
            // 2. 압력 계산 및 투영 (비압축성 조건 유지)
            project(velocityX, velocityY, pressure);
            
            // 3. 이류 단계 (흐름에 따른 속도 변화)
            advect(velocityX, velocityY);
        }
        
        // 유체 흐름 벡터 필드와 환경 특성을 결합한 결과 생성
        double[][] flowField = new double[resolution * resolution][5]; // x, y, vx, vy, pressure
        int index = 0;
        
        for (int i = 0; i < resolution; i++) {
            for (int j = 0; j < resolution; j++) {
                // 격자 위치를 실제 좌표로 변환 (중심점 기준)
                double x = (i - centerI) * gridSize;
                double y = (j - centerJ) * gridSize;
                
                // 결과 배열에 저장
                flowField[index][0] = x;
                flowField[index][1] = y;
                flowField[index][2] = velocityX[i][j];
                flowField[index][3] = velocityY[i][j];
                flowField[index][4] = pressure[i][j];
                
                index++;
            }
        }
        
        return flowField;
    }
    
    /**
     * 초기 속도 벡터 설정
     */
    private void initializeVelocities(double[][] velocityX, double[][] velocityY, double heading) {
        // 방향을 라디안으로 변환
        double radians = Math.toRadians(heading);
        double vx = Math.cos(radians);
        double vy = Math.sin(radians);
        
        int centerI = resolution / 2;
        int centerJ = resolution / 2;
        
        // 주행 방향에 따라 속도 초기화
        for (int i = 0; i < resolution; i++) {
            for (int j = 0; j < resolution; j++) {
                // 중심에서 거리에 따른 감쇠
                double distSq = Math.pow(i - centerI, 2) + Math.pow(j - centerJ, 2);
                double factor = Math.exp(-distSq / (resolution * resolution / 4.0));
                
                velocityX[i][j] = vx * factor;
                velocityY[i][j] = vy * factor;
            }
        }
    }
    
    /**
     * 확산 과정 (점성에 의한 속도 확산)
     *
     * ∂v/∂t = μ∇²v
     */
    private void diffuse(double[][] velocityX, double[][] velocityY) {
        double[][] tempX = new double[resolution][resolution];
        double[][] tempY = new double[resolution][resolution];
        
        // 현재 속도 복사
        for (int i = 0; i < resolution; i++) {
            System.arraycopy(velocityX[i], 0, tempX[i], 0, resolution);
            System.arraycopy(velocityY[i], 0, tempY[i], 0, resolution);
        }
        
        double alpha = viscosity;
        double beta = 1.0 / (1.0 + 4.0 * alpha);
        
        // Gauss-Seidel 이완법으로 확산 방정식 해결
        for (int k = 0; k < 20; k++) {
            for (int i = 1; i < resolution - 1; i++) {
                for (int j = 1; j < resolution - 1; j++) {
                    velocityX[i][j] = beta * (
                        tempX[i][j] + alpha * (
                            velocityX[i+1][j] + velocityX[i-1][j] +
                            velocityX[i][j+1] + velocityX[i][j-1]
                        )
                    );
                    
                    velocityY[i][j] = beta * (
                        tempY[i][j] + alpha * (
                            velocityY[i+1][j] + velocityY[i-1][j] +
                            velocityY[i][j+1] + velocityY[i][j-1]
                        )
                    );
                }
            }
            
            setBoundary(velocityX, velocityY);
        }
    }
    
    /**
     * 경계 조건 설정 (속도 경계)
     */
    private void setBoundary(double[][] velocityX, double[][] velocityY) {
        // 벽에서 속도 = 0 (no-slip 조건)
        for (int i = 0; i < resolution; i++) {
            velocityX[i][0] = velocityY[i][0] = 0;
            velocityX[i][resolution-1] = velocityY[i][resolution-1] = 0;
        }
        
        for (int j = 0; j < resolution; j++) {
            velocityX[0][j] = velocityY[0][j] = 0;
            velocityX[resolution-1][j] = velocityY[resolution-1][j] = 0;
        }
    }
    
    /**
     * 압력 경계 조건 설정
     */
    private void setBoundaryScalar(double[][] scalar) {
        // 경계에서 내부 값 복사 (Neumann 조건)
        for (int i = 1; i < resolution - 1; i++) {
            scalar[i][0] = scalar[i][1];
            scalar[i][resolution-1] = scalar[i][resolution-2];
        }
        
        for (int j = 1; j < resolution - 1; j++) {
            scalar[0][j] = scalar[1][j];
            scalar[resolution-1][j] = scalar[resolution-2][j];
        }
        
        // 모서리 값 처리
        scalar[0][0] = 0.5 * (scalar[1][0] + scalar[0][1]);
        scalar[0][resolution-1] = 0.5 * (scalar[1][resolution-1] + scalar[0][resolution-2]);
        scalar[resolution-1][0] = 0.5 * (scalar[resolution-2][0] + scalar[resolution-1][1]);
        scalar[resolution-1][resolution-1] = 0.5 * (scalar[resolution-2][resolution-1] + scalar[resolution-1][resolution-2]);
    }
    
    /**
     * 이류 단계 (속도에 의한 물질 이동)
     *
     * ∂v/∂t + (v·∇)v = 0
     */
    private void advect(double[][] velocityX, double[][] velocityY) {
        double[][] tempX = new double[resolution][resolution];
        double[][] tempY = new double[resolution][resolution];
        
        // 현재 속도 복사
        for (int i = 0; i < resolution; i++) {
            System.arraycopy(velocityX[i], 0, tempX[i], 0, resolution);
            System.arraycopy(velocityY[i], 0, tempY[i], 0, resolution);
        }
        
        double dt = 0.1; // 시간 단계
        
        for (int i = 1; i < resolution - 1; i++) {
            for (int j = 1; j < resolution - 1; j++) {
                // 역추적하여 이전 위치 계산
                double x = i - dt * tempX[i][j];
                double y = j - dt * tempY[i][j];
                
                // 경계 안에 있도록
                x = Math.max(0.5, Math.min(resolution - 1.5, x));
                y = Math.max(0.5, Math.min(resolution - 1.5, y));
                
                // 격자점 계산
                int i0 = (int) x;
                int i1 = i0 + 1;
                int j0 = (int) y;
                int j1 = j0 + 1;
                
                // 보간 가중치
                double s1 = x - i0;
                double s0 = 1 - s1;
                double t1 = y - j0;
                double t0 = 1 - t1;
                
                // 쌍선형 보간으로 속도 업데이트
                velocityX[i][j] = s0 * (t0 * tempX[i0][j0] + t1 * tempX[i0][j1]) +
                                s1 * (t0 * tempX[i1][j0] + t1 * tempX[i1][j1]);
                
                velocityY[i][j] = s0 * (t0 * tempY[i0][j0] + t1 * tempY[i0][j1]) +
                                s1 * (t0 * tempY[i1][j0] + t1 * tempY[i1][j1]);
            }
        }
        
        setBoundary(velocityX, velocityY);
    }
    
    /**
     * 비압축성 조건 투영 단계
     *
     * ∇·v = 0
     */
    private void project(double[][] velocityX, double[][] velocityY, double[][] pressure) {
        double[][] divergence = new double[resolution][resolution];
        
        // 1. 속도의 발산 계산
        for (int i = 1; i < resolution - 1; i++) {
            for (int j = 1; j < resolution - 1; j++) {
                divergence[i][j] = -0.5 * (
                    velocityX[i+1][j] - velocityX[i-1][j] +
                    velocityY[i][j+1] - velocityY[i][j-1]
                ) / resolution;
                
                pressure[i][j] = 0;
            }
        }
        
        setBoundaryScalar(divergence);
        setBoundaryScalar(pressure);
        
        // 2. 압력 푸아송 방정식 해결
        for (int k = 0; k < 20; k++) {
            for (int i = 1; i < resolution - 1; i++) {
                for (int j = 1; j < resolution - 1; j++) {
                    pressure[i][j] = (
                        divergence[i][j] +
                        pressure[i+1][j] + pressure[i-1][j] +
                        pressure[i][j+1] + pressure[i][j-1]
                    ) / 4;
                }
            }
            
            setBoundaryScalar(pressure);
        }
        
        // 3. 압력 구배를 속도에서 뺌
        for (int i = 1; i < resolution - 1; i++) {
            for (int j = 1; j < resolution - 1; j++) {
                velocityX[i][j] -= 0.5 * (pressure[i+1][j] - pressure[i-1][j]) * resolution;
                velocityY[i][j] -= 0.5 * (pressure[i][j+1] - pressure[i][j-1]) * resolution;
            }
        }
        
        setBoundary(velocityX, velocityY);
    }
}

// 2. 자율주행차 위치 추적 서비스
package com.blocktracker.service.physics;

import com.blocktracker.model.VehiclePosition;
import com.blocktracker.model.RiskZone;
import com.blocktracker.repository.VehicleRepository;
import com.blocktracker.service.ml.PathPredictor;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;
import java.util.List;
import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;

@Service
public class AutonomousVehicleTracker {
    
    @Autowired
    private NavierStokesSimulator navierStokes;
    
    @Autowired
    private DuffingOscillator duffingModel;
    
    @Autowired
    private PathPredictor pathPredictor;
    
    @Autowired
    private VehicleRepository vehicleRepository;
    
    /**
     * 자율주행차 경로 예측
     * @param vehicleId 차량 ID
     * @param currentPosition 현재 위치
     * @param steps 예측 단계 수
     * @param userId 사용자 ID (null 가능)
     * @return 예측된 위치 목록
     */
    public List<VehiclePosition> predictVehiclePath(String vehicleId, VehiclePosition currentPosition, 
                                                  int steps, Long userId) {
        // 추적 정보 저장 (사용자 ID가 제공된 경우)
        long trackingId = -1;
        if (userId != null) {
            trackingId = vehicleRepository.saveTracking(userId, vehicleId);
            currentPosition.setTrackingId(trackingId);
            vehicleRepository.savePosition(currentPosition);
        }
        
        // 나비에-스토크스 방정식을 사용하여 유체 흐름 필드 생성
        double[][] flowField = navierStokes.generateFlowField(
            currentPosition.getLongitude(), 
            currentPosition.getLatitude(), 
            currentPosition.getHeading()
        );
        
        // 물리 모델과 머신러닝을 결합한 경로 예측
        List<VehiclePosition> predictedPath;
        
        try {
            // 머신러닝 서비스를 사용한 경로 예측 시도
            predictedPath = pathPredictor.predictPath(vehicleId, currentPosition, flowField, steps);
        } catch (Exception e) {
            // ML 서비스 실패 시 간단한 물리 모델만으로 예측
            predictedPath = generateSimplePath(currentPosition, steps);
        }
        
        // 예측 결과에 추적 ID 설정 및 저장
        if (trackingId != -1) {
            for (VehiclePosition position : predictedPath) {
                position.setTrackingId(trackingId);
            }
            
            vehicleRepository.savePositions(predictedPath);
        }
        
        return predictedPath;
    }
    
    /**
     * 간단한 물리 모델 기반 경로 생성 (백업 방법)
     */
    private List<VehiclePosition> generateSimplePath(VehiclePosition currentPosition, int steps) {
        List<VehiclePosition> path = new ArrayList<>();
        
        // 현재 위치 및 방향
        double lat = currentPosition.getLatitude();
        double lon = currentPosition.getLongitude();
        double heading = currentPosition.getHeading();
        double speed = currentPosition.getSpeed();
        
        // 지구의 반경 (km)
        final double EARTH_RADIUS = 6371.0;
        
        // 각 시간 단계별 위치 계산
        for (int i = 1; i <= steps; i++) {
            // 시간 간격 (초)
            double dt = 2.0;
            
            // 이동 거리 (km)
            double distance = speed * dt / 3600.0; // km/h를 km/s로 변환 후 거리 계산
            
            // 위도, 경도 변화량 계산 (근사값)
            double headingRad = Math.toRadians(heading);
            double latChange = distance * Math.cos(headingRad) / EARTH_RADIUS * Math.toDegrees(Math.PI);
            double lonChange = distance * Math.sin(headingRad) / (EARTH_RADIUS * Math.cos(Math.toRadians(lat))) * Math.toDegrees(Math.PI);
            
            // 새 위치 계산
            lat += latChange;
            lon += lonChange;
            
            // 약간의 무작위성 추가 (실제 주행 환경 시뮬레이션)
            heading += (Math.random() - 0.5) * 2.0; // ±1도 변화
            speed += (Math.random() - 0.5) * 2.0; // ±1km/h 변화
            
            // 새 위치를 경로에 추가
            VehiclePosition pos = new VehiclePosition();
            pos.setLatitude(lat);
            pos.setLongitude(lon);
            pos.setHeading(heading);
            pos.setSpeed(speed);
            pos.setPredicted(true);
            pos.setTrackingId(currentPosition.getTrackingId());
            
            path.add(pos);
        }
        
        return path;
    }
    
    /**
     * 위험 지점 예측
     * @param position 현재 위치
     * @param roadCondition 도로 상태 (0-1)
     * @param weatherFactor 날씨 상태 (0-1)
     * @return 예측된 위험 지점 목록
     */
    public List<RiskZone> predictRiskZones(VehiclePosition position, double roadCondition, double weatherFactor) {
        // 더핑 진동자 매개변수 설정
        // 도로 상태와 날씨에 따라 매개변수 조정
        double k = 0.2 + 0.3 * roadCondition; // 강성 계수
        double alpha = 1.0; // 비선형 항 계수
        double beta = 0.3 + 0.4 * weatherFactor; // 감쇠 계수
        double gamma = 0.5; // 외력 진폭
        
        // 더핑 진동자 모델을 사용하여 위험 지점 계산
        List<RiskZone> riskZones = new ArrayList<>();
        
        try {
            // ML 서비스를 통한 위험 지점 예측
            riskZones = pathPredictor.calculateRiskZones(position, k, alpha, beta, gamma);
        } catch (Exception e) {
            // ML 서비스 실패 시 간단한 위험 지점 생성
            riskZones = generateSimpleRiskZones(position, roadCondition, weatherFactor);
        }
        
        // 위험 지점에 추적 ID 설정
        if (position.getTrackingId() != null && position.getTrackingId() > 0) {
            for (RiskZone zone : riskZones) {
                zone.setTrackingId(position.getTrackingId());
            }
            
            // 위험 지점 저장
            vehicleRepository.saveRiskZones(riskZones);
        }
        
        return riskZones;
    }
    
    /**
     * 간단한 위험 지점 생성 (백업 방법)
     */
    private List<RiskZone> generateSimpleRiskZones(VehiclePosition position, double roadCondition, double weatherFactor) {
        List<RiskZone> riskZones = new ArrayList<>();
        
        // 현재 위치 및 방향
        double lat = position.getLatitude();
        double lon = position.getLongitude();
        double heading = position.getHeading();
        
        // 환경 요소에 따른 기본 위험도
        double baseRisk = 0.2 + 0.3 * roadCondition + 0.5 * weatherFactor;
        baseRisk = Math.min(0.9, baseRisk); // 최대 0.9
        
        // 진행 방향으로 여러 위험 지점 추가
        for (int i = 0; i < 3; i++) {
            // 거리를 증가시키며 위험도 감소
            double distance = 0.05 + i * 0.1; // 50m, 150m, 250m
            double riskLevel = baseRisk * Math.exp(-i * 0.5); // 지수 감소
            
            double headingRad = Math.toRadians(heading);
            double latChange = distance * Math.cos(headingRad) / 111.0; // 1도 약 111km
            double lonChange = distance * Math.sin(headingRad) / (111.0 * Math.cos(Math.toRadians(lat)));
            
            RiskZone zone = new RiskZone();
            zone.setLatitude(lat + latChange);
            zone.setLongitude(lon + lonChange);
            zone.setRiskLevel(riskLevel);
            zone.setDescription("위험 지점 " + (i+1) + " - 거리: " + (distance * 1000) + "m");
            
            riskZones.add(zone);
        }
        
        return riskZones;
    }
    
    /**
     * 위험 경고 전송
     * @param vehicleId 차량 ID
     * @param riskZone 위험 지점 정보
     * @return 전송 성공 여부
     */
    public boolean sendWarning(String vehicleId, RiskZone riskZone) {
        // 사용자 ID 조회
        Long userId = vehicleRepository.getUserIdByVehicleId(vehicleId);
        
        // 로그 저장
        if (userId != null) {
            vehicleRepository.logWarning(
                userId, 
                vehicleId, 
                riskZone.getRiskLevel(), 
                riskZone.getLatitude(), 
                riskZone.getLongitude(), 
                "0.0.0.0" // 실제 구현에서는 클라이언트 IP 사용
            );
        }
        
        // 실제 구현에서는 웹소켓 또는 푸시 알림 등으로 경고 전송
        // 여기서는 로그만 저장하고 성공 반환
        return true;
    }
    
    /**
     * 종합적인 위험 분석
     */
    public Map<String, Object> analyzeRiskComprehensive(String vehicleId, VehiclePosition position, 
                                                     double roadCondition, double weatherFactor, Long userId) {
        // 추적 정보 저장
        long trackingId = vehicleRepository.saveTracking(userId, vehicleId);
        position.setTrackingId(trackingId);
        vehicleRepository.savePosition(position);
        
        // 경로 예측
        List<VehiclePosition> predictedPath = predictVehiclePath(vehicleId, position, 10, null);
        
        // 위험 지점 예측
        List<RiskZone> riskZones = predictRiskZones(position, roadCondition, weatherFactor);
        
        // 충돌 위험 분석
        List<Map<String, Object>> collisionRisks = analyzeCollisionRisks(vehicleId, position, predictedPath);
        
        // 결과 통합
        Map<String, Object> result = new HashMap<>();
        result.put("vehicleId", vehicleId);
        result.put("currentPosition", position);
        result.put("predictedPath", predictedPath);
        result.put("riskZones", riskZones);
        result.put("collisionRisks", collisionRisks);
        result.put("environmentalFactors", Map.of("roadCondition", roadCondition, "weatherFactor", weatherFactor));
        
        return result;
    }
    
    /**
     * 충돌 위험 분석
     */
    private List<Map<String, Object>> analyzeCollisionRisks(String vehicleId, VehiclePosition position, 
                                                         List<VehiclePosition> predictedPath) {
        List<Map<String, Object>> collisionRisks = new ArrayList<>();
        
        // 주변 차량 검색 (실제 구현에서는 DB에서 조회)
        double searchRadius = 1.0; // 1km 반경
        List<Map<String, Object>> nearbyVehicles = vehicleRepository.findNearbyVehicles(
            position.getLatitude(), position.getLongitude(), searchRadius, vehicleId
        );
        
        // 각 차량마다 충돌 가능성 분석
        for (Map<String, Object> nearbyVehicle : nearbyVehicles) {
            String otherVehicleId = (String) nearbyVehicle.get("vehicleId");
            double latitude = (double) nearbyVehicle.get("latitude");
            double longitude = (double) nearbyVehicle.get("longitude");
            double heading = (double) nearbyVehicle.get("heading");
            double speed = (double) nearbyVehicle.get("speed");
            
            // 다른 차량의 위치 객체 생성
            VehiclePosition otherPosition = new VehiclePosition();
            otherPosition.setLatitude(latitude);
            otherPosition.setLongitude(longitude);
            otherPosition.setHeading(heading);
            otherPosition.setSpeed(speed);
            
            // 다른 차량의 경로 예측
            List<VehiclePosition> otherPath = predictVehiclePath(otherVehicleId, otherPosition, 10, null);
            
            // 경로 교차점 분석
            Map<String, Object> intersectionAnalysis = analyzePathIntersection(predictedPath, otherPath);
            
            // 충돌 가능성이 있는 경우에만 결과에 추가
            if ((boolean) intersectionAnalysis.get("willIntersect")) {
                Map<String, Object> collisionRisk = new HashMap<>(intersectionAnalysis);
                collisionRisk.put("otherVehicleId", otherVehicleId);
                collisionRisks.add(collisionRisk);
            }
        }
        
        return collisionRisks;
    }
    
    /**
     * 두 경로의 교차점 분석
     */
    private Map<String, Object> analyzePathIntersection(List<VehiclePosition> path1, List<VehiclePosition> path2) {
        Map<String, Object> result = new HashMap<>();
        result.put("willIntersect", false);
        
        double minDistance = Double.MAX_VALUE;
        int minDistanceIndex1 = -1;
        int minDistanceIndex2 = -1;
        
        // 두 경로에서 가장 가까운 지점 찾기
        for (int i = 0; i < path1.size(); i++) {
            for (int j = 0; j < path2.size(); j++) {
                double distance = calculateDistance(
                    path1.get(i).getLatitude(), path1.get(i).getLongitude(),
                    path2.get(j).getLatitude(), path2.get(j).getLongitude()
                );
                
                if (distance < minDistance) {
                    minDistance = distance;
                    minDistanceIndex1 = i;
                    minDistanceIndex2 = j;
                }
            }
        }
        
        // 교차점 임계값 (km)
        double intersectionThreshold = 0.02; // 20m
        
        if (minDistance < intersectionThreshold) {
            result.put("willIntersect", true);
            result.put("intersectionLat", (path1.get(minDistanceIndex1).getLatitude() +
                                          path2.get(minDistanceIndex2).getLatitude()) / 2);
            result.put("intersectionLng", (path1.get(minDistanceIndex1).getLongitude() +
                                          path2.get(minDistanceIndex2).getLongitude()) / 2);
            result.put("distance", minDistance * 1000); // m 단위로 변환
            
            // 충돌 위험도 계산 (거리가 가까울수록, 속도가 높을수록 위험)
            double speed1 = path1.get(minDistanceIndex1).getSpeed();
            double speed2 = path2.get(minDistanceIndex2).getSpeed();
            double avgSpeed = (speed1 + speed2) / 2;
            
            // 최대값이 1이 되도록 정규화된 위험도
            double collisionRisk = Math.min(1.0, (1.0 - minDistance / intersectionThreshold) *
                                        (0.5 + avgSpeed / 200.0));
            result.put("collisionRisk", collisionRisk);
            
            // 교차점까지 예상 시간 (초)
            double timeUntilIntersection = (minDistanceIndex1 * 2) * 1.0; // 각 단계는 약 2초로 가정
            result.put("timeUntilIntersection", timeUntilIntersection);
        }
        
        return result;
    }
    
    /**
     * 두 지점 간의 거리 계산 (하버사인 공식)
     */
    private double calculateDistance(double lat1, double lon1, double lat2, double lon2) {
        final int R = 6371; // 지구 반경 (km)
        
        double latDistance = Math.toRadians(lat2 - lat1);
        double lonDistance = Math.toRadians(lon2 - lon1);
        double a = Math.sin(latDistance / 2) * Math.sin(latDistance / 2)
                + Math.cos(Math.toRadians(lat1)) * Math.cos(Math.toRadians(lat2))
                * Math.sin(lonDistance / 2) * Math.sin(lonDistance / 2);
        double c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        double distance = R * c;
        
        return distance;
    }
}