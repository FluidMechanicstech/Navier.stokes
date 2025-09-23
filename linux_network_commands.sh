#!/bin/bash
# 수식-품사 모델 기반 네트워크 보안 명령어 시스템
# 각 품사별로 고유한 명령어를 사용하여 겹치지 않게 구성

echo "=== 네트워크 보안 수식-품사 모델 구현 ==="
echo "∂ρv/∂t + (v·∇)v + ρ(∂e/∂t)∇v + ρ∇Φ + ∇p = μ∇²v + F_friction + J×B + F_bio"
echo

# ==========================================
# ∇p (보안 정책 구배) - 조사 역할 - TEE 보호 영역
# ==========================================
echo "1. ∇p (보안 정책/조사) - 핵심 보안 제어"
echo "----------------------------------------"

# 방화벽 정책 설정 (조사의 핵심 역할)
sudo iptables -A INPUT -p tcp --dport 22 -j ACCEPT  # SSH는 허용한다
sudo iptables -A INPUT -p tcp --dport 80 -j ACCEPT  # HTTP는 허용한다  
sudo iptables -A INPUT -p tcp --dport 443 -j ACCEPT # HTTPS는 허용한다
sudo iptables -A INPUT -j DROP                      # 나머지는 차단한다

# 보안 정책 상태 확인
sudo iptables -L -n | grep -E "ACCEPT|DROP|REJECT"
echo "보안 정책(조사)이 트래픽을 제어하고 있습니다."
echo

# ==========================================
# IPv4 주소 - 명사 (주체/객체)
# ==========================================
echo "2. IPv4 주소 - 명사 (통신 주체/객체)"
echo "------------------------------------"

# 호스트 정보 확인 (명사 식별)
hostname -I | awk '{print "현재 호스트(명사): " $1}'
who am i | awk '{print "사용자(명사): " $1}'

# 네트워크 인터페이스 확인
ip addr show | grep "inet " | awk '{print "인터페이스(명사): " $2}'

# 활성 연결의 주체와 객체 확인
netstat -tuln | grep LISTEN | awk '{print "서비스(명사) " $4 "에서 대기중"}'
echo

# ==========================================
# 서브넷 마스크 - 형용사 (범위 한정)
# ==========================================
echo "3. 서브넷 마스크 - 형용사 (네트워크 범위)"
echo "---------------------------------------"

# 네트워크 범위 정보 (형용사로 크기 표현)
ip route show | grep -E "/24|/16|/8" | while read route; do
    case "$route" in
        *"/24"*) echo "소규모(형용사) 네트워크: $route" ;;
        *"/16"*) echo "중규모(형용사) 네트워크: $route" ;;
        *"/8"*)  echo "대규모(형용사) 네트워크: $route" ;;
        *)       echo "사용자정의(형용사) 네트워크: $route" ;;
    esac
done
echo

# ==========================================
# TCP/UDP/ICMP - 동사 (네트워크 행위)
# ==========================================
echo "4. 프로토콜 - 동사 (네트워크 행위)"
echo "--------------------------------"

# TCP 연결 행위 모니터링
ss -t | grep ESTAB | wc -l | awk '{print "TCP가 " $1 "개 연결을 수행하고있다(동사)"}'

# UDP 통신 행위 확인  
ss -u | grep -v State | wc -l | awk '{print "UDP가 " $1 "개 소켓을 열어두고있다(동사)"}'

# ICMP 핑 행위 수행
if ping -c 1 8.8.8.8 >/dev/null 2>&1; then
    echo "ICMP가 구글DNS에 성공적으로 응답요청했다(동사)"
else
    echo "ICMP가 구글DNS 응답요청에 실패했다(동사)"
fi

# 포트 스캔 탐지 (의심스러운 행위)
netstat -tuln | grep ":22 " >/dev/null && echo "SSH가 22번포트에서 대기한다(동사)"
echo

# ==========================================
# ARP - 접속사 (Layer 2-3 연결)
# ==========================================
echo "5. ARP - 접속사 (물리-논리 주소 연결)"
echo "-----------------------------------"

# ARP 테이블로 연결 관계 확인
arp -a | head -5 | while read entry; do
    echo "ARP가 $entry 그리고 MAC주소를 연결한다(접속사)"
done

# 새로운 ARP 엔트리 확인
arping -c 1 $(ip route | grep default | awk '{print $3}') 2>/dev/null | \
grep "reply" | awk '{print "ARP가 게이트웨이와 MAC를 연결했다(접속사)"}'
echo

# ==========================================
# DNS - 관사 (도메인 지정)
# ==========================================
echo "6. DNS - 관사 (도메인 이름 특정)"
echo "-------------------------------"

# 도메인 이름 해석 (특정 지정)
nslookup google.com | grep "Address:" | tail -1 | \
awk '{print "DNS가 그(관사) google.com을 " $2 "로 지정했다"}'

# 역방향 DNS 조회
dig -x 8.8.8.8 +short | head -1 | \
awk '{print "DNS가 그(관사) 8.8.8.8을 " $1 "로 지정했다"}'

# 로컬 DNS 캐시 확인
systemd-resolve --status | grep "DNS Servers:" | \
awk '{print "시스템이 그(관사) " $3 "을 DNS서버로 지정했다"}'
echo

# ==========================================
# 스위치 - 부사 (전달 방식)
# ==========================================
echo "7. 스위치 - 부사 (데이터 전달 방식)"
echo "--------------------------------"

# 브리지 정보 확인 (스위칭 방식)
brctl show 2>/dev/null | grep -v "bridge name" | while read bridge; do
    echo "브리지가 $bridge 를 효율적으로(부사) 전달한다"
done

# VLAN 태깅 확인
ip link show | grep -E "vlan|@" | awk '{print "VLAN이 " $2 "을 선택적으로(부사) 분리한다"}'

# MAC 주소 테이블 시뮬레이션
ip neigh show | head -3 | awk '{print "스위치가 " $1 "을 신속하게(부사) 전달한다"}'
echo

# ==========================================
# 라우터 - 전치사 (네트워크 간 위치)
# ==========================================
echo "8. 라우터 - 전치사 (네트워크 위치 관계)"
echo "------------------------------------"

# 라우팅 테이블로 위치 관계 확인
ip route show | while read route; do
    case "$route" in
        "default"*) echo "라우터가 모든트래픽을 $route 를통해(전치사) 전달한다" ;;
        *"dev"*)    echo "라우터가 로컬트래픽을 $route 에서(전치사) 처리한다" ;;
    esac
done

# 경로 추적으로 위치 관계 확인
echo "라우터들의 위치 관계 추적:"
traceroute -m 3 8.8.8.8 2>/dev/null | grep -E "^ [1-3]" | \
awk '{print "패킷이 " $2 "를 거쳐(전치사) " $3 "으로(전치사) 이동했다"}'
echo

# ==========================================
# 수식의 나머지 8개 항 - 일반 데이터 처리
# ==========================================
echo "9. 나머지 8개 항 - 일반 네트워크 데이터 처리"
echo "--------------------------------------------"

echo "∂ρv/∂t (패킷 밀도 변화):"
iostat -n 1 1 | grep -E "eth|ens|enp" | awk '{print "인터페이스 " $1 ": 초당 " $3 " 패킷 수신"}'

echo "(v·∇)v (대역폭 흐름):"
vnstat -i $(ip route | grep default | grep -o 'dev [^ ]*' | awk '{print $2}') --oneline 2>/dev/null | \
awk -F';' '{print "오늘 다운로드: " $4 ", 업로드: " $5}'

echo "ρ(∂e/∂t)∇v (지연 변화):"
ping -c 3 8.8.8.8 2>/dev/null | tail -1 | awk -F'/' '{print "평균 지연시간: " $5 " ms"}'

echo "ρ∇Φ (라우팅 압력):"
ip route show table all | wc -l | awk '{print "총 라우팅 엔트리: " $1 " 개"}'

echo "μ∇²v (네트워크 점성/오버헤드):"
cat /proc/net/dev | grep -E "eth|ens|enp" | awk '{print $1 " 오류패킷: " $4}'

echo "F_friction (충돌 및 손실):"
cat /proc/net/snmp | grep Tcp: | tail -1 | awk '{print "TCP 재전송: " $13 " 개"}'

echo "J×B (전자기 간섭/하드웨어 오류):"
dmesg | grep -i "network\|ethernet" | tail -2

echo "F_bio (적응형 트래픽/QoS):"
tc -s qdisc show | grep -E "qdisc|rate" | head -2
echo

# ==========================================
# 통합 보안 모니터링
# ==========================================
echo "10. 통합 보안 상황 문장 생성"
echo "=============================="

# 현재 네트워크 상황을 완전한 문장으로 구성
CURRENT_IP=$(hostname -I | awk '{print $1}')
GATEWAY=$(ip route | grep default | awk '{print $3}')
CONNECTIONS=$(ss -t | grep ESTAB | wc -l)
DNS_SERVER=$(systemd-resolve --status | grep "DNS Servers:" | awk '{print $3}' | head -1)

echo "현재 상황 문장:"
echo "호스트(명사) $CURRENT_IP 가 라우터(전치사) $GATEWAY 를통해(전치사) 인터넷에 연결되어(접속사) TCP로(동사) $CONNECTIONS 개의(형용사) 세션을 유지하면서(부사) DNS서버(관사) $DNS_SERVER 를 사용하되(조사) 방화벽정책이(조사) 모든 트래픽을 제어한다."

echo
echo "=== 네트워크 보안 명령어 실행 완료 ==="
echo "각 품사별 명령어가 겹치지 않게 구성되었습니다."
echo "TEE(조사)와 OS(8품사) 영역이 명확히 분리되어 실행됩니다."