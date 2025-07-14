import os
import hashlib
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes

# 디렉터리 설정
TARGET_DIR = "C:/temp/"
BLOCK_SIZE = 16

# 결정론적 계산 (P: 암호화) ↔ 고전 계산
def pad(data):
    padding = BLOCK_SIZE - len(data) % BLOCK_SIZE
    return data + bytes([padding]) * padding

# NP 검증과정 (맥스웰 ↔ 비결정 검증)
def unpad(data):
    padding = data[-1]
    return data[:-padding]

def get_key(password):
    return hashlib.sha256(password.encode()).digest()

# 상호보완적 계산 시스템 (P 방식)
def encrypt_file(file_path, key):
    with open(file_path, "rb") as f:
        data = f.read()

    # NP ↔ P 교차 연산: IV는 랜덤 (비결정적 요소)
    cipher = AES.new(key, AES.MODE_CBC)
    iv = cipher.iv
    encrypted = cipher.encrypt(pad(data))

    encrypted_path = file_path + ".encrypted"
    with open(encrypted_path, "wb") as f:
        f.write(iv + encrypted)

    print(f"[P] 🔐 Encrypted: {file_path} → {encrypted_path}")
    os.remove(file_path)

# NP ↔ 검증 및 역변환 (맥스웰 검증 개념)
def decrypt_file(file_path, key):
    with open(file_path, "rb") as f:
        iv = f.read(16)
        ct = f.read()

    cipher = AES.new(key, AES.MODE_CBC, iv)
    try:
        decrypted = unpad(cipher.decrypt(ct))
    except ValueError:
        print(f"[NP] ❌ 복호화 실패: {file_path}")
        return

    decrypted_path = file_path.replace(".encrypted", "")
    with open(decrypted_path, "wb") as f:
        f.write(decrypted)

    print(f"[NP] 🔓 Decrypted: {file_path} → {decrypted_path}")
    os.remove(file_path)

def process_all_files(directory, key, mode="encrypt"):
    for file in os.listdir(directory):
        file_path = os.path.join(directory, file)
        if os.path.isfile(file_path):
            if mode == "encrypt" and not file.endswith(".encrypted"):
                encrypt_file(file_path, key)
            elif mode == "decrypt" and file.endswith(".encrypted"):
                decrypt_file(file_path, key)

if __name__ == "__main__":
    print("🧠 Maxwell-Navier Complementary Cipher System (P ↔ NP)")
    mode = input("모드 선택 (encrypt / decrypt): ").strip().lower()
    password = input("암호 키 입력: ").strip()

    key = get_key(password)

    if mode in ("encrypt", "decrypt"):
        process_all_files(TARGET_DIR, key, mode)
    else:
        print("❌ 잘못된 모드입니다. encrypt 또는 decrypt 입력하세요.")
