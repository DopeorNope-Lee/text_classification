"""
환경 설정 파일 (setup_environment.py)
====================================

이 파일은 Windows 11 환경에서 텍스트 분류 프로젝트를 위한 환경을 설정합니다.
학생들이 처음 프로젝트를 시작할 때 이 파일을 실행하여 필요한 환경을 구성할 수 있습니다.

사용법:
    python setup_environment.py
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Python 버전을 확인합니다."""
    print("Python 버전 확인 중...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Python 3.8 이상이 필요합니다.")
        print(f"현재 버전: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"Python 버전: {version.major}.{version.minor}.{version.micro}")
    return True

def install_requirements():
    """필요한 패키지들을 설치합니다."""
    print("필요한 패키지 설치 중...")
    
    # 기본 필수 패키지들
    packages = [
        "torch",
        "transformers",
        "datasets",
        "peft",
        "accelerate"
    ]
    
    for package in packages:
        try:
            print(f"설치 중: {package}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"{package} 설치 완료")
        except subprocess.CalledProcessError:
            print(f"{package} 설치 실패")
            return False
    
    return True

def create_sample_data():
    """샘플 데이터 파일을 생성합니다."""
    print("샘플 데이터 생성 중...")
    
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 샘플 데이터셋 생성
    sample_data = """text,label
안녕하세요 고객센터입니다,neutral
이 제품 정말 최고예요!,positive
서비스가 너무 나빠요,negative
배송이 빨라서 좋았어요,positive
품질이 기대에 못 미쳐요,negative
친절한 응대 감사합니다,positive
환불하고 싶어요,negative
추천하고 싶은 제품입니다,positive
불만이 있어요,negative
만족스러운 구매였습니다,positive"""
    
    with open(data_dir / "sample_dataset.csv", "w", encoding="utf-8") as f:
        f.write(sample_data)
    
    # 테스트용 텍스트 파일 생성
    test_texts = """안녕하세요
이 제품은 정말 좋아요
서비스가 별로예요
배송이 빨라요
품질이 나빠요"""
    
    with open(data_dir / "test_texts.txt", "w", encoding="utf-8") as f:
        f.write(test_texts)
    
    print("샘플 데이터 생성 완료")

def create_directories():
    """필요한 디렉토리들을 생성합니다."""
    print("디렉토리 생성 중...")
    
    directories = [
        "models",
        "checkpoints",
        "logs"
    ]
    
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"{dir_name} 디렉토리 생성 완료")

def main():
    """메인 설정 함수"""
    print("텍스트 분류 프로젝트 환경 설정을 시작합니다...")
    print("=" * 50)
    
    # 1. Python 버전 확인
    if not check_python_version():
        return
    
    # 2. 패키지 설치
    if not install_requirements():
        print("패키지 설치에 실패했습니다.")
        return
    
    # 3. 디렉토리 생성
    create_directories()
    
    # 4. 샘플 데이터 생성
    create_sample_data()
    
    print("=" * 50)
    print("환경 설정이 완료되었습니다!")
    print("다음 단계:")
    print("1. train.py를 실행하여 모델을 학습하세요")
    print("2. inference.py를 실행하여 예측을 해보세요")
    print("3. README.md를 읽어서 프로젝트 구조를 이해하세요")

if __name__ == "__main__":
    main() 