"""
환경 설정 파일 (setup.py)
====================================

이 파일은 Windows 11 환경에서 텍스트 분류 프로젝트를 위한 환경을 설정합니다.
학생들이 처음 프로젝트를 시작할 때 이 파일을 실행하여 필요한 환경을 구성할 수 있습니다.

사용법:
    python setup.py
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
    
    # 2. 디렉토리 생성
    create_directories()
    
    print("=" * 50)
    print("환경 설정이 완료되었습니다!")
    print("다음 단계:")
    print("1. train.py를 실행하여 모델을 학습하세요")
    print("2. inference.py를 실행하여 예측을 해보세요")
    print("3. README.md를 읽어서 프로젝트 구조를 이해하세요")

if __name__ == "__main__":
    main() 
