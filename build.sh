#!/bin/bash

# 사용방법
#./build.sh or ./build.sh release => Release 모드로 빌드
#./build debug => debug 모드로 빌드 
#./build clean => 기존 빌드 파일 모두 삭제하기

# 스크립트 실행 중 오류가 발생하면 즉시 중단
set -e

# 스크립트의 첫 번째 인자를 확인
ARG1=${1:-release} # 인자가 없으면 기본값으로 'release' 사용

# 'clean' 인자가 주어지면 build 디렉토리를 삭제하고 종료
if [ "$ARG1" == "clean" ]; then
    echo ">> Cleaning up the build directory..."
    rm -rf build
    echo ">> Done."
    exit 0
fi

# 빌드 타입 설정
BUILD_TYPE="Release"
if [ "$ARG1" == "debug" ]; then
    BUILD_TYPE="Debug"
fi

# 빌드 디렉토리 설정
BUILD_DIR="build"

echo ">> Build Type: $BUILD_TYPE"

# 1. 빌드 디렉토리 생성
echo ">> Creating build directory..."
mkdir -p $BUILD_DIR

# 2. 빌드 디렉토리로 이동
cd $BUILD_DIR

# 3. CMake 실행
echo ">> Configuring project with CMake..."
cmake -G Ninja -DCMAKE_BUILD_TYPE=$BUILD_TYPE ..

echo ""
echo ">> To build the project, run 'ninja' inside the '$BUILD_DIR' directory."
