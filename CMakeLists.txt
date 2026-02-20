cmake_minimum_required(VERSION 3.16)
project(microgpt LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release 빌드를 기본으로 설정
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 컴파일러별 최적화 플래그
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")

# 실행 파일 생성
add_executable(microgpt microgpt.cpp)

# 경고 플래그 (교육용 코드이므로 엄격하게)
target_compile_options(microgpt PRIVATE
    -Wall -Wextra -Wpedantic
)

# math 라이브러리 링크 (Linux에서 필요할 수 있음)
target_link_libraries(microgpt PRIVATE m)
