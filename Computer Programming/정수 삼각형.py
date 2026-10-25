# 동적계획법(Dynamic Programming) > 정수 삼각형

def solution(triangle):
    # 아래에서 두 번째 줄부터 위로 이동
    for i in range(len(triangle) - 2, -1, -1):
        for j in range(len(triangle[i])):
            # 왼쪽 아래, 오른쪽 아래 중 큰 값을 현재 값에 누적
            triangle[i][j] += max(triangle[i+1][j], triangle[i+1][j+1])

    # 꼭대기에 누적된 최대 경로 합 반환
    return triangle[0][0]
