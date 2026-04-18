# 동적계획법(Dynamic Programming) > 정수 삼각형

def solution(triangle):
    for i in range(len(triangle)-2, -1, -1):
        for j in range(len(triangle[i])):
             triangle[i][j] += max(triangle[i+1][j], triangle[i+1][j+1])
    return triangle[i][j]