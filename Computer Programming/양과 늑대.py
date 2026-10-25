# 2022 KAKAO BLIND RECRUITMENT > 양과 늑대

def solution(info, edges):    
    answer = 0
    
    # 각 노드의 자식 노드 저장
    child = [[] for _ in range(len(info))]
    
    for p, c in edges:
        child[p].append(c)
    
    # (현재 양 수, 현재 늑대 수, 다음에 방문 가능한 노드)
    s = [(1, 0, child[0])]
    
    while s:
        sheep, wolf, candidates = s.pop()
        
        # 지금까지 모은 최대 양의 수 갱신
        answer = max(answer, sheep)
        
        # 방문 가능한 노드를 하나씩 선택
        for i in candidates:
            nxt_sheep = sheep + (info[i] == 0)
            nxt_wolf = wolf + (info[i] == 1)
            
            # 늑대가 양보다 많거나 같으면 해당 경로 중단
            if nxt_wolf >= nxt_sheep:
                continue
            
            # 방문한 노드는 제거하고, 해당 노드의 자식은 후보에 추가
            nxt_candidates = candidates.copy()
            nxt_candidates.remove(i)
            nxt_candidates.extend(child[i])
            
            # 다음 상태 탐색
            s.append((nxt_sheep, nxt_wolf, nxt_candidates))
                
    return answer
