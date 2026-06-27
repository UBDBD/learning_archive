# 2022 KAKAO BLIND RECRUITMENT > 양과 늑대

def solution(info, edges):    
    answer = 0
    
    child = [[] for _ in range(len(info))]
    
    for p, c in edges:
        child[p].append(c)
        
    s = [(1, 0, child[0])]
    
    while s:
        sheep, wolf, candidates = s.pop()
        answer = max(answer, sheep)
        
        for i in candidates:
            nxt_sheep = sheep + (info[i]==0)
            nxt_wolf = wolf + (info[i]==1)
            
            if nxt_wolf >= nxt_sheep:
                continue
            
            nxt_candidates = candidates.copy()
            nxt_candidates.remove(i)
            nxt_candidates.extend(child[i])
            
            s.append((nxt_sheep, nxt_wolf, nxt_candidates))
                
    return answer
