# 2022 KAKAO BLIND RECRUITMENT > 양과 늑대

def solution(info, edges):    
    answer = 1
    
    children = [[] for _ in range(len(info))]
    
    for parent, child in edges:
        children[parent].append(child)
    
    stack = [(1, 0, children[0])]
    
    while stack:
        sheep, wolf, candidates = stack.pop()
        answer = max(answer, sheep)
        
        for node in candidates:
            next_sheep = sheep + (info[node]==0)
            next_wolf = wolf + (info[node]==1)
            
            if next_wolf >= next_sheep:
                continue
                
            next_candidates = candidates.copy()
            next_candidates.remove(node)
            next_candidates.extend(children[node])
            
            stack.append((next_sheep, next_wolf, next_candidates))
        
    return answer