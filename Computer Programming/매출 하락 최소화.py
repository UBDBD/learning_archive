# 2021 KAKAO BLIND RECRUITMENT > 매출 하락 최소화

def solution(sales, links):
    n = len(sales)

    # 각 직원의 직속 팀원 저장 (0-based)
    children = [[] for _ in range(n)]
    for parent, child in links:
        children[parent - 1].append(child - 1)

    # dp[x][0]: x 불참 시 최소 비용
    # dp[x][1]: x 참석 시 최소 비용
    dp = [[0, 0] for _ in range(n)]

    # DFS로 탐색 순서 저장
    order = []
    stack = [0]

    while stack:
        x = stack.pop()
        order.append(x)

        for child in children[x]:
            stack.append(child)

    # 자식부터 부모 순서로 DP 계산
    for x in reversed(order):

        # x 참석: 자식은 참석/불참 중 최소 비용 선택
        dp[x][1] = sales[x]

        for child in children[x]:
            dp[x][1] += min(dp[child][0], dp[child][1])

        # 자식이 없는 경우 x 불참 비용은 0
        if not children[x]:
            dp[x][0] = 0
            continue

        base = 0
        min_extra = float('inf')

        for child in children[x]:
            # 자식별 최소 비용 선택
            base += min(dp[child][0], dp[child][1])

            # x 불참 시 자식 한 명을 참석시키기 위한 추가 비용
            extra = max(0, dp[child][1] - dp[child][0])
            min_extra = min(min_extra, extra)

        # x 불참: 자식 중 최소 한 명은 반드시 참석
        dp[x][0] = base + min_extra

    # CEO 참석/불참 중 최소 비용
    return min(dp[0][0], dp[0][1])
