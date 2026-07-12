 [배경] 
삼성전자는 2015년부터 대학생을 대상으로 프로그래밍 챌린지 SCPC(Samsung Collegiate Programming Challenge)를 개최하며, 알고리즘 기반 문제 해결 능력을 갖춘 우수 인재를 지속적으로 발굴해왔습니다.

최근 생성형 AI를 비롯한 인공지능 기술의 발전과 함께 산업 현장에서 실무형 AI 역량에 대한 요구가 강화되면서, 삼성전자는 AI 기반 실전 문제 해결 역량을 갖춘 인재를 발굴하기 위해 '2025 SCPC : AI 챌린지'를 신설했습니다.

2026년 AI 챌린지 참가자는 사용자의 요청을 이해하고 정보를 기획, 검색, 검증해 최종 의사결정을 수행하는 AI Agent 실행 체계(Harness)를 설계 및 구현하게 되며 우수 참가자에게는 상금과 함께 삼성전자 채용 시 우대 혜택을 제공합니다.





[대회 방식]
본 챌린지는 개인전(1인)이며 1차, 2차 예선과 본선 순으로 진행됩니다.

🔹1차 예선 : 2차 예선 진출자를 선발하기 위한 과정이며 Private 리더보드 상위 40명이 2차 예선에 진출하게 됩니다.

🔹2차 예선 : 본선 진출자를 선발하기 위한 과정이며 Private 리더보드 상위 20명이 본선에 진출하게 됩니다.

🔹본선 : 2차 예선 산출물에 대한 솔루션 PT 및 예선 문제 해결 과정에 대한 질의응답을 진행합니다.

※ 본선은 오프라인으로 진행됩니다.



 [예선 주제] 
AI Agent Harness 설계를 통한 창의적 문제 해결



 [문제 설명]
참가자는 주어진 과제 데이터를 바탕으로 개인 기기 agent가 마주치는 요청 맥락을 이해하고, 필요한 판단과 실행 계획을 수립하여 정해진 형식의 답안을 생성하는 AI Agent Harness를 설계·구현합니다.

본 과제의 핵심은 단순히 하나의 답을 맞히는 것이 아니라, 동일한 공개 데이터와 동일한 제출 형식 안에서 task state, visible history, session memory, 정책·안전 신호를 얼마나 일관되게 해석하는지에 있습니다. 따라서 평가는 모델 크기나 외부 API 성능 차이가 아니라, 참가자가 설계한 agent logic이 주어진 맥락을 얼마나 정확하고 신뢰성 있게 처리하는지를 중심으로 이루어집니다.

좋은 GPU나 외부 대형 모델을 많이 쓰는 것이 본질인 대회가 아닙니다. 좋은 성능은 주어진 fixed SLM interface와 task JSON을 바탕으로 parser, session memory, safety/control logic, content scope, plan construction을 얼마나 잘 설계했는지에서 나와야 합니다.

참가자는 단순한 키워드 분류기나 특정 공개 예시에만 맞춘 lookup table이 아니라, 새로운 task stream에도 적용 가능한 작은 agent harness를 만드는 것을 목표로 해야 합니다. 좋은 harness는 보통 다음 질문을 순서대로 처리합니다.

현재 요청에서 중심이 되는 object는 무엇인가?
최종 동작의 대상 또는 수신처는 어디인가?
그대로 진행할지, 범위를 줄여 진행할지, 보류할지, 사용자 확인이 필요한지 어떻게 판단할 것인가?
민감 정보, 동의,
판단 결과를 구조화된 content_scope, policy, plan_events로 어떻게 표현할 것인가?
 보안 알림, 세션 이력, 사용자 메모리를 어떻게 함께 반영할 것인가?
예선 단계에서 참가자는 제공된 Public Screening 과제에 대해 로컬 환경에서 답안을 생성하고, 생성된 answer JSON을 DACON 제출 형식에 맞춰 submission.csv의 단일 셀에 담아 제출합니다. 제출된 CSV는 서버에서 JSON으로 복원된 뒤, 서버 보관 정답과 비교되어 점수가 산출됩니다.
입출력 형식, 데이터 구성, 제출 예시 및 용어집은 대회 데이터 페이지를 통해 제공됩니다.



[참가자격]
대학(원) 재학 또는 휴학생

※ 전공 및 학년 제한 없음

※ 졸업유예생 참여불가

※ SCPC Algorithm 챌린지와 중복 참여 불가



[주최 / 운영]
주최: 삼성전자

주관: 삼성리서치

운영: 데이콘

---
1. 리더 보드 (예선)
평가 산식은 참가자가 제출한 700개 Public Screening 답안을 서버 보관 정답과 비교하여 산출한 overall 점수입니다. 
점수가 높을수록 우수합니다.
Public leaderboard 점수는 공개 screening task에 대한 점수입니다. 
공개 screening은 제출 형식, 기본 동작, 문제 이해도를 확인하기 위한 리더보드로 활용됩니다. 
예선 종료 후 상위권 참가자는 주최측 안내에 따라 동일한 아이디어를 구현한 harness.py 실행 가능본을 추가 제출해야 할 수 있으며, 이 코드는 주최측 검증 환경에서 별도 비공개 task stream으로 재현성 및 일반화 성능을 확인할 수 있습니다.
예선 제출 방식은 다음과 같습니다.
				1. 참가자는 제공된 screening_tasks.jsonl의 700개 과제에 대해 답안을 생성합니다.

				2. 생성된 answer JSON 전체를 submission.csv의 submission 컬럼 단일 셀에 저장합니다.

				3. DACON 서버는 submission.csv를 업로드받아 JSON을 복원합니다.

				4. 서버 보관 정답과 비교하여 700개 Public Screening 기준 overall 점수를 산출합니다.

				5. 산출된 overall 점수가 public leaderboard에 반영됩니다.

제출 파일 구조는 다음과 같습니다.
파일명: submission.csv
컬럼명: submission
데이터 행: 1행
인코딩: UTF-8
내용: submission 컬럼의 단일 셀에 answer JSON 전체 저장
JSON 파일 직접 제출은 지원하지 않습니다.
예시:
submission
"{""schema"":""scpc.final.answer.v1"",""meta"":{...},""answers"":{...}}"
세부 제출 형식은 데이터 페이지의 sample_submission.csv, submission_schema.json, TERMS_GUIDE.md를 참고해 주시기 바랍니다.



2. 평가 방식
참가자가 submission.csv를 업로드하면 서버는 CSV의 submission 셀에 포함된 JSON을 복원하여 채점합니다. 
채점은 Public Screening 평가 대상 과제 700개를 기준으로 진행되며, 서버에 보관된 비공개 정답과 비교하여 overall 점수 하나를 산출합니다.
1차 예선 리더보드에는 제출물의 overall 점수가 반영됩니다.
예선 종료 후 상위권 참가자는 주최측 안내에 따라 harness.py 코드 실행 가능본과 README를 제출해야 할 수 있습니다. 
제출된 코드는 재현성 검증 및 내부 검증 절차를 거치며, 검증 결과에 따라 다음 단계 진출 여부가 확정됩니다.
상위권 검증 단계에서는 예선 제출물의 재현 가능성, 제공된 fixed SLM interface 사용 여부, 외부 모델/API 사용 여부, 하드코딩 여부 등을 확인할 수 있습니다. 
필요 시 주최측 내부 평가 환경에서 별도의 비공개 과제를 활용한 추가 검증이 진행될 수 있습니다.
따라서 공개 screening 점수를 높이는 것만이 아니라, 같은 harness가 새로운 task에서도 작동하도록 설계하는 것이 중요합니다. 
공개 dev 정답을 보고 제출 형식과 필드 의미를 학습하는 것은 허용되지만, 특정 task id, 공개 예시 문장, 공개 screening 항목에만 맞춘 답안표를 만드는 방식은 상위권 검증에서 재현성 또는 일반화 문제가 될 수 있습니다.

---
1. 참여 규칙
개인(1인)으로만 참여할 수 있습니다.
개인 참가 방법 : 팀 신청 없이, 자유롭게 제출탭에서 제출 가능합니다.
동일인의 다계정 참가 등록은 금지되며, 적발 시 부정행위로 처리됩니다.
SCPC 알고리즘 챌린지와 중복하여 참가 불가합니다.
  

2. API, 외부 데이터 및 사전 학습 모델 관련 규칙
2-1. 추론 모델 고정
본 과제에서는 참가자에게 별도 대형 모델 설치를 필수로 요구하지 않습니다. 
공개 안내의 fixed SLM interface는 task evidence, risk, redaction, confirmation 관련 신호를 보조적으로 활용하기 위한 고정 interface입니다.
FixedSLMClient는 정답을 직접 반환하는 모델이 아닙니다. 최종 답안의 focal_id, target, control, content_scope, policy, plan_events는 참가자의 Harness logic이 직접 구조화해야 합니다.
summarize_task()는 task를 읽고 위험 신호, 삭제 또는 축약 필요성, 사용자 확인 필요성 같은 evidence를 일정한 형식으로 요약해 주는 보조 함수로 이해하면 됩니다. 
이 출력은 정답표가 아니며, 참가자는 해당 evidence를 자신의 rule, parser, session memory, plan builder와 결합해 최종 answer를 만들어야 합니다. 
즉, fixed SLM은 "답을 대신 써 주는 장치"가 아니라 "agent가 task를 더 안정적으로 읽도록 돕는 같은 조건의 보조 입력"입니다.
공식 실행 조건은 다음과 같습니다.
제공 interface: FixedSLMClient
제출 JSON의 meta.fixed_slm_policy: local_fixed_slm_only
제출 JSON의 meta.model_id: scpc-final-fixed-slm-local-facade
외부 유료 LLM API, 네트워크 호출, 임의 외부 모델 사용 금지
권장 기본값: temperature=0.0, seed=42
상위권 코드 검증에서 사용하는 Harness는 FinalHarness.answer_task(task, session) 형태입니다. 
참가자는 task의 prompt, device_state, records, visible_history, 이전 session 상태 등을 바탕으로 판단하고, 필요한 경우 FixedSLMClient facade가 제공하는 evidence 신호를 보조적으로 활용해 다음 형식의 답안을 반환해야 합니다.
class FinalHarness:
    def __init__(self):
        self.slm = FixedSLMClient()
        self.user_memory = {}

    def answer_task(self, task, session):
        evidence = self.slm.summarize_task(task)
        answer = build_structured_answer(task, session, evidence)
        return answer
답안 형식:
{
    "focal_id": "...",
    "target": "...",
    "control": "proceed|amend|hold|ask",
    "content_scope": {...},
    "policy": {...},
    "plan_events": [...]
}
권장 구현 구조는 다음과 같습니다.
choose_focal: 현재 요청에서 중심 object를 고릅니다.
infer_target: 최종 수신처, 앱, 채널, 장치, 메모리 저장소 등을 정합니다.
decide_control: proceed, amend, hold, ask 중 하나를 결정합니다.
build_content_scope: 어떤 정보는 쓰고 어떤 정보는 제외할지 정합니다.
build_policy: 위험 신호, 위반 가능성, 확인 필요 여부를 정리합니다.
build_plan_events: 읽기, 확인, 요약, 삭제, 전송, 보류, 업데이트 같은 계획 단계를 만듭니다.
update_session_memory: 같은 실행 흐름에서 이후 task가 참고할 정보를 저장합니다.
제공 baseline은 위 구조를 한 파일 안에서 실행해 submission.csv를 만드는 예시입니다. 
baseline은 제출 형식과 구현 흐름을 보여주는 약한 출발점이며 높은 점수를 위해서는 참가자가 각 모듈의 판단 로직을 직접 개선해야 합니다.
상위권 참가자는 제출 종료 후 재현성 검증을 위해 사용한 Harness 코드, 실행 방법, 주요 규칙·프롬프트·파싱 로직, fixed SLM facade 활용 방식을 제출해야 할 수 있습니다. 
제출 CSV의 metadata가 공식 interface 사용을 주장하더라도, 실제 생성 과정이 외부 모델/API/수동 라벨링에 의존한 경우 평가 기준 위반으로 처리될 수 있습니다.

2-2. 개발 도구 사용
Harness 코드 작성, 디버깅, 개선 등 개발 단계에서는 AI 코딩 도구를 사용할 수 있습니다. 
단, 최종 제출 답안을 생성하는 과정에서는 주최측이 제공한 데이터와 허용된 fixed SLM interface 사용 기준을 따라야 하며, 참가자 간 성능 차이는 Harness 설계와 구현 방식에서 발생해야 합니다.

2-3. 데이터 활용 기준
대회에서 제공하는 데이터와 공개된 연습용 dev 데이터는 문제 구조 이해, 제출 형식 확인, 로컬 테스트에 활용할 수 있습니다.
다만 평가 대상 과제의 정답이나 특정 패턴을 직접 추정하거나, 특정 평가 과제에만 맞춘 방식으로 답안을 구성하는 행위는 허용되지 않습니다.
권장되는 활용 방식은 dev task와 dev answer를 통해 "답안 JSON이 어떤 구조를 가져야 하는지", "각 필드가 어떤 역할을 하는지", "내 harness가 스키마를 만족하는지"를 점검하는 것입니다. 
공개 dev 예시에서 보이는 특정 문장, record 값, object 순서를 그대로 외워 screening 또는 검증 task에 적용하는 방식은 권장하지 않습니다.

2-4. 하드코딩 금지
특정 task_id, session_id 또는 평가 항목에 대한 정답을 코드에 직접 입력하거나, 일반화되지 않은 방식으로 평가 과제를 푸는 행위는 무효 처리될 수 있습니다.
참가자는 다양한 과제에 적용 가능한 일반화된 Agent Harness를 설계·구현해야 합니다.


3. 코드 및 PPT 제출 규칙
3-1. 예선 제출물
예선 단계에서는 지정된 형식의 submission.csv 파일을 제출해야 합니다.
submission.csv는 다음 조건을 만족해야 합니다.
파일명: submission.csv
UTF-8 인코딩
컬럼명: submission
데이터 행: 1행
submission 셀 안에 answer JSON 전체 포함
JSON 최상위 구조는 submission_schema.json을 따름
JSON 파일 직접 제출은 지원하지 않습니다. 참가자는 최종 제출 파일로 submission.csv만 업로드해야 합니다.
상위권 참가자는 주최측 안내에 따라 harness.py 코드 실행 가능본과 README를 제출해야 할 수 있습니다.
제출된 코드는 재현성 검증을 거치며, 검증 결과에 따라 최종 진출 여부가 확정됩니다. 
이 단계에서 주최측 내부 평가 환경의 별도 비공개 과제를 활용한 추가 검증이 진행될 수 있습니다.
상위팀 코드 제출의 세부 형식과 제출 방법은 추후 주최측 안내에 따릅니다.

3-2. 본선 제출물
본선 단계에서는 예선에서 사용한 Harness 설계 및 구현 내용을 바탕으로 솔루션 발표자료(PPT)와 코드를 제출해야 합니다.
발표자료에는 Harness의 설계 의도, 전체 아키텍처, 문제 해결 전략, fixed SLM evidence 활용 방식, 세션 메모리 관리 방식, 한계점 및 개선 방향 등을 포함하는 것을 권장합니다. 
발표자료 분량은 15페이지 이내로 작성하며, 본선에서는 발표와 질의응답이 함께 진행됩니다.


4. 유의 사항
1일 최대 제출 횟수: 3회(7/8 00:00 적용) (단, 대회 운영 상황 및 점수 분포에 따라 제출 횟수는 조정될 수 있습니다)
제출 파일은 반드시 submission.csv 형식이어야 하며, JSON 파일을 직접 제출하는 방식은 지원하지 않습니다.
CSV 내부의 JSON이 파싱되지 않거나, 제출 스키마를 만족하지 않거나, 필수 과제 ID가 누락된 경우 제출 오류 또는 낮은 점수로 처리될 수 있습니다.
dev_answers.json은 일부 dev_tasks.jsonl 문제에 대한 참조 답안 예시입니다. 전체 dev 문제의 상세 해설이나 screening 정답이 아니며, 제출 구조와 로컬 동작을 점검하기 위한 용도입니다.
dev_answers.json은 screening 답안 생성이나 최종 제출 파일에 포함해서는 안 됩니다. 
실제 리더보드 평가는 screening_tasks.jsonl의 700개 과제에 대한 submission.csv 제출 결과로 산출됩니다.
사용 가능 언어: Python
대회 기간과 참가자들의 점수 분포 등을 고려하여, 주최측의 요청에 따라 일정 기간 동안 '코드 공유' 탭이 일시적으로 비활성화될 수 있습니다.
모든 csv 형식의 데이터와 제출 파일은 UTF-8 인코딩을 적용합니다.
모델 학습과 추론에서 평가 데이터셋 정보 활용(Data Leakage)시 실격 또는 본선 진출이 불가능합니다.
평가용 이미지 또는 지문을 수작업으로 라벨링하거나, 이를 기반으로 정답을 직접 추정하여 학습 데이터처럼 사용하는 행위
평가 데이터셋에서 특정 패턴이나 정답 분포를 분석해 모델 구조, 전처리 방식, 정답 후보 설정 등에 반영하는 행위 등
모든 학습, 추론의 과정 그리고 추론의 결과물들은 정상적인 코드를 바탕으로 이루어져야하며, 비정상적인 방법으로 얻은 제출물들은 적발 시 규칙 위반에 해당됩니다.
대회 직후 공개되는 Private 랭킹은 최종 순위가 아니며 본선 진행 후, 최종 수상자가 결정됨
데이콘은 부정 제출 행위를 금지하고 있으며 데이콘 대회 부정 제출 이력이 있는 경우 평가가 제한됩니다. 자세한 사항은 아래의 링크를 참고해 주시기 바랍니다.
https://dacon.io/notice/notice/13

 

5. 토론(질문)
대회 운영 및 데이터 이상에 관련된 질문 외에는 답변을 드리지 않고 있습니다. 기타 질문은 토론 페이지를 통해 자유롭게 토론해주시기 바랍니다.
데이콘 답변을 희망하는 경우 토크 게시글 댓글로 질문을 올려 주시기 바랍니다.
예) [DACON 답변 요청] 시상식은 언제 열리나요?

---
[배포용 데이터 구조]

대회 데이터 페이지에는 다음 파일이 제공됩니다.

screening_tasks.jsonl: 예선 평가 대상 700개 문제
dev_tasks.jsonl: 연습용 문제
dev_answers.json: 일부 연습용 dev 문제에 대한 참조 답안 예시
sample_submission.csv: 제출 CSV 예시
submission_schema.json: 제출 JSON 구조
TERMS_GUIDE.md: task와 제출 JSON의 주요 용어 설명
SCPC2026_Final_baseline.ipynb: 참가자용 Python 통합 실행 예시
처음 시작하는 참가자는 TERMS_GUIDE.md로 task와 answer JSON의 필드를 확인한 뒤, SCPC2026_Final_baseline.ipynb를 실행해 전체 흐름을 보는 것을 권장합니다. 이후 baseline의 FinalHarness 내부 함수들을 하나씩 개선하면 됩니다.