    # KatFishNet 기반 하이브리드 접근법

자료 유형: 코드
핵심 내용: KoBigBird + Linguistic Features
사용된 방법론: 딥러닝, 머신러닝, 자연어 처리(NLP)
활용 가능성: 상 (높음)
AUC: 0.85233
담당자: 임수민
모델 완성일: 2025년 7월 2일

### 요약: 해당 모델은 어떻게 동작하는가?

> 글의 '의미'를 이해하는 딥러닝 모델(BigBird)과 글의 '스타일'을 분석하는 언어학적 특징(KatFishNet)을 결합하여, 사람과 AI의 미묘한 차이를 종합적으로 판단합니다.
> 

[https://github.com/Shinwoo-Park/katfishnet](https://github.com/Shinwoo-Park/katfishnet)

### ⚙️ 꼭 알아야 할 핵심 개념

### **1. 트랜스포머 (Transformer) 모델이란? (e.g., BigBird)**

- **"글의 맥락을 이해하는 인공지능 엔진"**
- 문장 속 단어들의 관계와 전체적인 의미를 파악하는 데 특화된 딥러닝 모델입니다. GPT, BERT 등이 모두 트랜스포머를 기반으로 합니다.
- 텍스트의 의미적, 문맥적 정보를 심층적으로 분석하여 1차적인 판단 근거를 마련합니다. 특히 **BigBird** 모델은 긴 글을 처리하는 데 최적화되어 있어, 장문에서도 일관성을 파악하는 데 유리합니다.

### 2. 언어학적 특징 공학 (Linguistic Feature Engineering)이란?

- 텍스트에서 쉼표 사용 빈도, 평균 단어 길이, 특정 품사의 사용 패턴 등 수치로 계산할 수 있는 '스타일' 정보를 추출하는 과정입니다.
- **우리 모델의 역할**: AI는 사람과 다른 독특한 글쓰기 습관(e.g., 일관된 문장 길이, 특정 쉼표 사용 패턴)을 보입니다. 우리는 **KatFishNet** 논문에서 제안된 한국어 특화 특징(쉼표, 품사 다양성 등)을 추출하여, 딥러닝 모델이 놓칠 수 있는 미세한 스타일 차이를 포착합니다.

**3. 하이브리드 모델 (Hybrid Model)이란?**

- **"최강의 엔진과 최첨단 내비게이션을 함께 쓰는 자동차"**
- 위의 두 가지 접근법, 즉 의미를 파악하는 트랜스포머(엔진)와 스타일을 분석하는 특징 공학(내비게이션)을 결합한 모델입니다.
- 두 가지 다른 관점의 정보를 함께 활용하므로, 한 가지 방법에만 의존하는 모델보다 훨씬 더 정교하고 강건한 판단을 내릴 수 있습니다.

### ✅ 우리 모델의 핵심 이점 (타 방법론 대비)

### 1. 의미와 스타일, 두 마리 토끼를 잡다

단순히 딥러닝 모델을 미세조정(Fine-tuning)하는 방식은 텍스트의 의미에만 집중하는 경향이 있습니다. 우리 모델은 **언어학적 특징**을 함께 학습하여, AI가 따라 하기 힘든 사람 고유의 글쓰기 스타일(비일관성, 다양한 문체 등)까지 포착하므로 판별 능력이 뛰어납니다.

### 2. 새로운 AI 모델에 대한 강건함 (높은 일반화 성능)

AI 탐지 모델의 가장 큰 숙제는 '학습 때 보지 못한 새로운 AI'가 쓴 글을 잘 맞추는 것입니다. KatFishNet 논문에 따르면, 언어학적 특징, 특히 **구두점(Punctuation) 기반 특징**은 특정 AI 모델에 과적합되지 않아 새로운 유형의 AI 텍스트를 탐지하는 데 매우 효과적입니다.

### 3. 한국어에 최적화된 분석

`KoNLPy`와 같은 한국어 형태소 분석기를 사용하여 **한국어의 구조적 특성(조사, 어미 변화 등)을 반영한 품사(POS) 다양성**을 측정합니다. 이는 영어 기반의 일반적인 탐지 모델이 가질 수 없는 강력한 이점입니다.

### 📊 실제 논문 기반 성능 비교

`KatFishNet` 논문에서는 자신들의 언어학적 특징 기반 모델이 다른 방법론에 비해 얼마나 우수한지 '처음 보는 AI'에 대한 탐지 성능(Out-of-Distribution)으로 증명했습니다.

**장르: 에세이 (Essay) / unseen LLM에 대한 탐지 정확도(%)**

| 탐지 방법 | Solar | Qwen2 | Llama3.1 | **평균** |
| --- | --- | --- | --- | --- |
| Fine-tuning (단순 딥러닝) | 66.77 | 66.65 | 64.37 | 65.93 |
| LLM Paraphrasing | 92.08 | 79.74 | 72.00 | 81.27 |
| **KatFishNet (Punctuation)** | **97.57** | **94.63** | **92.45** | **94.88** |

> 출처: Park, S., et al. (2025). Detecting LLM-Generated Korean Text through Linguistic Feature Analysis. arXiv:2503.00032
> 

**결론**: 표에서 보듯이, 단순 Fine-tuning에 비해 **언어학적 특징(특히 구두점)을 활용하는 것이 처음 보는 AI를 탐지하는 데 압도적으로 유리합니다.** 해당 모델은 이 아이디어를 적극적으로 채택했습니다.

### 💻 코드 핵심 로직 (PyTorch)

**HybridModel 클래스 정의 보기**

```python
class HybridModel(nn.Module):
    def __init__(self, model_name, num_numerical_features):
        super(HybridModel, self).__init__()

        # 1. 글의 '의미'를 분석할 엔진: BigBird 모델
        self.backbone = AutoModel.from_pretrained(model_name)

        # 2. 글의 '스타일'을 분석할 작은 신경망
        self.numerical_fc = nn.Sequential(
            nn.Linear(num_numerical_features, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # 3. 두 분석 결과를 합쳐 최종 판단을 내릴 분류기
        # (BigBird 출력 크기 + 스타일 분석 결과 크기 -> 최종 1개 값)
        self.classifier = nn.Linear(self.backbone.config.hidden_size + 32, 1)

    def forward(self, input_ids, attention_mask, numerical_features):
        # 1-1. BigBird 엔진으로 의미 분석
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        text_output = outputs.pooler_output

        # 2-1. 작은 신경망으로 스타일 분석
        num_output = self.numerical_fc(numerical_features)

        # 3-1. 두 결과(의미, 스타일)를 합침
        combined = torch.cat([text_output, num_output], dim=1)

        # 3-2. 최종 판별 결과 출력
        logits = self.classifier(combined)
        return logits

```
