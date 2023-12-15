# GPT-1-_-2020049398-_-HanChaeeun
# 📑 GPT-1 논문 리뷰 및 다음 단어 예측하기 모델 테스트 보고서

## ✔ 들어가기 전

- 이 보고서는 GPT-1 논문 리뷰와 다음 단어를 예측하는 모델을 테스트하는 과정을 포함한 딥러닝 초보자를 위한 문서입니다.
- 이 보고서에서 다루는 테스트 과정은 Hugging Face Transformers 라이브러리에 GPT-1이 구현되어 있지 않은 이유로, transformers 라이브러리에 구현되어 있는 GPT-2 모델의 기능을 테스트하는 것으로 진행됩니다.
- Google Colab에서 실행되도록 설계되었으며 Hugging Face Transformers 라이브러리를 설치해야 합니다.

## 1. 서론

GPT-1(Generative Pre-trained Transformer)은 OpenAI에서 개발한 언어 생성 모델로, Transformer 아키텍처를 사용하여 대규모 텍스트 데이터에서 사전 훈련된 후, 다양한 자연어 처리 작업에 적용할 수 있도록 설계되었습니다. 이 논문 리뷰에서는 GPT-1의 주요 아이디어와 기술적 특징, 그리고 모델이 가져온 혁신적인 점에 대해 살펴보겠습니다.

## 2. 주요 아이디어

GPT-1의 주요 아이디어는 대규모 데이터셋에서 사전 훈련된 언어 모델을 사용하여 다양한 자연어 처리 작업을 해결하는 것입니다. 이를 위해 Transformer 아키텍처를 활용하며, 주어진 문맥에서 다음 단어를 예측하는 언어 모델을 학습합니다. 이렇게 학습된 모델은 downstream 작업에 fine-tuning되어 활용됩니다.

## 3. 기술적 특징

### 3.1 Transformer 아키텍처 활용

GPT-1은 Attention 기반의 Transformer 아키텍처를 사용합니다. 이를 통해 문장 내 단어 간의 관계를 효과적으로 모델링하고, 병렬 처리를 통해 효율적인 학습이 가능해졌습니다.

### 3.2 사전 훈련 및 Fine-tuning

GPT-1은 먼저 대규모 텍스트 데이터에서 사전 훈련된 후, 각각의 downstream 작업에 대해 fine-tuning됩니다. 이를 통해 모델은 다양한 작업에서 뛰어난 성능을 발휘할 수 있게 되었습니다.

### 3.3 Autoregressive 언어 모델

모델은 문맥 내 이전 단어들을 사용하여 다음 단어를 예측하는 autoregressive한 특성을 가지고 있습니다. 이는 훈련 중에 모델이 문맥을 이해하고 문장의 일관성을 유지하도록 도와줍니다.

## 4. 혁신적인 점

### 4.1 전이 학습의 활용

GPT-1은 전이 학습을 통해 다양한 자연어 처리 작업에 적용할 수 있는 범용성을 갖추고 있습니다. 이는 대량의 데이터에서 학습된 모델이 다른 작업에서도 좋은 성능을 보이게 함으로써 자연어 처리 분야에서의 선행 학습의 효과를 입증했습니다.

### 4.2 큰 모델의 활용

GPT-1은 모델의 크기를 확장함으로써 성능 향상을 이루어냈습니다. 모델 크기의 중요성을 강조하며, 더 큰 모델이 더 나은 성능을 보일 수 있음을 보여주었습니다.

## 5. 결론

GPT-1은 Transformer 아키텍처와 전이 학습의 아이디어를 통해 언어 모델링 분야에서 큰 발전을 이루어냈습니다. 사전 훈련과 fine-tuning을 통한 학습 방법, autoregressive한 특성, 큰 모델의 활용 등은 자연어 처리 분야에서의 기술적 혁신을 나타냅니다. 그러나 긴 문장에서의 일관성 유지와 특정 도메인에서의 성능 향상에 대한 한계가 있습니다.


# 💻 실행 결과 리포트

## 1. 라이브러리 설치 확인:

- **transformers 라이브러리 확인:**
  - 이미 설치되어 있습니다.
  - 설치된 버전: 4.35.2

## 2. 모델 및 토크나이저 로드:

- **GPT-2 모델 및 토크나이저 로드:**
  ```python
  from transformers import GPT2LMHeadModel, GPT2Tokenizer

  model_name = "gpt2"
  tokenizer = GPT2Tokenizer.from_pretrained(model_name)
  model = GPT2LMHeadModel.from_pretrained(model_name)
  
## 3. 🐱 입력값 및 모델 실행

- **입력값 및 모델 실행 :**
  ```python
  input_text = "I love cats because"
  
  input_ids = tokenizer.encode(input_text, return_tensors="pt") 
  output = model.generate(input_ids, max_length=50, num_return_sequences=1, no_repeat_ngram_size=2) 
  generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

## 4. 생성된 텍스트 확인
  
I love cats because I've been around them for a long time. I love to play with them, and they are my favorite. They are so cute, and I can't wait to see what they do with their little ones.

## 5. 결과해석 및 결론

입력 문장에 대해 GPT-2 모델이 그럴듯한 응답을 생성한 것으로 보입니다. 모델이 주어진 문맥을 이해하고 일관된 텍스트를 생성하는 데 성공한 것으로 판단됩니다.
