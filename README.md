<div align="center">

# CNN + GRU [ELU] 수어 인식 모델

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=yellow)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15+-orange?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-3.0-purple?logo=keras&logoColor=white)](https://keras.io)
[![License](https://img.shields.io/badge/License-MIT-green?logo=github)](LICENSE)
[![Progress](https://img.shields.io/badge/Progress-30%25-yellow)](https://github.com/yourusername/cnn-gru-sign-language)

**졸업프로젝트: 시간적 연속성 기반 수어 동작 인식 시스템**  
프레임별 키포인트 시퀀스 데이터로 CNN + GRU[ELU] 모델 개발. AI Hub 데이터셋 + MediaPipe 키포인트 활용.

</div>

---

## 📋 프로젝트 소개

수어 영상 학습은 기본적으로 동작 인식(action recognition) 문제입니다. 사용자가 제공한 데이터가 프레임별 키포인트 좌표의 시간 순서(시퀀스)이므로, 단순한 2D 이미지 분류용 CNN보다는 시간적 연속성을 처리할 수 있는 모델이 핵심입니다. 핵심 아키텍처는 CNN으로 공간 특징(손 모양, 포즈 등)을 추출하고 GRU로 시간적 패턴(동작 흐름, 제스처 연속성)을 학습하며, ELU 활성화 함수로 그래디언트 소실을 완화합니다. 왜 CNN + GRU를 사용하나요? GRU의 효율성으로 LSTM보다 구조가 단순해 빠른 학습과 계산 효율이 높고 [web:203], 파라미터 수가 적어 메모리 절약이 가능하며, 3D CNN의 한계(정확도 87%지만 동적 단어 수화에 약함 [web:205])와 비시계열 모델(RandomForest 등)의 시간 관계 파악 어려움을 극복합니다. 포인트는 ELU로 그래디언트 소실을 개선해 안정적 학습을 보장하는 것입니다 [web:201]. 입력 형식은 프레임 i의 모든 키포인트 좌표 × 시퀀스 길이(예: (batch, 30, 42, 2) - 30프레임, 21개 손 키포인트 × X/Y 좌표)로, 각 프레임은 MediaPipe로 추출된 21개 손 랜드마크의 2D 좌표로 구성되며 전체 시퀀스는 시간적 흐름을 반영합니다.

---

## 🛤️ 프로젝트 타임라인

| 기간          | 단계                  | 주요 작업                                                                 |
|---------------|-----------------------|---------------------------------------------------------------------------|
| **9월~10월 초** | 주제 및 모델 탐색    | 시계열 분류기 연구 - LSTM은 장기 의존성 학습에 강점이 있지만 파라미터가 많아 계산 비용이 높고, GRU는 LSTM보다 단순하면서도 유사한 성능을 제공하며 더 빠른 학습이 가능하며, Transformer는 대규모 데이터와 병렬 처리에 우수하지만 소규모 데이터셋에서는 과적합 위험이 크고, CNN-RNN 병합은 이미지 기반 공간 특징 추출과 시퀀스 처리의 최적 조합으로 판단됩니다. |
| **10월 중순**   | 모델 결정 및 공부    | CNN+GRU[ELU] 아키텍처 설계 - 수어 영상 데이터 특성 분석으로 프레임 i의 모든 키포인트 좌표를 시퀀스 길이만큼 쌓아 시간적 연속성을 강조하며, 단순 2D CNN의 한계(시간 정보 무시)를 극복하기 위해 GRU를 도입합니다. |
| **10월 말~11월**| 데이터 준비          | AI Hub 데이터셋 다운로드 및 MediaPipe를 통한 키포인트 추출 - 데이터 증강 기법 적용(왼손잡이 데이터 수평 반전 등)과 홀드아웃 검증을 위한 데이터 분할(80% 훈련 / 10% 검증 / 10% 테스트)을 수행합니다. |
| **11월~12월**   | 모델 구현 및 학습    | CNN 레이어로 특징 추출 + GRU 레이어로 시퀀스 처리 + ELU 활성화 적용 - 크로스 엔트로피 손실 함수와 역전파 알고리즘 구현, Early Stopping 콜백을 통한 과적합 방지를 진행합니다. |
| **12월~1월**    | 평가 및 최적화       | 정확도(Accuracy), F1-Score, 혼동 행렬을 통한 성능 분석 - 화자 독립성 평가(학습하지 않은 새로운 수화자 테스트)와 손 우세성(오른손 vs 왼손) 보완을 위한 추가 실험을 합니다. |
| **1월~2월**     | 배포 및 응용         | 실시간 수어 번역 시스템 구축 - 음성 변환 및 텍스트 출력 연계 기능 개발, 사용자 인터페이스와 확장팩 구현을 목표로 합니다. |

현재 프로젝트 진행률은 약 30%로, 모델 아키텍처 설계와 초기 데이터 전처리가 완료된 상태입니다. 다음 마일스톤은 10월 말까지 프로토타입 모델 학습을 목표로 하며, 주요 위험 요인으로는 키포인트 데이터의 품질 문제와 GPU 계산 자원 확보가 있습니다. 데이터 증강과 홀드아웃 검증을 통해 안정적인 학습 기반을 마련 중입니다.

---

## 🔬 핵심 머신러닝 개념

홀드아웃 검증(Hold-out Validation)은 데이터를 세 부분으로 나누어 모델의 성능을 체계적으로 평가하는 방법으로, 80%를 훈련 세트로 모델의 가중치와 파라미터를 학습하고 10%를 검증 세트로 학습 과정에서 성능을 점검하며 하이퍼파라미터(학습률, 배치 크기 등)를 튜닝하며 10%를 테스트 세트로 최종 학습 완료 후 모델이 실제 새로운 데이터에서 얼마나 잘 작동하는지 평가합니다. 장점은 검증 세트를 통해 학습 중 과적합(overfitting)을 실시간으로 감지하고 방지할 수 있으며, 테스트 세트로 새로운 데이터에 대한 성능을 신뢰성 있게 측정할 수 있다는 점으로, 모델의 일반화 능력을 보장하는 표준 방법입니다. 역전파(Back Propagation)는 신경망이 올바른 예측을 하도록 가중치(W)와 편향(b)을 조정하는 핵심 알고리즘으로 모델 학습의 근간을 이루며, 순전파(Forward Propagation)에서 입력 데이터가 네트워크를 통해 순차적으로 전달되어 최종 예측값이 출력되고 각 레이어는 이전 레이어의 출력을 입력으로 받아 변환한 후, 오차 계산에서 예측값과 실제 정답값 간의 차이를 손실 함수(Loss Function)로 측정하며 수어 인식처럼 다중 클래스 문제에서는 크로스 엔트로피 함수를 주로 사용하고, 미분 계산(체인 룰, Chain Rule)에서 손실이 각 가중치와 편향에 미치는 영향을 미분으로 계산하며 체인 룰을 통해 네트워크의 깊은 레이어들 간 영향을 효율적으로 전달하고, 가중치 업데이트에서 계산된 미분값을 바탕으로 가중치를 조정합니다. 수식은 W ← W - η ∂L/∂W (η: 학습률)로, η(이타, eta)는 학습률(learning rate)로 한 번 업데이트에서 가중치를 얼마나 크게 변경할지 결정하는 하이퍼파라미터이며 편향 b도 동일한 방식으로 업데이트되며, 이 과정을 한 에포크(epoch) 동안 모든 데이터에 대해 반복하며 여러 에포크를 거쳐 모델이 점차 정확해집니다. 크로스 엔트로피 손실 함수는 다중 클래스 분류 문제(수어 동작처럼 26개 이상 클래스)에서 사용되는 표준 손실 함수로 예측 확률 분포와 실제 정답 분포 간 차이를 측정하며 수식은 L = -1/N ∑_{i=1}^N ∑_{j=1}^K y_{i,j} log(ŷ_{i,j})로, N은 전체 데이터 샘플 개수로 전체 데이터에 대한 손실을 평균 내어 한눈에 비교할 수 있게 하고 K는 클래스 개수(예: 한국 수어문자 26개 알파벳)로 각 클래스에 대한 확률을 계산하며 y_{i,j}는 실제 정답 레이블로 원-핫 인코딩(One-Hot Encoding) 방식으로 표현되며 정답 클래스만 1이고 나머지는 0으로 예를 들어 3개 클래스 중 두 번째가 정답이면 y = [0, 1, 0]입니다. ŷ_{i,j}는 모델의 예측 확률로 소프트맥스(Softmax) 함수로 출력되며 0~1 사이 값이고 모든 클래스 확률의 합은 1이며 예측이 좋으면 정답 클래스의 ŷ가 1에 가깝습니다. 로그 함수의 역할은 예측이 틀릴수록(정답 클래스 확률이 낮을수록) 로그 값이 크게 음수로 떨어져 손실이 급격히 증가하며 이는 모델이 잘못된 예측에 대해 강한 페널티를 부여하여 정확한 분류를 유도합니다. 예시로 3개 클래스 중 두 번째가 정답(y = [0, 1, 0])일 때 모델 예측이 [0.1, 0.8, 0.1]이면 손실은 작지만 [0.9, 0.05, 0.05]이면 손실이 매우 큽니다. 이는 모델이 정답 클래스를 낮게 예측한 것을 강하게 벌주는 메커니즘입니다. 업데이트 식 상세로 η(학습률)은 모델이 한 번 업데이트에서 가중치를 얼마나 크게 수정할지 결정하는 핵심 하이퍼파라미터로 너무 크면 학습이 불안정해지고(발산) 너무 작으면 학습이 느려지며, 미분값 ∂L/∂W에 η를 곱해 가중치와 편향을 조금씩 조정하며 이 과정을 반복하여 손실을 최소화합니다. 수어 인식처럼 복잡한 패턴에서는 Adam 옵티마이저와 함께 사용되어 적응형 학습률을 제공합니다. Early Stopping 콜백은 학습 과정에서 검증 손실(validation loss)이 더 이상 개선되지 않으면 학습을 자동으로 중단하는 기법으로 과적합을 방지하고 계산 효율을 높이며, from keras.callbacks import EarlyStopping으로 불러온 후 early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)로 설정합니다. 여기서 monitor='val_loss'는 모니터링할 지표로 검증 손실(val_loss)을 선택하고 patience=3은 연속으로 3 에포크 동안 개선이 없으면 중단하며 restore_best_weights=True는 학습 중단 시 최고 성능의 가중치를 복원하고 verbose=1은 중단 시 메시지를 출력합니다. 모델 학습 시 history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping])으로 콜백을 적용하며, 각 에포크 끝에 val_loss를 계산하고 이전 최고 val_loss보다 개선되지 않으면 patience 카운터를 증가시키며 patience 횟수만큼 연속으로 개선이 없으면 학습을 멈추고 restore_best_weights=True로 인해 가장 낮은 val_loss를 기록한 에포크의 가중치로 모델을 복원합니다. 수어 인식처럼 데이터가 제한적인 경우 이 기법으로 불필요한 에포크를 줄여 최적의 일반화 성능을 확보하며, 결과적으로 val_loss가 3회 연속 개선되지 않으면 학습이 중단되고 최고 성능 상태로 복원됩니다. ELU 활성화 함수는 비선형 변환 함수로 입력값에 따라 출력 신호를 조절하며 ReLU의 단점을 보완하며 수식은 f(x) = x (x ≥ 0: 양수 영역은 선형적으로 그대로 출력)와 f(x) = α(e^x - 1) (x < 0: 음수 영역은 지수 함수로 부드럽게 음수 값 변환, α는 보통 1)로 구성됩니다. ReLU vs ELU 상세 비교에서 ReLU의 수식은 f(x) = max(0, x)로 입력 x가 0 이상이면 x 그대로 0 이하면 0으로 출력되며 양수 영역 처리는 입력값을 그대로 출력하여 선형성을 유지하고 계산이 빠르지만 음수 영역 처리에서 모든 음수 입력을 0으로 강제 차단해 해당 뉴런의 기여도가 완전히 사라지며 주요 문제점은 죽은 뉴런(Dying ReLU) 문제로 음수 입력이 지속되면 출력이 항상 0이 되어 해당 뉴런이 더 이상 학습되지 않고 가중치 업데이트가 멈추는 현상입니다. 학습 효과는 초기 학습은 빠르지만 깊은 네트워크에서 음수 입력이 쌓이면 일부 뉴런이 "죽어" 전체 성능 저하가 발생합니다. 반면 ELU의 수식은 f(x) = x (x ≥ 0)와 f(x) = α(e^x - 1) (x < 0)로 음수 영역에서 0에 수렴하는 부드러운 곡선이며 양수 영역 처리는 입력값을 그대로 출력하여 ReLU와 동일한 효율성을 보이고 음수 영역 처리에서 음수 입력을 부드럽게 음수 값으로 변환(최대 -α까지)해 뉴런이 여전히 약간의 신호를 전달하며 주요 문제점은 그래디언트 소실(Gradient Vanishing) 완화로 음수 영역에서도 미분값이 0이 되지 않아 학습이 지속됩니다. 학습 효과는 학습이 더 안정적이고 빠른 수렴(Convergence)을 보이며 음수 입력에서도 뉴런이 활성화되어 네트워크 전체가 균형 있게 학습합니다. 죽은 뉴런 문제 상세로 ReLU에서 입력이 지속적으로 음수가 되면 출력이 항상 0이 되어 해당 뉴런의 가중치가 업데이트되지 않으며 이는 특히 초기 가중치가 불균형하거나 데이터 분포가 음수 편향일 때 발생하며 네트워크의 20-30% 뉴런이 죽을 수 있습니다. ELU는 음수 영역에서 e^x - 1으로 약간의 음수 출력을 허용하여 뉴런이 완전히 비활성화되지 않게 합니다. 그래디언트 소실 문제 상세로 깊은 신경망에서 역전파 시 기울기(gradient)가 레이어를 거칠수록 0에 가까워져 앞단 레이어의 학습이 정체되지만 ELU는 음수 영역의 미분값이 α e^x로 항상 양수이므로 소실을 줄이고 결과적으로 CNN+GRU 같은 복합 구조에서 더 안정적인 학습을 가능하게 하며 수어 인식처럼 시퀀스 데이터에서 ELU는 GRU 레이어의 시간적 의존성을 더 잘 학습시킵니다. 핵심 평가 지표는 모델 성능을 정량적으로 측정하는 지표들로 수어 인식처럼 클래스 불균형이 있을 수 있는 문제에 적합하며 Accuracy(정확도)는 전체 데이터 샘플 중 모델이 올바르게 예측한 비율로 수식은 (TP + TN)/(TP + TN + FP + FN) (TP: True Positive, TN: True Negative, FP: False Positive, FN: False Negative)이며 전체 100개 수어 동작 중 90개를 맞히면 90%지만 희귀 동작(예: 특정 알파벳)이 많으면 왜곡될 수 있습니다. Precision(정밀도)은 모델이 "양성(True)"으로 예측한 샘플 중 실제로 양성인 비율로 수식은 TP/(TP + FP)이며 모델이 "A 동작"으로 예측한 100개 중 실제 A가 95개라면 95%로 오분류(다른 동작을 A로 잘못 예측) 최소화 측정을 합니다. Recall(재현율, 민감도)은 실제 양성 샘플 중 모델이 양성으로 올바르게 예측한 비율로 수식은 TP/(TP + FN)이며 실제 A 동작 100개 중 80개를 A로 예측했다면 80%로 놓친 동작(FN) 최소화하며 실시간 번역에서 중요합니다. F1-Score(F1 점수)는 정밀도와 재현율의 조화 평균으로 클래스 불균형 시 유용하며 수식은 2 × (Precision × Recall)/(Precision + Recall)로 Precision 90% Recall 80%라면 F1 = 84.6%로 두 지표의 균형을 고려하여 전체 성능을 평가합니다. 혼동 행렬(Confusion Matrix)은 클래스별 실제 레이블 vs 예측 레이블을 2D 테이블로 나타내며 대각선 값(TP)이 클수록 정확한 분류가 많고 예를 들어 26개 수어문자 클래스에서 대각선이 강하면 대부분의 알파벳이 정확히 분류된 것을 의미하며 오분류(비대각선)가 적어 모델의 신뢰성이 높습니다. 분석 시 유사 손 모양(A와 E처럼) 간 혼동 패턴을 식별하여 추가 학습 데이터를 보강할 수 있으며 수어 인식에서 대부분 대각선 값이 크다는 것은 오분류 사례가 적어 실용적 성능이 우수함을 나타냅니다. 고려사항인 서명자 의존성 평가(Speaker Dependency)는 모델이 학습하지 않은 새로운 수화자의 데이터에서 정확도가 변동되는 현상으로 개인화된 수어 스타일 차이에서 발생하며 주요 원인으로 손 우세성(Handedness)이 있으며 테스트자 10명 중 1명이 왼손잡이로 오른손 중심 데이터로 학습된 모델이 왼손 동작을 제대로 인식하지 못합니다. 문제 영향으로 오른손잡이 데이터로 학습 시 왼손잡이 정확도가 10-15% 하락하며 수어는 주로 우세 손으로 표현되므로 이 차이가 성능 저하를 초래합니다. 해결책으로 데이터 증강(Augmentation) 기법 적용 - 왼손잡이 데이터를 수평으로 뒤집어 오른손 데이터처럼 변환하여 학습시키고 추가로 회전 크기 조정 등의 변형을 적용하면 모델의 견고성이 향상되며 효과로 증강 후 왼손잡이 정확도가 76%에서 87%로 개선되며 전체 화자 독립성(모든 사용자에 대한 일반화)이 강화됩니다. 절대 프레임 차이(Absolute Frame Difference)는 영상 처리에서 두 연속 프레임 간 변화(움직임)를 감지하는 특징 추출 방법으로 수어 동작의 시간적 변화를 수치화하며 계산 수식은 D(t) = ∑_{i,j} |I_t(i,j) - I_{t-1}(i,j)|로 I_t는 t번째 프레임의 픽셀 값(밝기 색상 등)이고 I_{t-1}은 t-1번째 프레임의 픽셀 값이며 (i,j)는 이미지의 각 픽셀 위치이고 | · |은 절댓값으로 차이의 방향 무시하고 크기만 측정합니다. 작동 원리로 손이 움직이면 해당 영역의 픽셀 값(위치 모양)이 변하므로 프레임 간 차이가 커지며 예를 들어 정지 상태에서 차이는 거의 0이지만 손 제스처(예: A 동작) 시 차이값이 급증합니다. 이 값을 모델 입력 특징으로 추가하면 움직임 강도와 속도를 반영하여 동작 인식 정확도가 향상되며 활용 예시로 연속된 30프레임 시퀀스에서 각 쌍의 차이를 계산해 "움직임 벡터"를 생성하고 이는 키포인트 좌표와 결합되어 CNN 입력으로 사용되며 변화 추적(동작 시작/끝 감지)과 수화 동작 인식에 효과적입니다. 결론적으로 영상 전후 프레임 간 실제 변화를 정량적으로 계산하여 모델의 시간적 이해를 보강합니다. 개념 상세 텍스트 설명으로 역전파 흐름은 순전파 단계에서 입력 데이터(키포인트 시퀀스)가 CNN → GRU → 출력 레이어를 거쳐 예측값이 생성되고 오차 계산 후 역전파에서 손실이 출력 레이어부터 입력 레이어로 거꾸로 전달되며 각 레이어의 가중치가 체인 룰로 미분되어 업데이트되는 반복으로 모델이 수어 패턴을 점차 학습합니다. ELU vs ReLU 텍스트 설명으로 ReLU는 x < 0일 때 출력이 0으로 고정되어 뉴런이 "죽지만" ELU는 x < 0에서도 약간의 음수 출력을 허용하여 뉴런이 지속적으로 기여하며 그래프로 보면 ReLU는 0에서 꺾이는 직선 ELU는 음수 영역에서 지수 곡선으로 부드럽게 내려가는 모양입니다. 혼동 행렬 텍스트 예시로 5x5 행렬에서 행은 실제 클래스(A,B,C,D,E) 열은 예측 클래스이며 대각선(A-A, B-B 등)이 90% 이상 크면 정확도가 높고 A-B 위치의 값이 작으면 유사 동작 간 혼동이 적습니다. 예: A 실제 100개 중 92개 A 예측 8개 B로 오분류입니다.

---

## 📚 개발 커리큘럼 (정리된 단계별 계획)

데이터 준비 및 전처리 단계에서는 AI Hub 수어 데이터셋 구하기(5,000+ 영상, 회원가입 필요)로 시작해 MediaPipe로 프레임별 키포인트 추출(21개 손 랜드마크, 신뢰도 0.5 이상 필터링)을 하고 데이터 증강으로 왼손잡이 데이터 수평 반전 회전/크기 변형 등을 통해 손 우세성 보완하며 데이터셋 분할로 홀드아웃 방식 80% 훈련 / 10% 검증 / 10% 테스트를 적용하며 주요 도구/기법은 MediaPipe OpenCV NumPy입니다. 모델 구현 단계에서는 CNN+GRU 하이브리드 설계로 CNN으로 각 프레임의 공간 특징(손 모양) 추출 GRU로 시간 연속성(동작 흐름) 처리하고 ELU 활성화 함수 적용으로 그래디언트 소실 완화와 죽은 뉴런 방지 크로스 엔트로피 손실 함수와 역전파 알고리즘 구현으로 다중 클래스 분류 최적화 Early Stopping 콜백 적용으로 과적합 방지하며 주요 도구/기법은 TensorFlow/Keras Adam Optimizer입니다. 모델 평가 및 분석 단계에서는 지표 산출로 Accuracy Precision Recall F1-Score를 통해 전체/클래스별 성능 측정하고 혼동 행렬 생성으로 클래스 오분류 패턴 파악(대각선 강하면 정확 분류 많음 비대각선 약하면 오분류 적음) 화자 독립성 평가로 학습하지 않은 다양한 수화자 데이터로 검증 손 우세성(왼/오른손) 영향 분석 절대 프레임 차이 특징 도입으로 프레임 간 픽셀 변화 수치화로 움직임 강도 추가 학습을 하며 주요 도구/기법은 Scikit-learn Matplotlib입니다. 성능 최적화 및 배포 단계에서는 하이퍼파라미터 튜닝으로 학습률 배치 크기 GRU 유닛 수 최적화(Grid Search 또는 Bayesian Optimization) 모델 경량화로 TensorFlow Lite 변환으로 모바일/엣지 디바이스 배포 준비 다양한 환경 테스트로 조명 변화 배경 노이즈 카메라 기기 차이 견고성 확인 실시간 예측 시스템 구축으로 웹캠 입력 → 0.1초 내 처리 목표를 세우며 주요 도구/기법은 TensorFlow Lite Flask/Django입니다. 응용 및 확장 단계에서는 음성 변환 연계로 인식된 수어를 TTS(텍스트-음성)로 변환 텍스트 변환 기능으로 실시간 자막 출력과 문장 완성 사용자 인터페이스 개발로 웹/모바일 앱으로 확장팩 구현(예: 수어 학습 모드 맞춤 사전 추가)을 하며 주요 도구/기법은 Streamlit React Native입니다. 모델 구현 예시 코드로 CNN + GRU 하이브리드를 import tensorflow as tf from tensorflow.keras.layers import Input TimeDistributed Conv1D MaxPooling1D Flatten GRU Dense Dropout from tensorflow.keras.models import Model from tensorflow.keras.callbacks import EarlyStopping으로 불러와 def create_cnn_gru_model(input_shape=(30, 42, 2), num_classes=26): inputs = Input(shape=input_shape) x = TimeDistributed(Conv1D(filters=64, kernel_size=3, activation='elu', padding='same'))(inputs) x = TimeDistributed(MaxPooling1D(pool_size=2))(x) x = TimeDistributed(Flatten())(x) x = GRU(units=128, return_sequences=False)(x) x = Dense(64, activation='elu')(x) x = Dropout(0.5)(x) outputs = Dense(num_classes, activation='softmax')(x) model = Model(inputs, outputs) model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy']) return model로 정의하고 model = create_cnn_gru_model() early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True) history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32, callbacks=[early_stopping])로 생성 및 학습합니다. 여기서 입력은 (배치, 시퀀스 길이 30, 키포인트 42, 좌표 2)이고 CNN은 TimeDistributed로 각 프레임별 특징 추출(공간 패턴 학습) GRU는 시간적 패턴 학습(프레임 간 연속성 처리) 출력은 수어 클래스 확률(소프트맥스)이며 컴파일은 적응형 학습률의 adam과 크로스 엔트로피 손실 정확도 추적으로 홀드아웃 데이터 80/10/10 분할로 학습하며 조기 중단을 적용합니다. 평가 예시 코드로 혼동 행렬 및 지표를 from sklearn.metrics import confusion_matrix classification_report import numpy as np import matplotlib.pyplot as plt import seaborn as sns로 불러와 y_pred = model.predict(X_test) y_pred_classes = np.argmax(y_pred, axis=1)로 테스트 데이터 예측하고 print(classification_report(y_test, y_pred_classes, target_names=['A', 'B', 'C', ..., 'Z']))로 분류 보고서 출력(Precision Recall F1-Score) cm = confusion_matrix(y_test, y_pred_classes) plt.figure(figsize=(10, 8)) sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['A', 'B', ..., 'Z'], yticklabels=['A', 'B', ..., 'Z']) plt.title('Confusion Matrix - 대각선 값이 크면 정확 분류가 많고 오분류가 적음') plt.ylabel('실제 클래스 (True Label)') plt.xlabel('예측 클래스 (Predicted Label)') plt.show()로 혼동 행렬 생성 및 시각화하며 분석으로 대각선(예: A-A 위치)이 90% 이상이면 해당 클래스 정확도가 높고 비대각선(예: A-B)이 작으면 유사 동작 간 혼동이 적습니다.

---

## 🛠️ 기술 스택

| 카테고리 | 기술/라이브러리 | 버전 | 주요 용도 |
|----------|-----------------|------|-----------|
| **프로그래밍 언어** | Python | 3.10+ | 전체 개발 및 스크립팅 |
| **딥러닝 프레임워크** | TensorFlow | 2.15+ | 모델 빌드 학습 배포 |
| **고수준 API** | Keras | 3.0 | 레이어 정의와 컴파일 간소화 |
| **키포인트 추출** | MediaPipe | 0.10+ | 실시간 손 랜드마크(21개) 검출 |
| **데이터 처리** | NumPy | 1.24+ | 배열 연산과 키포인트 변환 |
| **데이터 관리** | Pandas | 2.0+ | CSV/JSON 데이터 로딩과 분할 |
| **시각화 및 분석** | Matplotlib | 3.7+ | 학습 곡선과 혼동 행렬 플롯 |
| **통계/평가** | Seaborn | 0.13+ | 히트맵과 지표 시각화 |
| **머신러닝 유틸** | Scikit-learn | 1.3+ | 분류 보고서와 혼동 행렬 계산 |
| **개발 환경** | Jupyter Notebook | 1.0+ | 실험과 데이터 탐색 |
| **버전 관리** | Git/GitHub | 최신 | 코드 공유와 협업 |

## 📁 프로젝트 구조

cnn-gru-sign-language/ 아래 data/는 원본 및 처리된 데이터로 raw/는 AI Hub 다운로드 영상 파일 keypoints/는 MediaPipe로 추출된 JSON/CSV 키포인트 좌표 processed/는 홀드아웃 분할 완료(train/val/test 폴더) models/는 학습된 모델 파일로 best_model.keras는 Early Stopping으로 복원된 최고 성능 모델 checkpoints/는 에포크별 중간 저장 파일 src/는 소스 코드로 data_loader.py는 데이터 로딩 증강 홀드아웃 분할 model.py는 CNN+GRU[ELU] 모델 정의 함수 train.py는 학습 스크립트(역전파 콜백 포함) evaluate.py는 지표 계산과 혼동 행렬 분석 utils.py는 유틸리티(절대 프레임 차이 계산 등) preprocess.py는 키포인트 추출과 전처리 notebooks/는 Jupyter 노트북(탐색용)으로 01_data_exploration.ipynb는 데이터셋 분석과 통계 02_model_experiment.ipynb는 아키텍처 실험과 ELU 테스트 03_results_analysis.ipynb는 성능 지표와 화자 독립성 평가 results/는 학습 결과 파일로 loss_history.csv는 에포크별 손실/정확도 기록 predictions.csv는 테스트 예측 결과 metrics/는 Precision/Recall/F1-Score 상세 로그 demo/는 데모 스크립트로 realtime_translation.py는 웹캠 실시간 수어 인식입니다.

---

## 🚀 설치 및 실행

요구사항으로 Python 3.10+ GPU(CUDA 11.8+ 권장)를 설치하고 git clone https://github.com/yourusername/cnn-gru-sign-language.git cd cnn-gru-sign-language pip install -r requirements.txt (tensorflow mediapipe scikit-learn 등)으로 환경을 설정합니다. 데이터 준비로 python src/preprocess.py --videos data/raw/ --output data/keypoints/로 키포인트 추출(MediaPipe) python src/data_loader.py --data_dir data/keypoints/ --split으로 분할(홀드아웃 80/10/10)을 합니다. 학습 및 평가로 python src/train.py --epochs 100 --patience 3로 학습(ELU + Early Stopping) python src/evaluate.py --model models/best.keras --test data/processed/test/로 평가(지표 + 혼동 행렬)를 수행합니다. 실시간 데모로 python demo/realtime.py --model models/best.keras (웹캠 입력 → 텍스트 출력)를 실행합니다.

---

## 📈 예상 성능

메트릭으로 Accuracy ≥ 90% (GRU 시간 패턴 강화) F1-Score ≥ 0.88 (클래스 불균형 고려) Precision/Recall ≥ 0.87/0.87 (손 우세성 증강 후) 추론 시간 ≤ 0.1초 (ELU 안정 학습)를 목표로 하며 예상 학습 곡선으로 val_loss 3회 정체 시 Early Stopping 적용을 통해 과적합을 방지합니다.

---

## 🔮 미래 계획 및 응용

최적화로 하이퍼파라미터 튜닝 경량화(TensorFlow Lite) 테스트로 조명/배경/기기 변화 견고성 응용으로 실시간 번역(텍스트/음성 출력) 교육 앱 의료 통역 확장으로 오픈소스 배포 논문 작성입니다.

## 📚 참고 자료

AI Hub 수어 데이터셋 [aihub.or.kr] MediaPipe Hands [mediapipe.dev] ELU 논문 [arxiv.org/abs/1511.07289] 유사 프로젝트: CNN 수어 인식 [github.com/kaushikrohit004] [web:178]입니다.

## 🤝 기여 방법

Fork 후 브랜치 생성 데이터/코드 개선 PR 환영 (Issues 먼저) 버그는 Issues에 재현 코드 + 환경 설명입니다.

## 📄 라이선스

MIT License 자유 사용/수정 가능 (저작자 표기) [LICENSE] 참조입니다.

---
