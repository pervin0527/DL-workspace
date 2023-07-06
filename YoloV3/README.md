
앵커 박스(Anchor Box)와 바운딩 박스 회귀(Bounding Box Regression)는 객체 탐지(Object Detection) 작업에서 사용되는 중요한 개념입니다. 앵커 박스는 이미지 내에서 객체의 위치와 크기를 대략적으로 예측하기 위한 사전 정의된 상자이며, 바운딩 박스 회귀는 앵커 박스를 이용하여 실제 객체의 위치와 크기를 보정하는 과정입니다.

아래는 앵커 박스와 바운딩 박스 회귀 학습의 전체적인 과정입니다.

데이터 준비:

학습 데이터셋에는 입력 이미지와 해당 이미지의 객체들의 바운딩 박스 정보가 필요합니다.
바운딩 박스 정보는 객체의 클래스 레이블과 상자의 좌표(x, y, width, height)로 구성됩니다.
학습 데이터셋을 불러와서 객체의 클래스 레이블과 바운딩 박스 정보를 추출합니다.
앵커 박스 생성:

입력 이미지를 사용하여 앵커 박스를 생성합니다.
앵커 박스는 이미지를 격자 형태로 분할하고, 각 격자 셀마다 여러 개의 앵커 박스를 생성합니다.
앵커 박스는 다양한 크기와 종횡비를 가지며, 객체의 크기와 비율을 대표할 수 있는 다양한 형태를 포함합니다.
훈련 데이터 생성:

입력 이미지의 각 앵커 박스와 실제 객체 사이의 IoU(Intersection over Union)를 계산합니다.
IoU가 높은 앵커 박스와 해당 객체를 매칭시킵니다.
매칭된 앵커 박스와 객체의 바운딩 박스 정보를 이용하여 훈련 데이터를 생성합니다.
훈련 데이터에는 앵커 박스와 객체의 상대적인 위치와 크기 정보, 그리고 객체의 클래스 레이블이 포함됩니다.
모델 학습:

생성된 훈련 데이터를 사용하여 앵커 박스와 바운딩 박스 회귀를 학습하는 모델을 구축합니다.
일반적으로 합성곱 신경망(CNN) 기반의 네트워크를 사용합니다.
네트워크의 출력은 앵커 박스와 바운딩 박스의 상대적인 변화를 예측하는 회귀 값입니다.
테스트 및 예측:

훈련된 모델을 사용하여 입력 이미지에서 객체를 탐지합니다.
입력 이미지를 격자로 분할하고, 각 격자 셀에 대해 앵커 박스와 객체의 클래스 및 바운딩 박스 정보를 예측합니다.
예측된 앵커 박스와 바운딩 박스를 이용하여 객체의 위치와 크기를 보정합니다.

Anchor Box and Ground-Truth Matching:

Calculate the IoU between each anchor box and ground-truth boxes.
If an anchor box has an IoU higher than a predefined threshold with any ground-truth box, it is assigned a positive label.
If an anchor box has an IoU lower than another predefined threshold with all ground-truth boxes, it is assigned a negative label.
Bounding Box Regression:

For each positive anchor box, the bounding box regression learns to minimize the offset between the predicted anchor box coordinates and the corresponding ground-truth box coordinates.
The regression process adjusts the positive anchor box predictions to better match the ground-truth box.

Here's the correct process during inference:

Generate Anchor Boxes:

Predefined anchor boxes are generated based on different scales and aspect ratios.
Run Inference:

Pass the input image through the trained model.
For each anchor box, the model predicts the class probabilities for the object classes and the regression offsets.
Decode Predictions:

Apply the predicted regression offsets to the anchor boxes to obtain the adjusted predicted bounding boxes.
Filter and Post-process Predictions:

Apply a confidence threshold to filter out low-confidence predictions.
Perform non-maximum suppression (NMS) to remove overlapping predictions.