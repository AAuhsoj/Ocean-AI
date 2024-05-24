import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 텐서 값 추출
x_center, y_center, width, height = tensor[0]

# 직사각형 생성
rectangle = patches.Rectangle((x_center - width/2, y_center - height/2), width, height, linewidth=1, edgecolor='r', facecolor='none')

# 시각화
fig, ax = plt.subplots(1)
ax.add_patch(rectangle)
plt.xlim(0, 2000)  # X축 범위 설정
plt.ylim(0, 1000)  # Y축 범위 설정
plt.gca().invert_yaxis()  # Y축을 아래쪽이 아닌 위쪽이 0인 방향으로 변경
plt.show()
