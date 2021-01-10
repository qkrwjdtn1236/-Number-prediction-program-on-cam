import tensorflow as tf


mnist = tf.keras.datasets.mnist # 숫자  mnist 데이터셋

(x_train, y_train),(x_test, y_test) = mnist.load_data() #파일 불러옴
x_train, x_test = x_train / 255.0, x_test / 255.0#데이터 전처리

model = tf.keras.models.Sequential([ #모델 설계
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
# 옵티마이저 설정
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#훈련
model.fit(x_train, y_train, epochs=5)
#모의 테스트
model.evaluate(x_test, y_test)
#모델 세이브
model.save_weights('mnist_checkpoint')