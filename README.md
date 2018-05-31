# QuijoteNet
#### Red neuronal que analiza el texto de El Quijote y capaz de generar textos a partir de una pequeña cadena de caracteres.
La IA es una de las áreas más novedosas y desconocidas en gran parte de España y, en general, del mundo. Para cualquiera que esté buscando cursos de Inteligencia Artificial y redes neuronales será muy difícil encontrar un tutorial o curso motivante y que muestre avance a corto plazo en español, ya que la mayoría de estos se encuentran en inglés y muchas veces dejan muchas partes sin clarificar.

Por ello, me he propuesto intentar portar uno de estos tutoriales en inglés al español, de forma que sea fácil de entender y no conlleve muchos términos y variables complejas.

# ¿Qué es QuijoteNet?
QuijoteNet es un prototipo de **red neuronal recurrente** (RNN) que analiza un fragmento de la obra *"El Quijote"*, de Miguel de Cervantes, una de las obras más representativas de la literatura española, dividiéndolo por enunciados, que a su vez se dividen carácter por carácter.

# ¿Cómo funciona?
QuijoteNet usa una "red neuronal" formada por varias capas de "bloques" o "núcleos" **LSTM** ("Redes de gran memoria a corto plazo").
Al poseer al menos una capa oculta, podemos precisar que nos encontramos ante un ejemplo de **aprendizaje profundo**.

La red neuronal analiza los patrones del lenguaje que existen en el texto, es decir, la forma en la que se juntan las letras para formar grupos lingüísticos más complejos. Obviamente esta no es una tarea precisamente fácil y menos para un ordenador, por eso es preferible que este ejemplo se ejecute en una tarjeta gráfica (esto puede suponer un tiempo de entrenamiento decenas, cientos o incluso miles de veces menor que en un procesador tradicional o CPU).

**IMPORTANTE:** actualmente solamente las gráficas NVIDIA con CUDA Compute Capability superior a 3.0 son compatibles con TensorFlow (comprueba si tu gráfica cumple los requisitos [aquí](https://developer.nvidia.com/cuda-gpus)

# Instalación y requisitos previos

Para poder ejecutar correctamente este ejemplo, es necesario cumplir una serie de requisitos en tu ordenador para entrenar el modelo:
+ Windows 7 o mayor / macOS 10.12.6 o mayor / Ubuntu 16.04 o mayor
+ [Python 3.5 o superior (64-bit)](https://www.python.org/downloads/release/python-362/)
+ Editor de código a elección propia (recomendado [Pycharm](https://www.jetbrains.com/pycharm/) para entornos virtuales y depuración)
+ **Solamente si quieres usar tu GPU:**
	+ [CUDA Toolkit 9.0 (ni superior ni inferior)](https://developer.nvidia.com/cuda-90-download-archive)
	+ cuDNN 7.0 (ni superior ni inferior). Necesitas pedir acceso a los archivos en la [página oficial de NVIDIA](https://developer.nvidia.com/rdp/form/cudnn-download-survey)
+ Numpy (recomendada 1.14.3 o superior)
+ [TensorFlow](https://www.tensorflow.org/install/) en cualquiera de sus variantes (CPU o GPU). Algunos métodos para instalar:
	+ CPU: `pip3 install --upgrade tensorflow`
	+ GPU: `pip3 install --upgrade tensorflow-gpu`
+ **(OPCIONAL)** Librería [Keras](https://keras.io/#installation) independiente. En TensorFlow existe un módulo llamado `tensorflow.contrib.keras` con todo incluido, aunque no implica que no puedas instalar la librería aparte.
