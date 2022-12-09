import tensorflow as tf
import numpy as np


class AksaraJawaModels:
    AksaraJawa = [
        'carakan_ba',
        'carakan_ca',
        'carakan_da',
        'carakan_dha',
        'carakan_ga',
        'carakan_ha',
        'carakan_ja',
        'carakan_ka',
        'carakan_la',
        'carakan_ma',
        'carakan_na',
        'carakan_nga',
        'carakan_nya',
        'carakan_pa',
        'carakan_ra',
        'carakan_sa',
        'carakan_ta',
        'carakan_tha',
        'carakan_wa',
        'carakan_ya',
        'taling_ba',
        'taling_ca',
        'taling_da',
        'taling_dha',
        'taling_ga',
        'taling_ha',
        'taling_ja',
        'taling_ka',
        'taling_la',
        'taling_ma',
        'taling_na',
        'taling_nga',
        'taling_nya',
        'taling_pa',
        'taling_ra',
        'taling_sa',
        'taling_ta',
        'taling_tha',
        'taling_wa',
        'taling_ya'
    ]

    def __init__(self):
        self.aksaraJawaModel = tf.keras.models.load_model(
            'saved_model/aksara_jawa')

    def aksara_predict(self, images):
        classes = self.aksaraJawaModel.predict(images, batch_size=32)
        predicted_class_indices = np.argmax(classes)
        print(classes)
        return {
            "code": 200,
            "message": "Berhasil diklasifikasi",
            "data": {
                "confidence": classes.tolist()[0][predicted_class_indices],
                "label": self.AksaraJawa[predicted_class_indices]
            }
        }
