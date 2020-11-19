torch-model-archiver \
--model-name img2caption \
--version 1.0 \
--serialized-file models/image-caption/NIC/model.tar \
--export-path model_store \
--extra-files handlers/i2c_handler.py,models/image-caption/NIC/word_map.json,models/image-caption/NIC/models.py \
--handler handlers/i2c_handler.py  \
-f