import keras

from dataset_loading import load_raw_tiny_image_net_from_files


def draft_function():
    model = keras.applications.resnet50.ResNet50(weights="imagenet", include_top=True)
    #                                                 , input_tensor=resizer)
    decode_predictions = keras.applications.resnet50.decode_predictions
    raw_generator_test, val_classes = load_raw_tiny_image_net_from_files(
        preprocess=keras.applications.resnet50.preprocess_input)
    # print(list(zip(model.metrics_names, model.evaluate_generator(test_generator, workers=6))))

    y_pred = model.predict_generator(raw_generator_test, workers=6)
    preds = decode_predictions(y_pred, top=1)
    return preds, val_classes
